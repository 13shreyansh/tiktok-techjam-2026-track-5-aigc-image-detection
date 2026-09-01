#!/usr/bin/env python3
"""Verify exact saved ensemble scores and resource use on one NVIDIA GPU."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import kaggle_evaluate_frontier_ensemble_promotion as promotion
import kaggle_evaluate_v8_promotion_gates as sealed
import kaggle_stress_eval_v6 as stress
import kaggle_train_v3 as runner
import kaggle_verify_ensemble_fixed_batch as fixed


OUTPUT = Path("/kaggle/working/track5-frontier-ensemble-single-gpu.json")
PHYSICAL_BATCH_SIZE = 64
EXPECTED_ROWS = 576
DEVICE = "cuda:0"


@torch.inference_mode()
def score_single_gpu(
    v6_model: torch.nn.Module,
    v9_model: torch.nn.Module,
    dataset: runner.ManifestDataset,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=PHYSICAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    digest = hashlib.sha256()
    labels: list[int] = []
    indices: list[int] = []
    v6_values: list[float] = []
    blend_values: list[float] = []
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize(0)
    wall_started = time.perf_counter()
    forward_seconds = 0.0
    for images, batch_labels, batch_indices in loader:
        if int(images.shape[0]) != PHYSICAL_BATCH_SIZE:
            raise RuntimeError(f"unexpected final batch: {tuple(images.shape)}")
        digest.update(images.contiguous().numpy().tobytes())
        images = images.to(DEVICE, non_blocking=True)
        torch.cuda.synchronize(0)
        forward_started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            v6_scores = torch.sigmoid(v6_model(images).flatten())
            v9_scores = torch.sigmoid(v9_model(images).flatten())
            blend_scores = (
                promotion.V6_WEIGHT * v6_scores
                + promotion.V9_WEIGHT * v9_scores
            )
        v6_cpu = v6_scores.float().cpu()
        blend_cpu = blend_scores.float().cpu()
        torch.cuda.synchronize(0)
        forward_seconds += time.perf_counter() - forward_started
        labels.extend(int(value) for value in batch_labels.tolist())
        indices.extend(int(value) for value in batch_indices.tolist())
        v6_values.extend(float(value) for value in v6_cpu.tolist())
        blend_values.extend(float(value) for value in blend_cpu.tolist())
    if indices != list(range(len(dataset))):
        raise RuntimeError("single-GPU audit did not preserve manifest order")
    return {
        "count": len(indices),
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "batches": len(loader),
        "input_tensor_sha256": digest.hexdigest(),
        "v6_scores": v6_values,
        "blend_scores": blend_values,
        "v6_auc": float(roc_auc_score(labels, v6_values)),
        "blend_auc": float(roc_auc_score(labels, blend_values)),
        "wall_seconds_including_decode_and_input_hash": time.perf_counter()
        - wall_started,
        "model_forward_seconds": forward_seconds,
        "images_per_forward_second": len(indices) / forward_seconds,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("single-GPU audit requires CUDA")
    sealed_root, sealed_package = sealed.validate_package()
    manifest = sealed_root / sealed.MANIFESTS["qwen_prompt_holdout"]["path"]
    if runner.file_sha256(manifest) != sealed.MANIFESTS["qwen_prompt_holdout"]["sha256"]:
        raise RuntimeError("Qwen holdout manifest SHA-256 mismatch")
    rows = promotion.read_jsonl(manifest)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"Qwen row mismatch: {len(rows)}")

    v6_path = promotion.selected_v6_path()
    v9_path = promotion.V9_ROOT / promotion.MODEL_NAME / "model.pt"
    load_started = time.perf_counter()
    v6_model, v6_checkpoint = promotion.load_model(
        v6_path, promotion.V6_SHA256, DEVICE
    )
    v9_model, v9_checkpoint = promotion.load_model(
        v9_path, promotion.V9_SHA256, DEVICE
    )
    torch.cuda.synchronize(0)
    load_seconds = time.perf_counter() - load_started
    mean = tuple(v6_checkpoint["normalization_mean"])
    std = tuple(v6_checkpoint["normalization_std"])
    if mean != tuple(v9_checkpoint["normalization_mean"]) or std != tuple(
        v9_checkpoint["normalization_std"]
    ):
        raise RuntimeError("v6/v9 normalization mismatch")
    dataset = runner.ManifestDataset(
        manifest, stress.condition_transform(mean, std), rows=rows
    )
    run = score_single_gpu(v6_model, v9_model, dataset)

    if not fixed.SAVED_CLEAN.is_file():
        raise RuntimeError(f"saved predictions missing: {fixed.SAVED_CLEAN}")
    saved = promotion.read_jsonl(fixed.SAVED_CLEAN)
    if len(saved) != len(rows):
        raise RuntimeError("saved prediction count mismatch")
    saved_by_hash = {str(row["image_sha256"]): row for row in saved}
    if any(str(row["image_sha256"]) not in saved_by_hash for row in rows):
        raise RuntimeError("manifest row absent from saved predictions")
    labels = [int(row["label"]) for row in rows]
    comparisons = {}
    for key, saved_key in (("v6", "v6_score"), ("blend", "score")):
        saved_scores = [
            float(saved_by_hash[str(row["image_sha256"])][saved_key])
            for row in rows
        ]
        comparisons[key] = fixed.compare_scores(
            saved_scores,
            run[f"{key}_scores"],
            float(roc_auc_score(labels, saved_scores)),
            run[f"{key}_auc"],
        )
    exact = all(
        values["max_absolute_score_drift"] == 0.0
        and values["auc_drift"] == 0.0
        and values["max_rank_displacement"] == 0
        for values in comparisons.values()
    )
    report = {
        "completed": True,
        "scope": "complete already-open 576-image clean Qwen holdout",
        "arithmetic_contract": (
            "one GPU; physical batch 64; sigmoid and 75/25 blend in FP16; "
            "FP32 conversion only after blend"
        ),
        "rows": len(rows),
        "manifest_sha256": runner.file_sha256(manifest),
        "promotion_package_inventory_sha256": sealed_package["inventory_sha256"],
        "v6_checkpoint_sha256": promotion.V6_SHA256,
        "v9_checkpoint_sha256": promotion.V9_SHA256,
        "checkpoint_bytes": {
            "v6": v6_path.stat().st_size,
            "v9": v9_path.stat().st_size,
        },
        "load_seconds": load_seconds,
        "saved_prediction_comparison": comparisons,
        "run": run,
        "passes": exact,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    del report["run"]["v6_scores"]
    del report["run"]["blend_scores"]
    fixed.atomic_json(OUTPUT, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
