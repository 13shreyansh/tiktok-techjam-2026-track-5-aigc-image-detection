#!/usr/bin/env python3
"""Verify paired-ensemble numerical stability across inference batch sizes.

This is a feasibility audit after statistical promotion. It opens no new data,
changes no model or blend weight, and scores only the already-open clean Qwen
holdout. The transformed CPU tensors are hashed so batch-size comparisons cannot
silently use different pixels.
"""

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


OUTPUT = Path("/kaggle/working/track5-frontier-ensemble-batch-stability.json")
SAVED_CLEAN = (
    promotion.OUTPUT_ROOT / "qwen_prompt_holdout/clean_predictions.jsonl"
)
BATCH_SIZES = (64, 128)
MAX_ABSOLUTE_SCORE_DRIFT = 1e-4
MAX_AUC_DRIFT = 1e-6
MAX_RANK_DISPLACEMENT = 1


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def rank_positions(values: np.ndarray) -> np.ndarray:
    order = np.lexsort((np.arange(len(values)), values))
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values), dtype=np.int64)
    return ranks


@torch.inference_mode()
def score_pair(
    v6_model: torch.nn.Module,
    v9_model: torch.nn.Module,
    dataset: runner.ManifestDataset,
    batch_size: int,
) -> dict:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    input_digest = hashlib.sha256()
    labels: list[int] = []
    indices: list[int] = []
    v6_values: list[float] = []
    blend_values: list[float] = []
    for device_index in (0, 1):
        torch.cuda.reset_peak_memory_stats(device_index)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    forward_seconds = 0.0
    wall_started = time.perf_counter()
    for images, batch_labels, batch_indices in loader:
        input_digest.update(images.contiguous().numpy().tobytes())
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        forward_started = time.perf_counter()
        images_v6 = images.to("cuda:0", non_blocking=True)
        images_v9 = images.to("cuda:1", non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            v6_scores = torch.sigmoid(v6_model(images_v6).flatten())
            v9_scores = torch.sigmoid(v9_model(images_v9).flatten())
        v6_cpu = v6_scores.float().cpu()
        v9_cpu = v9_scores.float().cpu()
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        forward_seconds += time.perf_counter() - forward_started
        blend_cpu = (
            promotion.V6_WEIGHT * v6_cpu + promotion.V9_WEIGHT * v9_cpu
        )
        labels.extend(int(value) for value in batch_labels.tolist())
        indices.extend(int(value) for value in batch_indices.tolist())
        v6_values.extend(float(value) for value in v6_cpu.tolist())
        blend_values.extend(float(value) for value in blend_cpu.tolist())
    wall_seconds = time.perf_counter() - wall_started
    if indices != list(range(len(dataset))):
        raise RuntimeError("batch audit did not preserve manifest order")
    return {
        "batch_size": batch_size,
        "count": len(indices),
        "input_tensor_sha256": input_digest.hexdigest(),
        "v6_scores": v6_values,
        "blend_scores": blend_values,
        "v6_auc": float(roc_auc_score(labels, v6_values)),
        "blend_auc": float(roc_auc_score(labels, blend_values)),
        "wall_seconds_including_decode_and_input_hash": wall_seconds,
        "paired_model_forward_seconds": forward_seconds,
        "images_per_forward_second": len(indices) / forward_seconds,
        "cuda_peak_allocated_bytes": {
            str(index): int(torch.cuda.max_memory_allocated(index))
            for index in (0, 1)
        },
    }


def comparison(left: dict, right: dict, key: str) -> dict:
    first = np.asarray(left[f"{key}_scores"], dtype=np.float64)
    second = np.asarray(right[f"{key}_scores"], dtype=np.float64)
    drift = np.abs(first - second)
    rank_drift = np.abs(rank_positions(first) - rank_positions(second))
    return {
        "max_absolute_score_drift": float(drift.max()),
        "mean_absolute_score_drift": float(drift.mean()),
        "auc_drift": float(abs(left[f"{key}_auc"] - right[f"{key}_auc"])),
        "max_rank_displacement": int(rank_drift.max()),
    }


def saved_comparison(current: dict, saved: list[dict], key: str) -> dict:
    saved_key = "v6_score" if key == "v6" else "score"
    saved_scores = np.asarray([float(row[saved_key]) for row in saved])
    current_scores = np.asarray(current[f"{key}_scores"], dtype=np.float64)
    saved_labels = [int(row["label"]) for row in saved]
    drift = np.abs(saved_scores - current_scores)
    saved_auc = float(roc_auc_score(saved_labels, saved_scores))
    return {
        "saved_auc": saved_auc,
        "current_auc": current[f"{key}_auc"],
        "auc_drift": float(abs(saved_auc - current[f"{key}_auc"])),
        "max_absolute_score_drift": float(drift.max()),
        "mean_absolute_score_drift": float(drift.mean()),
        "max_rank_displacement": int(
            np.abs(rank_positions(saved_scores) - rank_positions(current_scores)).max()
        ),
    }


def passes(values: dict) -> bool:
    return (
        values["max_absolute_score_drift"] <= MAX_ABSOLUTE_SCORE_DRIFT
        and values["auc_drift"] <= MAX_AUC_DRIFT
        and values["max_rank_displacement"] <= MAX_RANK_DISPLACEMENT
    )


def main() -> None:
    if torch.cuda.device_count() < 2:
        raise RuntimeError("batch stability audit requires the verified two-T4 session")
    sealed_root, sealed_package = sealed.validate_package()
    manifest = sealed_root / sealed.MANIFESTS["qwen_prompt_holdout"]["path"]
    if runner.file_sha256(manifest) != sealed.MANIFESTS["qwen_prompt_holdout"]["sha256"]:
        raise RuntimeError("Qwen holdout manifest SHA-256 mismatch")
    rows = promotion.read_jsonl(manifest)
    if len(rows) != 576:
        raise RuntimeError(f"Qwen holdout row mismatch: {len(rows)}")
    v6_model, v6_checkpoint = promotion.load_model(
        promotion.selected_v6_path(), promotion.V6_SHA256, "cuda:0"
    )
    v9_model, v9_checkpoint = promotion.load_model(
        promotion.V9_ROOT / promotion.MODEL_NAME / "model.pt",
        promotion.V9_SHA256,
        "cuda:1",
    )
    mean = tuple(v6_checkpoint["normalization_mean"])
    std = tuple(v6_checkpoint["normalization_std"])
    if mean != tuple(v9_checkpoint["normalization_mean"]) or std != tuple(
        v9_checkpoint["normalization_std"]
    ):
        raise RuntimeError("v6/v9 normalization mismatch")
    dataset = runner.ManifestDataset(
        manifest, stress.condition_transform(mean, std), rows=rows
    )
    runs = {str(size): score_pair(v6_model, v9_model, dataset, size) for size in BATCH_SIZES}
    pair_checks = {
        key: comparison(runs["64"], runs["128"], key)
        for key in ("v6", "blend")
    }
    if not SAVED_CLEAN.is_file():
        raise RuntimeError(f"saved clean predictions are missing: {SAVED_CLEAN}")
    saved = promotion.read_jsonl(SAVED_CLEAN)
    if len(saved) != len(rows):
        raise RuntimeError("saved clean prediction count mismatch")
    saved_checks = {
        key: saved_comparison(runs["128"], saved, key)
        for key in ("v6", "blend")
    }
    report = {
        "completed": True,
        "scope": "already-open Qwen clean holdout only",
        "rows": len(rows),
        "manifest_sha256": runner.file_sha256(manifest),
        "promotion_package_inventory_sha256": sealed_package["inventory_sha256"],
        "v6_checkpoint_sha256": promotion.V6_SHA256,
        "v9_checkpoint_sha256": promotion.V9_SHA256,
        "weights": {"v6": promotion.V6_WEIGHT, "v9": promotion.V9_WEIGHT},
        "tolerances": {
            "max_absolute_score_drift": MAX_ABSOLUTE_SCORE_DRIFT,
            "max_auc_drift": MAX_AUC_DRIFT,
            "max_rank_displacement": MAX_RANK_DISPLACEMENT,
        },
        "input_tensors_identical": (
            runs["64"]["input_tensor_sha256"]
            == runs["128"]["input_tensor_sha256"]
        ),
        "batch_runs": runs,
        "batch_64_vs_128": pair_checks,
        "batch_128_vs_saved_clean": saved_checks,
        "passes": (
            runs["64"]["input_tensor_sha256"]
            == runs["128"]["input_tensor_sha256"]
            and all(passes(values) for values in pair_checks.values())
            and all(passes(values) for values in saved_checks.values())
        ),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(index) for index in (0, 1)],
    }
    for run in report["batch_runs"].values():
        del run["v6_scores"]
        del run["blend_scores"]
    atomic_json(OUTPUT, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
