#!/usr/bin/env python3
"""Test fixed-physical-batch padding as the ensemble deployment policy."""

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


OUTPUT = Path("/kaggle/working/track5-frontier-ensemble-fixed-batch.json")
SAVED_CLEAN = promotion.OUTPUT_ROOT / "qwen_prompt_holdout/clean_predictions.jsonl"
PHYSICAL_BATCH_SIZE = 64
LOGICAL_BATCH_SIZES = (1, 17, 64)
ROWS_PER_LABEL = 32
MAX_ABSOLUTE_SCORE_DRIFT = 1e-6
MAX_AUC_DRIFT = 0.0
MAX_RANK_DISPLACEMENT = 0


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def rank_positions(values: np.ndarray) -> np.ndarray:
    order = np.lexsort((np.arange(len(values)), values))
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values), dtype=np.int64)
    return ranks


def compare_scores(
    reference_scores: list[float],
    candidate_scores: list[float],
    reference_auc: float,
    candidate_auc: float,
) -> dict:
    reference = np.asarray(reference_scores, dtype=np.float64)
    candidate = np.asarray(candidate_scores, dtype=np.float64)
    absolute = np.abs(reference - candidate)
    ranks = np.abs(rank_positions(reference) - rank_positions(candidate))
    return {
        "max_absolute_score_drift": float(absolute.max()),
        "mean_absolute_score_drift": float(absolute.mean()),
        "auc_drift": float(abs(reference_auc - candidate_auc)),
        "max_rank_displacement": int(ranks.max()),
    }


def passes(values: dict) -> bool:
    return (
        values["max_absolute_score_drift"] <= MAX_ABSOLUTE_SCORE_DRIFT
        and values["auc_drift"] <= MAX_AUC_DRIFT
        and values["max_rank_displacement"] <= MAX_RANK_DISPLACEMENT
    )


@torch.inference_mode()
def score_fixed_physical_batch(
    v6_model: torch.nn.Module,
    v9_model: torch.nn.Module,
    dataset: runner.ManifestDataset,
    logical_batch_size: int,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=logical_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    digest = hashlib.sha256()
    labels: list[int] = []
    indices: list[int] = []
    v6_values: list[float] = []
    blend_values: list[float] = []
    for device_index in (0, 1):
        torch.cuda.reset_peak_memory_stats(device_index)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    wall_started = time.perf_counter()
    forward_seconds = 0.0
    physical_batches = 0
    for images, batch_labels, batch_indices in loader:
        original_count = int(images.shape[0])
        digest.update(images.contiguous().numpy().tobytes())
        if original_count < PHYSICAL_BATCH_SIZE:
            padding = images[-1:].repeat(
                PHYSICAL_BATCH_SIZE - original_count, 1, 1, 1
            )
            images = torch.cat([images, padding], dim=0)
        if int(images.shape[0]) != PHYSICAL_BATCH_SIZE:
            raise RuntimeError(f"physical batch mismatch: {tuple(images.shape)}")
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        forward_started = time.perf_counter()
        images_v6 = images.to("cuda:0", non_blocking=True)
        images_v9 = images.to("cuda:1", non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            v6_scores = torch.sigmoid(v6_model(images_v6).flatten())
            v9_scores = torch.sigmoid(v9_model(images_v9).flatten())
        v6_cpu = v6_scores[:original_count].float().cpu()
        v9_cpu = v9_scores[:original_count].float().cpu()
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
        physical_batches += 1
    if indices != list(range(len(dataset))):
        raise RuntimeError("fixed-batch audit did not preserve row order")
    return {
        "logical_batch_size": logical_batch_size,
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "physical_batches": physical_batches,
        "count": len(indices),
        "input_tensor_sha256": digest.hexdigest(),
        "v6_scores": v6_values,
        "blend_scores": blend_values,
        "v6_auc": float(roc_auc_score(labels, v6_values)),
        "blend_auc": float(roc_auc_score(labels, blend_values)),
        "wall_seconds_including_decode_and_input_hash": time.perf_counter()
        - wall_started,
        "paired_model_forward_seconds": forward_seconds,
        "cuda_peak_allocated_bytes": {
            str(index): int(torch.cuda.max_memory_allocated(index))
            for index in (0, 1)
        },
    }


def main() -> None:
    if torch.cuda.device_count() < 2:
        raise RuntimeError("fixed-batch audit requires the verified two-T4 session")
    sealed_root, sealed_package = sealed.validate_package()
    manifest = sealed_root / sealed.MANIFESTS["qwen_prompt_holdout"]["path"]
    if runner.file_sha256(manifest) != sealed.MANIFESTS["qwen_prompt_holdout"]["sha256"]:
        raise RuntimeError("Qwen holdout manifest SHA-256 mismatch")
    all_rows = promotion.read_jsonl(manifest)
    rows = []
    for label in (0, 1):
        rows.extend([row for row in all_rows if int(row["label"]) == label][:ROWS_PER_LABEL])
    if len(rows) != 2 * ROWS_PER_LABEL or {int(row["label"]) for row in rows} != {0, 1}:
        raise RuntimeError("fixed-batch audit subset is not balanced")
    selected_digest = hashlib.sha256(
        "".join(f'{row["image_sha256"]}:{row["label"]}\n' for row in rows).encode()
    ).hexdigest()

    v6_path = promotion.selected_v6_path()
    v9_path = promotion.V9_ROOT / promotion.MODEL_NAME / "model.pt"
    load_started = time.perf_counter()
    v6_model, v6_checkpoint = promotion.load_model(
        v6_path, promotion.V6_SHA256, "cuda:0"
    )
    v9_model, v9_checkpoint = promotion.load_model(
        v9_path, promotion.V9_SHA256, "cuda:1"
    )
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
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
    runs = {
        str(size): score_fixed_physical_batch(v6_model, v9_model, dataset, size)
        for size in LOGICAL_BATCH_SIZES
    }
    reference = runs[str(PHYSICAL_BATCH_SIZE)]
    logical_checks = {}
    for size in LOGICAL_BATCH_SIZES:
        candidate = runs[str(size)]
        logical_checks[str(size)] = {
            key: compare_scores(
                reference[f"{key}_scores"],
                candidate[f"{key}_scores"],
                reference[f"{key}_auc"],
                candidate[f"{key}_auc"],
            )
            for key in ("v6", "blend")
        }

    if not SAVED_CLEAN.is_file():
        raise RuntimeError(f"saved clean predictions are missing: {SAVED_CLEAN}")
    saved_rows = promotion.read_jsonl(SAVED_CLEAN)
    saved_by_hash = {str(row["image_sha256"]): row for row in saved_rows}
    if any(str(row["image_sha256"]) not in saved_by_hash for row in rows):
        raise RuntimeError("audit subset is absent from saved clean predictions")
    saved_labels = [int(row["label"]) for row in rows]
    saved_checks = {}
    for key, saved_key in (("v6", "v6_score"), ("blend", "score")):
        saved_scores = [
            float(saved_by_hash[str(row["image_sha256"])][saved_key]) for row in rows
        ]
        saved_auc = float(roc_auc_score(saved_labels, saved_scores))
        saved_checks[key] = compare_scores(
            saved_scores,
            reference[f"{key}_scores"],
            saved_auc,
            reference[f"{key}_auc"],
        )

    input_hashes = {run["input_tensor_sha256"] for run in runs.values()}
    report = {
        "completed": True,
        "scope": "64 balanced rows from the already-open Qwen clean holdout",
        "rows": len(rows),
        "selected_rows_sha256": selected_digest,
        "manifest_sha256": runner.file_sha256(manifest),
        "promotion_package_inventory_sha256": sealed_package["inventory_sha256"],
        "v6_checkpoint_sha256": promotion.V6_SHA256,
        "v9_checkpoint_sha256": promotion.V9_SHA256,
        "checkpoint_bytes": {"v6": v6_path.stat().st_size, "v9": v9_path.stat().st_size},
        "load_seconds": load_seconds,
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "logical_batch_sizes": list(LOGICAL_BATCH_SIZES),
        "tolerances": {
            "max_absolute_score_drift": MAX_ABSOLUTE_SCORE_DRIFT,
            "max_auc_drift": MAX_AUC_DRIFT,
            "max_rank_displacement": MAX_RANK_DISPLACEMENT,
        },
        "original_input_tensors_identical": len(input_hashes) == 1,
        "input_tensor_sha256": next(iter(input_hashes)) if len(input_hashes) == 1 else None,
        "logical_batch_vs_native_64": logical_checks,
        "native_64_vs_saved_clean_subset": saved_checks,
        "runs": runs,
        "passes": (
            len(input_hashes) == 1
            and all(
                passes(values)
                for result in logical_checks.values()
                for values in result.values()
            )
            and all(passes(values) for values in saved_checks.values())
        ),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(index) for index in (0, 1)],
    }
    for run in report["runs"].values():
        del run["v6_scores"]
        del run["blend_scores"]
    atomic_json(OUTPUT, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
