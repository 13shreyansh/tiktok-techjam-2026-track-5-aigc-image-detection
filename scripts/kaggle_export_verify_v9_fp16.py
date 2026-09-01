#!/usr/bin/env python3
"""Export v9 with FP16 storage and compare exact frozen Qwen predictions.

This is a packaging ablation only. It does not promote the exported checkpoint:
even an exact clean-Qwen result still requires the frozen full promotion matrix.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score

import kaggle_evaluate_frontier_ensemble_promotion as promotion
import kaggle_evaluate_v8_promotion_gates as sealed
import kaggle_stress_eval_v6 as stress
import kaggle_train_v3 as runner
import kaggle_verify_ensemble_fixed_batch as fixed
import kaggle_verify_ensemble_single_gpu_contract as single
from export_inference_checkpoint import export_checkpoint


SOURCE = promotion.V9_ROOT / promotion.MODEL_NAME / "model.pt"
DESTINATION = Path("/kaggle/working/model_v9_fp16_candidate.pt")
OUTPUT = Path("/kaggle/working/track5-v9-fp16-export-audit.json")
DEVICE = "cuda:0"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("FP16 checkpoint audit requires CUDA")

    export_started = time.perf_counter()
    export_report = export_checkpoint(
        SOURCE,
        DESTINATION,
        expected_source_sha256=promotion.V9_SHA256,
    )
    export_seconds = time.perf_counter() - export_started
    destination_sha256 = str(export_report["destination_sha256"])

    sealed_root, sealed_package = sealed.validate_package()
    manifest = sealed_root / sealed.MANIFESTS["qwen_prompt_holdout"]["path"]
    if runner.file_sha256(manifest) != sealed.MANIFESTS["qwen_prompt_holdout"]["sha256"]:
        raise RuntimeError("Qwen holdout manifest SHA-256 mismatch")
    rows = promotion.read_jsonl(manifest)
    if len(rows) != single.EXPECTED_ROWS:
        raise RuntimeError(f"Qwen row mismatch: {len(rows)}")

    v6_path = promotion.selected_v6_path()
    load_started = time.perf_counter()
    v6_model, v6_checkpoint = promotion.load_model(
        v6_path, promotion.V6_SHA256, DEVICE
    )
    compact_v9_model, compact_v9_checkpoint = promotion.load_model(
        DESTINATION, destination_sha256, DEVICE
    )
    torch.cuda.synchronize(0)
    load_seconds = time.perf_counter() - load_started
    mean = tuple(v6_checkpoint["normalization_mean"])
    std = tuple(v6_checkpoint["normalization_std"])
    if mean != tuple(compact_v9_checkpoint["normalization_mean"]) or std != tuple(
        compact_v9_checkpoint["normalization_std"]
    ):
        raise RuntimeError("v6/compact-v9 normalization mismatch")
    dataset = runner.ManifestDataset(
        manifest, stress.condition_transform(mean, std), rows=rows
    )
    run = single.score_single_gpu(v6_model, compact_v9_model, dataset)

    if not fixed.SAVED_CLEAN.is_file():
        raise RuntimeError(f"saved predictions missing: {fixed.SAVED_CLEAN}")
    saved = promotion.read_jsonl(fixed.SAVED_CLEAN)
    saved_by_hash = {str(row["image_sha256"]): row for row in saved}
    if len(saved_by_hash) != len(rows):
        raise RuntimeError("saved prediction count or hash uniqueness mismatch")
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
        "status": "packaging_ablation_not_promoted",
        "scope": "complete already-open 576-image clean Qwen holdout",
        "source_checkpoint_sha256": promotion.V9_SHA256,
        "compact_checkpoint_sha256": destination_sha256,
        "export": {**export_report, "elapsed_seconds": export_seconds},
        "checkpoint_size_reduction_bytes": int(export_report["source_bytes"])
        - int(export_report["destination_bytes"]),
        "checkpoint_size_ratio": float(export_report["destination_bytes"])
        / float(export_report["source_bytes"]),
        "promotion_package_inventory_sha256": sealed_package["inventory_sha256"],
        "manifest_sha256": runner.file_sha256(manifest),
        "load_seconds": load_seconds,
        "saved_prediction_comparison": comparisons,
        "run": run,
        "passes_exact_clean_qwen_screen": exact,
        "promotion_boundary": (
            "A pass only admits the compact checkpoint to the unchanged full "
            "promotion matrix. It does not select or publish the artifact."
        ),
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
