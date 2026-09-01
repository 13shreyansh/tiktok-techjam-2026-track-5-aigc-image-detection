#!/usr/bin/env python3
"""Screen three predeclared v6/v9 probability blends on development gates."""

from __future__ import annotations

import json
import time
from pathlib import Path

import timm
import torch

import kaggle_train_v3 as runner
import kaggle_train_v6  # noqa: F401  # pins package and exclusion constants


MODEL_NAME = "vit_pe_core_large_patch14_336"
V6_SHA256 = "48ea50773fbd1b7247fff25fde6f985183e29f2eb517b5ac0f6319c1fe38b644"
V9_SHA256 = "dd6b26c7849489447c7e96823f5b5e87c31623ca8ae0d28a5b162bb2dcb65075"
V9_ROOT = Path("/kaggle/working/track5-v9-frontier-capped-candidate")
DIAGNOSIS_ROOT = Path("/kaggle/working/track5-frontier-diagnosis-comparison")
OUTPUT_ROOT = Path("/kaggle/working/track5-frontier-ensemble-screen")
V6_WEIGHTS = (0.90, 0.75, 0.50)
FLOORS = {
    "selection_clean_auc": 0.979830,
    "selection_worst_pair_auc": 0.862510,
    "content_clean_auc": 0.992353,
    "content_worst_pair_auc": 0.959759,
}


def selected_v6_path() -> Path:
    matches = [
        path
        for path in Path("/kaggle/input").rglob("model.pt")
        if "track5-v6-selected-fp16-checkpoint" in str(path)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one selected v6 checkpoint: {matches}")
    if runner.file_sha256(matches[0]) != V6_SHA256:
        raise RuntimeError("selected v6 checkpoint SHA-256 mismatch")
    return matches[0]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def canonical_scores(predictions: list[dict]) -> dict[str, dict]:
    by_hash: dict[str, list[dict]] = {}
    for row in predictions:
        image_sha256 = row.get("image_sha256")
        if not image_sha256:
            raise RuntimeError("prediction missing image SHA-256")
        by_hash.setdefault(image_sha256, []).append(row)
    canonical = {}
    for image_sha256, duplicates in by_hash.items():
        labels = {int(row["label"]) for row in duplicates}
        scores = [float(row["score"]) for row in duplicates]
        if len(labels) != 1 or max(scores) - min(scores) > 1e-6:
            raise RuntimeError(f"inconsistent duplicate prediction: {image_sha256}")
        canonical[image_sha256] = duplicates[0]
    return canonical


def aligned_blend(
    rows: list[dict],
    v6_predictions: list[dict],
    v9_predictions: list[dict],
    v6_weight: float,
) -> list[dict]:
    v6 = canonical_scores(v6_predictions)
    v9 = canonical_scores(v9_predictions)
    blended = []
    for index, row in enumerate(rows):
        image_sha256 = row.get("image_sha256")
        if not image_sha256 or image_sha256 not in v6 or image_sha256 not in v9:
            raise RuntimeError(f"missing aligned prediction: {image_sha256}")
        if int(v6[image_sha256]["label"]) != int(row["label"]) or int(
            v9[image_sha256]["label"]
        ) != int(row["label"]):
            raise RuntimeError(f"aligned label mismatch: {image_sha256}")
        blended.append(
            {
                **row,
                "index": index,
                "score": (
                    v6_weight * float(v6[image_sha256]["score"])
                    + (1.0 - v6_weight) * float(v9[image_sha256]["score"])
                ),
            }
        )
    return blended


def metrics(rows: list[dict], predictions: list[dict]) -> dict:
    labels = [int(row["label"]) for row in predictions]
    scores = [float(row["score"]) for row in predictions]
    from sklearn.metrics import roc_auc_score

    return {
        "count": len(rows),
        "clean_auc": float(roc_auc_score(labels, scores)),
        "groups": runner.grouped_metrics(rows, labels, scores),
    }


def main() -> None:
    package_sha256, package = runner.validate_package()
    selection_manifest = runner.WORK_ROOT / "manifests/eval_selection.jsonl"
    content_manifest = runner.WORK_ROOT / "manifests/eval_content_holdout.jsonl"
    selection_rows = runner.filter_evaluation_rows(
        read_jsonl(selection_manifest), runner.EXCLUDED_EVAL_SHA256
    )
    content_rows = runner.filter_evaluation_rows(
        read_jsonl(content_manifest), runner.EXCLUDED_EVAL_SHA256
    )

    v6_path = selected_v6_path()
    v6_checkpoint = torch.load(v6_path, map_location="cpu", weights_only=True)
    if v6_checkpoint["model_name"] != MODEL_NAME:
        raise RuntimeError("selected v6 model mismatch")
    model = timm.create_model(
        MODEL_NAME, pretrained=False, num_classes=1, img_size=runner.IMAGE_SIZE
    )
    model.load_state_dict(v6_checkpoint["state_dict"])
    model.cuda().eval()
    v6_dataset = runner.ManifestDataset(
        selection_manifest,
        runner.eval_transform(
            tuple(v6_checkpoint["normalization_mean"]),
            tuple(v6_checkpoint["normalization_std"]),
        ),
        rows=selection_rows,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    v6_selection_metrics, v6_selection_predictions = runner.evaluate(model, v6_dataset)
    v6_content_metrics, v6_content_predictions = runner.select_evaluation_from_predictions(
        v6_selection_predictions, content_rows
    )
    del model, v6_dataset
    torch.cuda.empty_cache()

    v9_model_root = V9_ROOT / MODEL_NAME
    v9_path = v9_model_root / "model.pt"
    if runner.file_sha256(v9_path) != V9_SHA256:
        raise RuntimeError("v9 checkpoint SHA-256 mismatch")
    v9_selection_predictions = read_jsonl(v9_model_root / "selection_predictions.jsonl")
    v9_content_predictions = read_jsonl(
        v9_model_root / "content_holdout_predictions.jsonl"
    )
    diagnosis_rows = read_jsonl(
        next(
            path
            for path in Path("/kaggle/input").rglob("manifests/combined_gate.jsonl")
            if "track5-frontier-diagnosis-comparison" in str(path)
        )
    )
    v6_diagnosis_predictions = read_jsonl(
        DIAGNOSIS_ROOT / "v6-selected-fp16/predictions.jsonl"
    )
    v9_diagnosis_predictions = read_jsonl(
        DIAGNOSIS_ROOT / "v9-frontier-capped/predictions.jsonl"
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    reports = []
    for v6_weight in V6_WEIGHTS:
        name = f"v6_{v6_weight:.2f}_v9_{1.0-v6_weight:.2f}"
        selection_predictions = aligned_blend(
            selection_rows,
            v6_selection_predictions,
            v9_selection_predictions,
            v6_weight,
        )
        content_predictions = aligned_blend(
            content_rows, v6_content_predictions, v9_content_predictions, v6_weight
        )
        diagnosis_predictions = aligned_blend(
            diagnosis_rows,
            v6_diagnosis_predictions,
            v9_diagnosis_predictions,
            v6_weight,
        )
        selection = metrics(selection_rows, selection_predictions)
        content = metrics(content_rows, content_predictions)
        diagnosis = metrics(diagnosis_rows, diagnosis_predictions)
        observed = {
            "selection_clean_auc": selection["clean_auc"],
            "selection_worst_pair_auc": selection["groups"][
                "worst_generator_real_source_pair_auc"
            ],
            "content_clean_auc": content["clean_auc"],
            "content_worst_pair_auc": content["groups"][
                "worst_generator_real_source_pair_auc"
            ],
        }
        passed = all(observed[key] >= value for key, value in FLOORS.items())
        report = {
            "name": name,
            "v6_probability_weight": v6_weight,
            "v9_probability_weight": 1.0 - v6_weight,
            "selection": selection,
            "content": content,
            "frontier_diagnosis": diagnosis,
            "floors": FLOORS,
            "observed_floor_metrics": observed,
            "passes_all_floors": passed,
        }
        reports.append(report)
        candidate_root = OUTPUT_ROOT / name
        candidate_root.mkdir(parents=True, exist_ok=True)
        (candidate_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (candidate_root / "selection_predictions.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in selection_predictions)
        )
        (candidate_root / "diagnosis_predictions.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in diagnosis_predictions)
        )
        print(json.dumps(report, indent=2), flush=True)

    passing = [report for report in reports if report["passes_all_floors"]]
    selected = None
    if passing:
        selected = max(
            passing,
            key=lambda report: (
                report["frontier_diagnosis"]["clean_auc"],
                report["frontier_diagnosis"]["groups"][
                    "worst_generator_real_source_pair_auc"
                ],
                report["v6_probability_weight"],
            ),
        )["name"]
    summary = {
        "screen_only": True,
        "sealed_holdout_opened": False,
        "selection_rule": (
            "pass all frozen floors; then maximum frontier diagnosis clean AUC; "
            "tie by diagnosis worst pair and larger v6 weight"
        ),
        "selected": selected,
        "v6_checkpoint_sha256": V6_SHA256,
        "v9_checkpoint_sha256": V9_SHA256,
        "v6_reference": {
            "selection": v6_selection_metrics,
            "content": v6_content_metrics,
        },
        "candidates": reports,
        "package_sha256": package_sha256,
        "package_inventory_sha256": package["inventory_sha256"],
        "elapsed_seconds": time.time() - started,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
