#!/usr/bin/env python3
"""Evaluate the preserved v6 fallback on the exact v12 CIFAKE gate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import timm
import torch
from sklearn.metrics import roc_auc_score

try:
    import kaggle_evaluate_cifake_matched_v12_gate as gate
    import kaggle_evaluate_v12_robustness as workshop
    import kaggle_train_v3 as runner
except ModuleNotFoundError:  # repository package import used by local tests
    from scripts import kaggle_evaluate_cifake_matched_v12_gate as gate
    from scripts import kaggle_evaluate_v12_robustness as workshop
    from scripts import kaggle_train_v3 as runner


SEED = 20260831
MODEL_NAME = "vit_pe_core_large_patch14_336"
CHECKPOINT_SHA256 = "48ea50773fbd1b7247fff25fde6f985183e29f2eb517b5ac0f6319c1fe38b644"
CHECKPOINT_BYTES = 631645967


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def evaluate(checkpoint_path: Path, output: Path) -> dict:
    if checkpoint_path.stat().st_size != CHECKPOINT_BYTES:
        raise RuntimeError("v6 checkpoint size mismatch")
    checkpoint_sha256 = runner.file_sha256(checkpoint_path)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise RuntimeError("v6 checkpoint checksum mismatch")

    runner.PACKAGE_NAME = gate.GATE_PACKAGE_NAME
    runner.EXPECTED_ZIP_SHA256 = gate.GATE_PACKAGE_SHA256
    runner.EXPECTED_INVENTORY_SHA256 = gate.GATE_INVENTORY_SHA256
    runner.WORK_ROOT = gate.GATE_WORK_ROOT
    _, package = runner.validate_package()
    manifest = runner.WORK_ROOT / "manifests/eval_matched.jsonl"
    if runner.file_sha256(manifest) != gate.GATE_MANIFEST_SHA256:
        raise RuntimeError("matched-source manifest checksum mismatch")
    rows = read_rows(manifest)
    compliance = gate.validate_gate_rows(rows)
    separation = gate.validate_identity_separation(rows, gate.locate_v12_root())

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != MODEL_NAME:
        raise RuntimeError("v6 checkpoint model mismatch")
    image_size = int(checkpoint["image_size"])
    mean = tuple(checkpoint["normalization_mean"])
    std = tuple(checkpoint["normalization_std"])
    model = timm.create_model(
        MODEL_NAME, pretrained=False, num_classes=1, img_size=image_size
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda().eval()

    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    signature = {
        "candidate": "preserved_v6_pe_core",
        "checkpoint_sha256": checkpoint_sha256,
        "package_inventory_sha256": package["inventory_sha256"],
        "manifest_sha256": runner.file_sha256(manifest),
        "rows": len(rows),
        "conditions": [condition[0] for condition in workshop.conditions()],
        "seed": SEED,
    }
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if progress.get("signature") != signature:
            raise RuntimeError("incompatible v6 gate resume state")
    else:
        progress = {"completed": False, "signature": signature, "conditions": {}}

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    clean_predictions = None
    robust_predictions: list[dict] = []
    for index, (condition_name, image_transform) in enumerate(workshop.conditions()):
        prediction_path = output / f"{condition_name}_predictions.jsonl"
        if condition_name in progress["conditions"] and prediction_path.is_file():
            predictions = read_rows(prediction_path)
            if len(predictions) != len(rows):
                raise RuntimeError(f"invalid resume count: {condition_name}")
            print(json.dumps({"resumed_condition": condition_name}), flush=True)
        else:
            torch.manual_seed(SEED + index)
            dataset = runner.ManifestDataset(
                manifest,
                workshop.condition_transform(image_size, mean, std, image_transform),
                rows=rows,
            )
            metrics, predictions = runner.evaluate(model, dataset)
            prediction_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions)
            )
            progress["conditions"][condition_name] = metrics
            workshop.atomic_json(progress_path, progress)
            print(
                json.dumps({"saved_condition": condition_name, "auc": metrics["clean_auc"]}),
                flush=True,
            )
        if condition_name == "clean":
            clean_predictions = predictions
        else:
            robust_predictions.extend(predictions)

    if clean_predictions is None:
        raise RuntimeError("clean predictions missing")
    clean_auc = float(
        roc_auc_score(
            [int(row["label"]) for row in clean_predictions],
            [float(row["score"]) for row in clean_predictions],
        )
    )
    robust = workshop.pooled_metrics(rows, robust_predictions)
    worst = min(value["clean_auc"] for value in progress["conditions"].values())
    progress.update(
        {
            "completed": True,
            "gate_compliance": compliance,
            "identity_separation": separation,
            "official_style": {
                "clean_auc": clean_auc,
                "pooled_robust_auc": robust["auc"],
                "score": 0.5 * clean_auc + 0.5 * robust["auc"],
            },
            "worst_individual_condition_auc": worst,
            "elapsed_seconds_this_process": time.time() - started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "boundary": (
                "This is an already-open within-source low-resolution comparison; "
                "it is not fresh hidden-transfer evidence."
            ),
        }
    )
    workshop.atomic_json(progress_path, progress)
    return progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/v6-cifake-matched-source-v12-gate"),
    )
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
