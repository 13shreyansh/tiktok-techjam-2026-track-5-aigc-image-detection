#!/usr/bin/env python3
"""Falsify the v12 candidates on a frozen, fresh SID_Set gate.

This evaluator never trains, tunes, calibrates or selects a threshold. It
verifies the sealed package, its evaluation-only licence contract, identity
separation from the mounted v12 manifests and the exact candidate checkpoint,
then evaluates clean plus the 19 individually applied workshop conditions.
The interpretation bands are frozen before scoring in
SID_FRESH_MATCHED_V12_GATE_PLAN.json.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import timm
import torch
from sklearn.metrics import roc_auc_score

import kaggle_evaluate_v12_robustness as workshop
import kaggle_train_v3 as runner


SEED = 20260831
GATE_PACKAGE_NAME = "sid-fresh-matched-v12-gate.zip"
GATE_PACKAGE_SHA256 = "439434c4e59b3dbbd4cbe98b9b94464f9e201a3e60cb4560dd87e11ff31f74b0"
GATE_INVENTORY_SHA256 = "19ca0c433aa4e5cb04f8e36262ff9ea430c382987bc0c5d535d3de42e8f71ca3"
GATE_MANIFEST_SHA256 = "092f981ce515ee2061b9f406dd34e358cf0dc4aac5076f65675095135dcb7a27"
V12_INVENTORY_SHA256 = "ec78d74e62d8e1b1f75e661f2ea3338fa95be11e96694a3ed168b463fe314fa6"
CANONICALIZATION = "exif_transpose_center_square_resize336_jpeg_q96_subsampling0"
SOURCE_REVISION = "dc03ead57929879319ce30a82bfcfb8d317b10bd"
GATE_WORK_ROOT = Path("/kaggle/working/sid-fresh-matched-v12-gate")

CANDIDATES = {
    "pe_core": {
        "model": "vit_pe_core_large_patch14_336",
        "root": Path("/kaggle/working/track5-v12-permissive"),
        "checkpoint_sha256": "f37bd6b445b12257ff29a9e54946c5bf9a9184e86a45dc5490537b9ea325ddd2",
    },
    "dinov2_control": {
        "model": "vit_large_patch14_dinov2.lvd142m",
        "root": Path("/kaggle/working/track5-v12-dino-control"),
        "checkpoint_sha256": "db07f30cbc94e4972f4a8c72c95bbe5df0dcd40b2dbd494ad82dc324cc1e2b5b",
    },
}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def validate_gate_rows(rows: list[dict]) -> dict:
    labels = Counter(int(row["label"]) for row in rows)
    if len(rows) != 568 or labels != Counter({0: 284, 1: 284}):
        raise RuntimeError(f"unexpected gate balance: rows={len(rows)}, labels={labels}")
    if len({row["image_sha256"] for row in rows}) != 568:
        raise RuntimeError("gate canonical images are not unique")
    if len({row["source_image_sha256"] for row in rows}) != 568:
        raise RuntimeError("gate source images are not unique")
    for index, row in enumerate(rows):
        if row.get("dataset") != "SID_Set-fresh-validation-v12-gate":
            raise RuntimeError(f"row {index}: dataset mismatch")
        if row.get("source_path_role") != "SID_Set/validation-00001-of-00034":
            raise RuntimeError(f"row {index}: source split mismatch")
        if row.get("source_revision") != SOURCE_REVISION:
            raise RuntimeError(f"row {index}: source revision mismatch")
        if row.get("source_license") != "CC-BY-4.0" or not row.get(
            "license_commercial_use_allowed"
        ):
            raise RuntimeError(f"row {index}: licence mismatch")
        if row.get("training_allowed") is not False or row.get(
            "evaluation_only"
        ) is not True:
            raise RuntimeError(f"row {index}: gate-use contract mismatch")
        if row.get("organizer_demo_row") is not False:
            raise RuntimeError(f"row {index}: organizer demo row")
        if row.get("canonicalization") != CANONICALIZATION:
            raise RuntimeError(f"row {index}: canonicalization mismatch")
        if row.get("canonical_format") != "JPEG" or (
            row.get("canonical_width"), row.get("canonical_height")
        ) != (336, 336):
            raise RuntimeError(f"row {index}: canonical image contract mismatch")
    return {
        "rows": len(rows),
        "labels": dict(labels),
        "unique_source_images": 568,
        "unique_canonical_images": 568,
        "organizer_demo_rows": 0,
        "training_allowed_rows": 0,
        "dataset": "SID_Set-fresh-validation-v12-gate",
        "source_path_role": "SID_Set/validation-00001-of-00034",
        "source_revision": SOURCE_REVISION,
        "source_license": "CC-BY-4.0",
    }


def locate_v12_root() -> Path:
    candidates = []
    for package_path in runner.mounted_root_files("package.json"):
        metadata = json.loads(package_path.read_text())
        if metadata.get("inventory_sha256") == V12_INVENTORY_SHA256:
            candidates.append(package_path.parent)
    if len(candidates) != 1:
        raise RuntimeError(f"expected one mounted v12 root, found {candidates}")
    return candidates[0]


def validate_identity_separation(gate_rows: list[dict], v12_root: Path) -> dict:
    gate_source = {row["source_image_sha256"] for row in gate_rows}
    gate_canonical = {row["image_sha256"] for row in gate_rows}
    compared_rows = 0
    v12_source: set[str] = set()
    v12_canonical: set[str] = set()
    for name in ("train.jsonl", "eval_frozen.jsonl"):
        path = v12_root / "manifests" / name
        if not path.is_file():
            raise RuntimeError(f"missing mounted v12 manifest: {path}")
        rows = read_rows(path)
        compared_rows += len(rows)
        v12_source.update(row["source_image_sha256"] for row in rows)
        v12_canonical.update(row["image_sha256"] for row in rows)
    source_overlap = gate_source & v12_source
    canonical_overlap = gate_canonical & v12_canonical
    if source_overlap or canonical_overlap:
        raise RuntimeError(
            "gate overlaps v12 identities: "
            f"source={len(source_overlap)}, canonical={len(canonical_overlap)}"
        )
    return {
        "v12_rows_compared": compared_rows,
        "source_identity_overlap": 0,
        "canonical_identity_overlap": 0,
        "v12_inventory_sha256": V12_INVENTORY_SHA256,
    }


def gate_decision(clean_auc: float, worst_condition_auc: float) -> dict:
    if clean_auc >= 0.80:
        interpretation = "useful_fresh_high_resolution_same_source_evidence_only"
    elif clean_auc >= 0.65:
        interpretation = "material_source_or_content_dependence_warning"
    else:
        interpretation = "reject_as_trusted_primary"
    return {
        "clean_auc_floor": 0.80,
        "minimum_transformed_floor": 0.60,
        "clean_interpretation": interpretation,
        "passes_frozen_gate": clean_auc >= 0.80 and worst_condition_auc >= 0.60,
        "boundary": (
            "A pass is only fresh high-resolution same-source SID_Set evidence. "
            "The source does not expose generator identity, so this is not an "
            "unseen-generator-family or hidden-set claim."
        ),
    }


def evaluate_candidate(name: str) -> dict:
    if name not in CANDIDATES:
        raise ValueError(f"unknown candidate: {name}")
    specification = CANDIDATES[name]

    runner.PACKAGE_NAME = GATE_PACKAGE_NAME
    runner.EXPECTED_ZIP_SHA256 = GATE_PACKAGE_SHA256
    runner.EXPECTED_INVENTORY_SHA256 = GATE_INVENTORY_SHA256
    runner.WORK_ROOT = GATE_WORK_ROOT
    _, package = runner.validate_package()
    manifest = runner.WORK_ROOT / "manifests/eval_matched.jsonl"
    if runner.file_sha256(manifest) != GATE_MANIFEST_SHA256:
        raise RuntimeError("fresh SID manifest checksum mismatch")
    rows = read_rows(manifest)
    compliance = validate_gate_rows(rows)
    separation = validate_identity_separation(rows, locate_v12_root())

    model_name = specification["model"]
    model_root = specification["root"] / model_name.replace(".", "_")
    checkpoint_path = model_root / "model.pt"
    report_path = model_root / "report.json"
    if not checkpoint_path.is_file() or not report_path.is_file():
        raise RuntimeError(f"missing frozen v12 artifact for {name}")
    clean_report = json.loads(report_path.read_text())
    if not clean_report.get("promotion", {}).get("passes_clean_screen"):
        raise RuntimeError(f"{name} did not pass the frozen v12 clean screen")
    checkpoint_sha256 = runner.file_sha256(checkpoint_path)
    if checkpoint_sha256 != specification["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint checksum mismatch for {name}: {checkpoint_sha256}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != model_name:
        raise RuntimeError(f"checkpoint model mismatch for {name}")
    if checkpoint.get("codec_normalization") != "jpeg_q96":
        raise RuntimeError(f"checkpoint codec contract mismatch for {name}")
    image_size = int(checkpoint["image_size"])
    mean = tuple(checkpoint["normalization_mean"])
    std = tuple(checkpoint["normalization_std"])
    model = timm.create_model(
        model_name, pretrained=False, num_classes=1, img_size=image_size
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda().eval()

    output = model_root / "sid-fresh-matched-v12-gate"
    progress_path = output / "progress.json"
    signature = {
        "candidate": name,
        "checkpoint_sha256": checkpoint_sha256,
        "package_inventory_sha256": package["inventory_sha256"],
        "manifest_sha256": runner.file_sha256(manifest),
        "rows": len(rows),
        "conditions": [condition[0] for condition in workshop.conditions()],
        "codec_normalization": "jpeg_q96",
        "seed": SEED,
    }
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if progress.get("signature") != signature:
            raise RuntimeError(f"incompatible gate resume state for {name}")
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
                raise RuntimeError(f"invalid resume count for {name}/{condition_name}")
            print(json.dumps({"resumed": name, "condition": condition_name}), flush=True)
        else:
            torch.manual_seed(SEED + index)
            dataset = runner.ManifestDataset(
                manifest,
                workshop.condition_transform(image_size, mean, std, image_transform),
                rows=rows,
            )
            metrics, predictions = runner.evaluate(model, dataset)
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            prediction_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions)
            )
            progress["conditions"][condition_name] = metrics
            workshop.atomic_json(progress_path, progress)
            print(
                json.dumps(
                    {"saved_condition": condition_name, "candidate": name, "auc": metrics["clean_auc"]}
                ),
                flush=True,
            )
        if condition_name == "clean":
            clean_predictions = predictions
        else:
            robust_predictions.extend(predictions)

    if clean_predictions is None:
        raise RuntimeError(f"clean predictions missing for {name}")
    clean_labels = [int(row["label"]) for row in clean_predictions]
    clean_scores = [float(row["score"]) for row in clean_predictions]
    clean_auc = float(roc_auc_score(clean_labels, clean_scores))
    robust = workshop.pooled_metrics(rows, robust_predictions)
    worst_condition_auc = min(
        value["clean_auc"] for value in progress["conditions"].values()
    )
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
            "pooled_robust_groups": robust["groups"],
            "worst_individual_condition_auc": worst_condition_auc,
            "frozen_gate_decision": gate_decision(clean_auc, worst_condition_auc),
            "elapsed_seconds_this_process": time.time() - started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        }
    )
    workshop.atomic_json(progress_path, progress)
    print("SID_FRESH_GATE_SUMMARY " + json.dumps(progress, sort_keys=True), flush=True)
    return progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    arguments = parser.parse_args()
    evaluate_candidate(arguments.candidate)


if __name__ == "__main__":
    main()
