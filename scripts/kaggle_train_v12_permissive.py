#!/usr/bin/env python3
"""Train the PE-Core-L v12 head on the workshop-compliant permissive mixture."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import WeightedRandomSampler

import kaggle_train_v3 as runner


runner.SEED = 20260831
runner.PACKAGE_NAME = "permissive-mixture-v12-canonical.zip"
runner.EXPECTED_ZIP_SHA256 = "6bfcb918676cef772b7a71e2fad8ad2fd0789efab9803fb028fb1302cd801447"
runner.EXPECTED_INVENTORY_SHA256 = "ec78d74e62d8e1b1f75e661f2ea3338fa95be11e96694a3ed168b463fe314fa6"
runner.WORK_ROOT = Path("/kaggle/working/permissive-mixture-v12")
runner.OUTPUT_ROOT = Path("/kaggle/working/track5-v12-permissive")
runner.MODEL_NAMES = ("vit_pe_core_large_patch14_336",)
runner.EXCLUDED_EVAL_SHA256 = set()
runner.EXPECTED_TRAIN_ROWS = 13574
runner.EXPECTED_EVAL_ROWS = 2000
runner.EXPECTED_CONTENT_EVAL_ROWS = 2000
runner.CODEC_NORMALIZATION = "jpeg_q96"
runner.PREPROCESS_MODE = "short_side_crop"
runner.AUGMENTATION_DESCRIPTION = (
    "at_most_one_workshop_listed_transformation_then_label_independent_jpeg_q96"
)

ALLOWED_DATASETS = {
    "CIFAKE",
    "COCO train2017 only",
    "COCO-train2017-commercial-compatible",
    "DiTFake",
    "Qwen-Image-Bench",
    "SID_Set",
    "WildFake",
}
REAL_DATASET_MASS = {
    "COCO-train2017-commercial-compatible": 0.60,
    "CIFAKE": 0.25,
    "SID_Set": 0.15,
}
FAKE_DATASET_MASS = {
    "CIFAKE": 0.20,
    "SID_Set": 0.05,
    "DiTFake": 0.25,
    "Qwen-Image-Bench": 0.20,
    "WildFake": 0.30,
}
PROMOTION_FLOORS = {
    "clean_auc": 0.85,
    "worst_fake_generator_auc": 0.65,
    "worst_real_source_auc": 0.65,
    "worst_generator_real_source_pair_auc": 0.60,
}
NONCOMMERCIAL_LICENSE_MARKERS = (
    "noncommercial",
    "by-nc",
    "-nc-",
    "nc-sa",
    "nc-nd",
)


def dataset_sampler(rows: list[dict]) -> tuple[WeightedRandomSampler, dict]:
    counts = Counter((int(row["label"]), str(row["dataset"])) for row in rows)
    weights = []
    for row in rows:
        label = int(row["label"])
        dataset = str(row["dataset"])
        masses = FAKE_DATASET_MASS if label else REAL_DATASET_MASS
        if dataset not in masses:
            raise ValueError(f"no frozen sampler mass for {(label, dataset)}")
        weights.append(0.5 * masses[dataset] / counts[(label, dataset)])
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(rows),
        replacement=True,
        generator=torch.Generator().manual_seed(runner.SEED),
    )
    return sampler, {
        "policy": "equal labels; frozen dataset-block mass within each label",
        "real_dataset_mass": REAL_DATASET_MASS,
        "fake_dataset_mass": FAKE_DATASET_MASS,
        "group_counts": {
            f'{"fake" if label else "real"}:{dataset}': count
            for (label, dataset), count in sorted(counts.items())
        },
    }


def validate_rows(train_rows: list[dict], eval_rows: list[dict]) -> dict:
    for role, rows in (("train", train_rows), ("eval", eval_rows)):
        for index, row in enumerate(rows):
            if row.get("dataset") not in ALLOWED_DATASETS:
                raise RuntimeError(f"{role} row {index}: unapproved dataset")
            if not row.get("license_commercial_use_allowed"):
                raise RuntimeError(f"{role} row {index}: commercial use not affirmed")
            if row.get("organizer_demo_row") is not False:
                raise RuntimeError(f"{role} row {index}: demo-only status is not false")
            expected_training = role == "train"
            if row.get("training_allowed") is not expected_training:
                raise RuntimeError(
                    f"{role} row {index}: training_allowed must be {expected_training}"
                )
            licence = str(row.get("source_license", "")).lower()
            if any(marker in licence for marker in NONCOMMERCIAL_LICENSE_MARKERS):
                raise RuntimeError(f"{role} row {index}: noncommercial licence")
            if "val2017" in str(row.get("path", "")).lower():
                raise RuntimeError(f"{role} row {index}: prohibited val2017 path")
            if row.get("canonicalization") != (
                "exif_transpose_center_square_resize336_jpeg_q96_subsampling0"
            ):
                raise RuntimeError(f"{role} row {index}: canonicalization mismatch")
            if row.get("canonical_format") != "JPEG":
                raise RuntimeError(f"{role} row {index}: canonical format mismatch")
            if (row.get("canonical_width"), row.get("canonical_height")) != (336, 336):
                raise RuntimeError(f"{role} row {index}: canonical geometry mismatch")
    labels = Counter(int(row["label"]) for row in train_rows)
    if labels != Counter({0: 6787, 1: 6787}):
        raise RuntimeError(f"unexpected train labels: {labels}")
    eval_labels = Counter(int(row["label"]) for row in eval_rows)
    if eval_labels != Counter({0: 1000, 1: 1000}):
        raise RuntimeError(f"unexpected eval labels: {eval_labels}")
    overlap = {row["image_sha256"] for row in train_rows} & {
        row["image_sha256"] for row in eval_rows
    }
    if overlap:
        raise RuntimeError(f"train/eval content overlap: {len(overlap)}")
    return {
        "workshop_noncommercial_rows": 0,
        "organizer_demo_rows": 0,
        "train_eval_content_overlap": 0,
        "train_labels": dict(labels),
        "eval_labels": dict(eval_labels),
        "codec_normalization": runner.CODEC_NORMALIZATION,
        "preprocess_mode": runner.PREPROCESS_MODE,
    }


def promotion_decision(report: dict) -> dict:
    metrics = report["selection_clean"]
    groups = metrics["groups"]
    observed = {
        "clean_auc": metrics["clean_auc"],
        "worst_fake_generator_auc": groups["worst_fake_generator_auc"],
        "worst_real_source_auc": groups["worst_real_source_auc"],
        "worst_generator_real_source_pair_auc": groups[
            "worst_generator_real_source_pair_auc"
        ],
    }
    checks = {
        name: observed[name] >= floor for name, floor in PROMOTION_FLOORS.items()
    }
    return {
        "floors_frozen_before_training": PROMOTION_FLOORS,
        "observed": observed,
        "checks": checks,
        "passes_clean_screen": all(checks.values()),
        "boundary": (
            "A clean-screen pass authorizes the unchanged individual-transform "
            "matrix; it does not select the submission artifact."
        ),
    }


def main() -> None:
    if len(runner.EXPECTED_ZIP_SHA256) != 64 or len(runner.EXPECTED_INVENTORY_SHA256) != 64:
        raise RuntimeError("v12 package hashes must be frozen before Kaggle execution")
    runner.source_balanced_sampler = dataset_sampler
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    package_sha256, metadata = runner.validate_package()
    train_manifest = runner.WORK_ROOT / "manifests/train.jsonl"
    eval_manifest = runner.WORK_ROOT / "manifests/eval_frozen.jsonl"
    train_rows = [json.loads(line) for line in train_manifest.read_text().splitlines() if line]
    eval_rows = [json.loads(line) for line in eval_manifest.read_text().splitlines() if line]
    compliance = validate_rows(train_rows, eval_rows)

    runner.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report = runner.train_candidate(
        runner.MODEL_NAMES[0],
        train_manifest,
        eval_manifest,
        eval_manifest,
        package_sha256,
        metadata["inventory_sha256"],
    )
    decision = promotion_decision(report)
    report["workshop_compliance"] = compliance
    report["promotion"] = decision
    model_dir = runner.OUTPUT_ROOT / runner.MODEL_NAMES[0].replace(".", "_")
    (model_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = {
        "model": runner.MODEL_NAMES[0],
        "workshop_compliance": compliance,
        "promotion": decision,
        "checkpoint_sha256": runner.file_sha256(model_dir / "model.pt"),
        "checkpoint_bytes": (model_dir / "model.pt").stat().st_size,
    }
    (runner.OUTPUT_ROOT / "comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
