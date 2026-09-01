#!/usr/bin/env python3
"""Evaluate the checkpoint saved by v6 after a post-inference report failure."""

from __future__ import annotations

import json
import time
from pathlib import Path

import timm
import torch

import kaggle_train_v3 as runner
import kaggle_train_v6  # noqa: F401  # applies checksum-pinned v6 constants


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    model_name = runner.MODEL_NAMES[0]
    model_output = runner.OUTPUT_ROOT / model_name.replace(".", "_")
    checkpoint_path = model_output / "model.pt"
    # Check the session-local artifact before spending several minutes hashing
    # the full package.  A Kaggle runtime reset clears /kaggle/working.
    if not checkpoint_path.is_file():
        raise RuntimeError(f"saved checkpoint is missing: {checkpoint_path}")

    package_sha256, metadata = runner.validate_package()
    train_manifest = runner.WORK_ROOT / "manifests/train.jsonl"
    eval_manifest = runner.WORK_ROOT / "manifests/eval_selection.jsonl"
    content_manifest = runner.WORK_ROOT / "manifests/eval_content_holdout.jsonl"
    train_rows = read_rows(train_manifest)
    eval_rows_unfiltered = read_rows(eval_manifest)
    content_rows_unfiltered = read_rows(content_manifest)
    if (
        len(train_rows) != runner.EXPECTED_TRAIN_ROWS
        or len(eval_rows_unfiltered) != runner.EXPECTED_EVAL_ROWS
        or len(content_rows_unfiltered) != runner.EXPECTED_CONTENT_EVAL_ROWS
    ):
        raise RuntimeError("unexpected v6 manifest row counts")
    train_hashes = {row["image_sha256"] for row in train_rows}
    eval_hashes = {row["image_sha256"] for row in eval_rows_unfiltered}
    content_hashes = {row["image_sha256"] for row in content_rows_unfiltered}
    overlap = train_hashes & (eval_hashes | content_hashes)
    if overlap:
        raise RuntimeError(f"train/eval content overlap: {len(overlap)}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["model_name"] != model_name:
        raise RuntimeError("saved checkpoint model mismatch")
    model = timm.create_model(
        model_name, pretrained=False, num_classes=1, img_size=runner.IMAGE_SIZE
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    mean = tuple(checkpoint["normalization_mean"])
    std = tuple(checkpoint["normalization_std"])
    eval_rows = runner.filter_evaluation_rows(
        eval_rows_unfiltered, runner.EXCLUDED_EVAL_SHA256
    )
    content_rows = runner.filter_evaluation_rows(
        content_rows_unfiltered, runner.EXCLUDED_EVAL_SHA256
    )
    eval_dataset = runner.ManifestDataset(
        eval_manifest, runner.eval_transform(mean, std), rows=eval_rows
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    evaluation, predictions = runner.evaluate(model, eval_dataset)
    content_evaluation, content_predictions = (
        runner.select_evaluation_from_predictions(predictions, content_rows)
    )
    report = {
        "recovery_mode": (
            "evaluation-only from checkpoint saved before the original "
            "content-subset duplicate-hash failure"
        ),
        "seed": checkpoint["seed"],
        "model": model_name,
        "image_size": runner.IMAGE_SIZE,
        "preprocess_mode": checkpoint["preprocess_mode"],
        "parameters": checkpoint["parameters"],
        "package_sha256": package_sha256,
        "inventory_sha256": metadata["inventory_sha256"],
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "excluded_eval_sha256": sorted(runner.EXCLUDED_EVAL_SHA256),
        "selection_clean": evaluation,
        "content_holdout_clean": content_evaluation,
        "evaluation_elapsed_seconds": time.time() - started,
        "evaluation_cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    model_output.mkdir(parents=True, exist_ok=True)
    (model_output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (model_output / "selection_predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions)
    )
    (model_output / "content_holdout_predictions.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in content_predictions
        )
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
