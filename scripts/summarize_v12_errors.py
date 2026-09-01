#!/usr/bin/env python3
"""Summarize ranking errors without turning an audit gate into calibration data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import roc_auc_score


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def summarize(manifest: Path, predictions: Path, threshold: float) -> dict:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("illustrative threshold must be in [0, 1]")
    rows = read_jsonl(manifest)
    scored = read_jsonl(predictions)
    if len(rows) != len(scored):
        raise RuntimeError("manifest/prediction row-count mismatch")
    for index, (row, prediction) in enumerate(zip(rows, scored)):
        if int(prediction.get("index", -1)) != index:
            raise RuntimeError(f"prediction index mismatch at row {index}")
        if int(prediction["label"]) != int(row["label"]):
            raise RuntimeError(f"prediction label mismatch at row {index}")
        if prediction.get("image_sha256") != row.get("image_sha256"):
            raise RuntimeError(f"prediction identity mismatch at row {index}")

    labels = [int(item["label"]) for item in scored]
    scores = [float(item["score"]) for item in scored]
    if set(labels) != {0, 1}:
        raise RuntimeError("both labels are required")
    false_positives = [
        (row, item)
        for row, item in zip(rows, scored)
        if int(item["label"]) == 0 and float(item["score"]) >= threshold
    ]
    false_negatives = [
        (row, item)
        for row, item in zip(rows, scored)
        if int(item["label"]) == 1 and float(item["score"]) < threshold
    ]

    def compact(selected: list[tuple[dict, dict]], reverse: bool) -> list[dict]:
        selected = sorted(selected, key=lambda pair: float(pair[1]["score"]), reverse=reverse)
        return [
            {
                "index": int(item["index"]),
                "label": int(item["label"]),
                "score": float(item["score"]),
                "image_sha256": str(item["image_sha256"]),
                "source_filename": str(row.get("source_filename", "undisclosed")),
            }
            for row, item in selected[:5]
        ]

    real_count = labels.count(0)
    fake_count = labels.count(1)
    return {
        "status": "completed_error_summary",
        "rows": len(rows),
        "real_rows": real_count,
        "fake_rows": fake_count,
        "clean_roc_auc": float(roc_auc_score(labels, scores)),
        "illustrative_threshold": threshold,
        "threshold_summary": {
            "false_positives": len(false_positives),
            "false_positive_rate": len(false_positives) / real_count,
            "false_negatives": len(false_negatives),
            "false_negative_rate": len(false_negatives) / fake_count,
        },
        "highest_scoring_authentic": compact(
            [(row, item) for row, item in zip(rows, scored) if int(item["label"]) == 0],
            reverse=True,
        ),
        "lowest_scoring_ai": compact(
            [(row, item) for row, item in zip(rows, scored) if int(item["label"]) == 1],
            reverse=False,
        ),
        "boundary": (
            "The threshold is illustrative and was not selected, tuned or calibrated. "
            "ROC AUC is the organizer metric; this summary is post-selection error analysis."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--illustrative-threshold", type=float, default=0.5)
    args = parser.parse_args()
    result = summarize(args.manifest, args.predictions, args.illustrative_threshold)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
