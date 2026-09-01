#!/usr/bin/env python3
"""Select a source-aware operating threshold from signed clean predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aigc_detector.data import discover_manifest_records
from aigc_detector.error_analysis import load_clean_progress_predictions


def threshold_metrics(
    labels: list[float],
    scores: list[float],
    paths: list[str],
    metadata: dict[str, dict],
    threshold: float,
) -> dict:
    rows = []
    for label, score, path in zip(labels, scores, paths):
        record = metadata[str(Path(path).resolve())]
        rows.append(
            (
                int(label),
                float(score),
                f"real:{record.get('real_source', 'unknown')}"
                if int(label) == 0
                else f"fake:{record.get('generator', 'unknown')}",
            )
        )
    return _threshold_metrics_rows(rows, threshold)


def _threshold_metrics_rows(rows: list[tuple[int, float, str]], threshold: float) -> dict:
    label_array = np.asarray([row[0] for row in rows], dtype=np.int8)
    score_array = np.asarray([row[1] for row in rows], dtype=np.float64)
    group_array = np.asarray([row[2] for row in rows], dtype=object)
    return _threshold_metrics_arrays(label_array, score_array, group_array, threshold)


def _threshold_metrics_arrays(
    label_array: np.ndarray,
    score_array: np.ndarray,
    group_array: np.ndarray,
    threshold: float,
) -> dict:
    predicted_fake = score_array >= threshold
    real_mask = label_array == 0
    fake_mask = label_array == 1
    real_total = int(real_mask.sum())
    fake_total = int(fake_mask.sum())
    if not real_total or not fake_total:
        raise ValueError("threshold selection requires both classes")
    false_positive_rate = float(predicted_fake[real_mask].mean())
    false_negative_rate = float((~predicted_fake[fake_mask]).mean())
    group_recall = {}
    for name in sorted(set(group_array.tolist())):
        mask = group_array == name
        if name.startswith("real:"):
            group_recall[name] = float((~predicted_fake[mask]).mean())
        else:
            group_recall[name] = float(predicted_fake[mask].mean())
    return {
        "threshold": float(threshold),
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "balanced_accuracy": 1.0
        - 0.5 * (false_positive_rate + false_negative_rate),
        "minimum_group_recall": min(group_recall.values()),
        "group_recall": group_recall,
    }


def select_threshold(
    labels: list[float],
    scores: list[float],
    paths: list[str],
    metadata: dict[str, dict],
) -> dict:
    rows = []
    for label, score, path in zip(labels, scores, paths):
        record = metadata[str(Path(path).resolve())]
        rows.append(
            (
                int(label),
                float(score),
                f"real:{record.get('real_source', 'unknown')}"
                if int(label) == 0
                else f"fake:{record.get('generator', 'unknown')}",
            )
        )
    candidates = sorted({0.0, 0.5, 1.0, *(float(score) for score in scores)})
    label_array = np.asarray([row[0] for row in rows], dtype=np.int8)
    score_array = np.asarray([row[1] for row in rows], dtype=np.float64)
    group_array = np.asarray([row[2] for row in rows], dtype=object)
    evaluated = [
        _threshold_metrics_arrays(label_array, score_array, group_array, threshold)
        for threshold in candidates
    ]
    selected = max(
        evaluated,
        key=lambda row: (
            row["minimum_group_recall"],
            row["balanced_accuracy"],
            -abs(row["threshold"] - 0.5),
        ),
    )
    default = next(row for row in evaluated if row["threshold"] == 0.5)
    return {
        "policy": (
            "maximize the weakest known real-source specificity or fake-generator "
            "sensitivity; break ties by global balanced accuracy"
        ),
        "selected": selected,
        "default_0.5": default,
        "candidate_thresholds": len(candidates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Choose a subgroup-minimax operating threshold"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation-progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels, scores, paths, signature = load_clean_progress_predictions(
        args.evaluation_progress, args.manifest
    )
    metadata = {
        str(record["path"]): record
        for record in discover_manifest_records(args.manifest)
    }
    payload = {
        "manifest": str(args.manifest),
        "evaluation_progress": str(args.evaluation_progress),
        "checkpoint": signature.get("checkpoint"),
        "checkpoint_sha256": signature.get("checkpoint_sha256"),
        "preprocess_mode": signature.get("preprocess_mode"),
        "codec_normalization": signature.get("codec_normalization"),
        "selection": select_threshold(labels, scores, paths, metadata),
        "limitations": [
            "selected on permitted internal development data, not hidden data",
            "does not calibrate sigmoid scores into empirical probabilities",
            "must never be fitted on the demo-only COCO/DALL-E resources",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
