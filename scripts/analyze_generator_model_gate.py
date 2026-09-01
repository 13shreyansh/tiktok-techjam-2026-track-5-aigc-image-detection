#!/usr/bin/env python3
"""Expose per-generator-model failures hidden by a pooled fake group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--condition", default="clean")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    by_path = {
        str((args.manifest.parent / row["path"]).resolve()): row for row in rows
    }
    progress = json.loads(args.progress.read_text())
    predictions = progress["predictions"][args.condition]
    observed = []
    for path, label, score in zip(
        predictions["paths"], predictions["labels"], predictions["scores"]
    ):
        resolved = str(Path(path).resolve())
        if resolved not in by_path:
            raise SystemExit(f"prediction path absent from manifest: {resolved}")
        observed.append((by_path[resolved], int(label), float(score)))

    real = [(label, score) for _, label, score in observed if label == 0]
    models = sorted(
        {
            str(row["generator_model"])
            for row, label, _ in observed
            if label == 1
        }
    )
    metrics = {}
    for model in models:
        fake = [
            (label, score)
            for row, label, score in observed
            if label == 1 and str(row["generator_model"]) == model
        ]
        selected = real + fake
        labels = [label for label, _ in selected]
        scores = [score for _, score in selected]
        metrics[model] = {
            "fake_count": len(fake),
            "auc_against_all_reals": float(roc_auc_score(labels, scores)),
            "fake_sensitivity_at_0.5": sum(score >= 0.5 for _, score in fake)
            / len(fake),
            "mean_fake_score": float(np.mean([score for _, score in fake])),
        }

    aucs = np.asarray([value["auc_against_all_reals"] for value in metrics.values()])
    sensitivities = np.asarray(
        [value["fake_sensitivity_at_0.5"] for value in metrics.values()]
    )
    report = {
        "warning": (
            "Each model has only four fake samples. Per-model values are high-variance "
            "failure alarms, not precise model-level performance estimates."
        ),
        "condition": args.condition,
        "models": len(models),
        "reals": len(real),
        "summary": {
            "auc_min": float(aucs.min()),
            "auc_q10": float(np.quantile(aucs, 0.10)),
            "auc_median": float(np.median(aucs)),
            "auc_q90": float(np.quantile(aucs, 0.90)),
            "models_auc_below_0.5": int((aucs < 0.5).sum()),
            "models_auc_below_0.7": int((aucs < 0.7).sum()),
            "models_zero_sensitivity_at_0.5": int((sensitivities == 0).sum()),
        },
        "per_model": dict(
            sorted(
                metrics.items(),
                key=lambda item: (item[1]["auc_against_all_reals"], item[0]),
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report["summary"], "models": len(models)}, indent=2))


if __name__ == "__main__":
    main()
