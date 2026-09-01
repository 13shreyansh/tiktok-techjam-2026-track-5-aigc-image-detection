#!/usr/bin/env python3
"""Compare fixed-gate candidate reports without hiding subgroup collapses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def extract_metrics(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if "history" in report:
        if not report["history"]:
            raise ValueError(f"training report has no completed epochs: {path}")
        clean = report["history"][-1]["evaluation"]["conditions"]["clean"]
        content = None
        model = report.get("arguments", {}).get("model", "unknown")
    elif "selection_clean" in report:
        selection = report["selection_clean"]
        clean = {"auc": selection["clean_auc"], "groups": selection["groups"]}
        content = report.get("content_holdout_clean")
        model = report.get("model", "unknown")
    else:
        raise ValueError(f"unrecognized candidate report schema: {path}")

    groups = clean.get("groups")
    if not groups:
        raise ValueError(f"candidate report lacks source/generator groups: {path}")
    metrics = {
        "clean_auc": float(clean["auc"]),
        "worst_fake_generator_auc": float(groups["worst_fake_generator_auc"]),
        "worst_real_source_auc": float(groups["worst_real_source_auc"]),
        "worst_generator_real_source_pair_auc": float(
            groups["worst_generator_real_source_pair_auc"]
        ),
    }
    if content is not None:
        metrics["content_holdout_clean_auc"] = float(content["clean_auc"])
        metrics["content_holdout_worst_pair_auc"] = float(
            content["groups"]["worst_generator_real_source_pair_auc"]
        )
    for name, value in metrics.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid {name}={value!r} in {path}")

    floor_values = [
        metrics["worst_fake_generator_auc"],
        metrics["worst_real_source_auc"],
        metrics["worst_generator_real_source_pair_auc"],
    ]
    floor_values.extend(
        value
        for name, value in metrics.items()
        if name.startswith("content_holdout_")
    )
    hidden_set_floor = min(floor_values)
    clean_floor_hmean = 2.0 / (
        (1.0 / metrics["clean_auc"]) + (1.0 / hidden_set_floor)
    )
    return {
        "report": str(path),
        "model": model,
        **metrics,
        "hidden_set_floor_auc": hidden_set_floor,
        "clean_floor_harmonic_mean": clean_floor_hmean,
        "below_chance_subgroup_veto": hidden_set_floor < 0.5,
    }


def compare(paths: list[Path]) -> dict:
    candidates = [extract_metrics(path) for path in paths]
    ranked = sorted(
        candidates,
        key=lambda row: (
            row["below_chance_subgroup_veto"],
            -row["clean_floor_harmonic_mean"],
            -row["hidden_set_floor_auc"],
            -row["clean_auc"],
            row["report"],
        ),
    )
    eligible = [row for row in ranked if not row["below_chance_subgroup_veto"]]
    return {
        "policy": {
            "veto": "reject any observed generator/source floor below chance (0.5)",
            "ranking": (
                "among non-vetoed candidates, maximize the harmonic mean of "
                "clean AUC and the weakest recorded held-out/content AUC"
            ),
            "warning": (
                "this fixed-gate comparison does not estimate the organizer's "
                "unknown hidden score and does not replace the 19-transform matrix"
            ),
        },
        "candidates": ranked,
        "provisional_leader": eligible[0]["report"] if eligible else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = compare(args.reports)
    encoded = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
