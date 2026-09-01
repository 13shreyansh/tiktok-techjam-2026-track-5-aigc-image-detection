#!/usr/bin/env python3
"""Summarize condition-specific and subgroup failures from frozen evaluations.

This script does not train, tune, or choose a checkpoint.  It only exposes
failure modes that can be hidden by an aggregate AUC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _lowest(mapping: dict[str, dict[str, Any]], metric: str) -> dict[str, Any]:
    key, value = min(mapping.items(), key=lambda item: float(item[1][metric]))
    return {"name": key, metric: value[metric]}


def summarize_evaluation(name: str, path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    evaluation = document["evaluation"]
    conditions = evaluation["conditions"]
    clean_auc = float(conditions["clean"]["auc"])

    rows: list[dict[str, Any]] = []
    for condition_name, condition in conditions.items():
        groups = condition["groups"]
        fake = groups["fake_generators"]
        real = groups["real_sources"]
        pairs = groups["generator_real_source_pairs"]
        pair_rows = [
            {
                "fake_generator": generator,
                "real_source": real_source,
                "auc": values["auc"],
            }
            for generator, real_sources in pairs.items()
            for real_source, values in real_sources.items()
        ]
        worst_pair = min(pair_rows, key=lambda row: float(row["auc"]))
        rows.append(
            {
                "condition": condition_name,
                "auc": condition["auc"],
                "auc_delta_from_clean": float(condition["auc"]) - clean_auc,
                "worst_fake_generator": _lowest(fake, "auc_against_all_reals"),
                "worst_real_source": _lowest(real, "auc_against_all_fakes"),
                "worst_pair": worst_pair,
            }
        )

    rows_by_auc = sorted(rows, key=lambda row: float(row["auc"]))
    rows_by_pair = sorted(rows, key=lambda row: float(row["worst_pair"]["auc"]))
    return {
        "name": name,
        "input": str(path),
        "checkpoint": document.get("checkpoint"),
        "model": document.get("model"),
        "dataset_source": document.get("dataset_source"),
        "official": evaluation.get("official"),
        "clean_auc": clean_auc,
        "worst_overall_condition": rows_by_auc[0],
        "worst_subgroup_pair_condition": rows_by_pair[0],
        "conditions_ranked_by_auc": rows_by_auc,
        "conditions_ranked_by_worst_pair": rows_by_pair,
    }


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=parse_input, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "purpose": "Frozen-model failure audit; not a model-selection result.",
        "interpretation_limits": [
            "AUC is threshold-free; rates at 0.5 are not used for selection here.",
            "A weak source-generator pair can be hidden by aggregate AUC.",
            "These datasets are proxies and do not prove hidden-set performance.",
        ],
        "evaluations": [summarize_evaluation(name, path) for name, path in args.input],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
