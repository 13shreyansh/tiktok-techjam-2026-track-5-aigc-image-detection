#!/usr/bin/env python3
"""Audit exact generator and raw family-label coverage across manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def generator_name(row: dict[str, Any]) -> str:
    return str(row.get("generator_model") or row.get("generator") or "unspecified")


def inventory(path: Path) -> dict[str, Any]:
    fakes = [row for row in read_jsonl(path) if int(row["label"]) == 1]
    names = Counter(generator_name(row) for row in fakes)
    families = Counter(str(row.get("family") or "unspecified") for row in fakes)
    return {
        "fake_rows": len(fakes),
        "generator_names": dict(sorted(names.items())),
        "raw_family_labels": dict(sorted(families.items())),
    }


def audit(
    candidates: dict[str, Path], gates: dict[str, Path]
) -> dict[str, Any]:
    candidate_inventories = {name: inventory(path) for name, path in candidates.items()}
    gate_inventories = {name: inventory(path) for name, path in gates.items()}
    comparisons: dict[str, Any] = {}
    for candidate, train in candidate_inventories.items():
        train_names = set(train["generator_names"])
        train_families = set(train["raw_family_labels"])
        candidate_results: dict[str, Any] = {}
        for gate, evaluation in gate_inventories.items():
            eval_names = set(evaluation["generator_names"])
            eval_families = set(evaluation["raw_family_labels"])
            candidate_results[gate] = {
                "exact_generator_names_seen_in_training": sorted(eval_names & train_names),
                "exact_generator_names_absent_from_training": sorted(eval_names - train_names),
                "exact_generator_name_holdout_fraction": (
                    len(eval_names - train_names) / len(eval_names) if eval_names else 0.0
                ),
                "raw_family_labels_seen_in_training": sorted(eval_families & train_families),
                "raw_family_labels_absent_from_training": sorted(eval_families - train_families),
            }
        comparisons[candidate] = candidate_results
    return {
        "purpose": "Manifest-only coverage audit; no inference, training or model selection.",
        "candidates": candidate_inventories,
        "gates": gate_inventories,
        "comparisons": comparisons,
        "interpretation": [
            "Exact-name absence is evidence that the named generator was not represented in the candidate manifest.",
            "Raw family labels are not canonical taxonomies; spelling differences can make semantically related families appear absent.",
            "Coverage is not performance. A gate needs completed model scores before it supports a transfer claim.",
            "Prompt-disjoint images from a generator present in training are not a generator-family holdout.",
            "An undisclosed generator label cannot establish exact-name or family novelty.",
        ],
    }


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", type=parse_named_path, required=True)
    parser.add_argument("--gate", action="append", type=parse_named_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(dict(args.candidate), dict(args.gate))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
