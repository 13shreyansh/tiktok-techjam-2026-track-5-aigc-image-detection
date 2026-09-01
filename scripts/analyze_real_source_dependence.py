#!/usr/bin/env python3
"""Measure real-source dependence in frozen detector scores.

This is a diagnosis of score instability across authentic-image sources after
the evaluator's normalization. It does not prove which pixels the model uses.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _between_share(groups: dict[str, list[float]]) -> float:
    values = [value for group in groups.values() for value in group]
    grand = statistics.fmean(values)
    total = sum((value - grand) ** 2 for value in values)
    between = sum(
        len(group) * (statistics.fmean(group) - grand) ** 2
        for group in groups.values()
    )
    return between / total if total else 0.0


def analyze(
    manifest_path: Path,
    progress_path: Path,
    *,
    permutations: int = 10_000,
    seed: int = 20260831,
) -> dict[str, Any]:
    rows = read_jsonl(manifest_path)
    progress = json.loads(progress_path.read_text())
    clean = progress["predictions"]["clean"]
    paths = [str(Path(path).resolve()) for path in clean["paths"]]
    scores = [float(score) for score in clean["scores"]]
    labels = [int(label) for label in clean["labels"]]
    if len(paths) != len(scores) or len(paths) != len(labels):
        raise RuntimeError("prediction paths, scores and labels must align")
    if len(paths) != len(set(paths)):
        raise RuntimeError("prediction paths must be unique")
    score_by_path = dict(zip(paths, scores))

    by_source: dict[str, list[float]] = defaultdict(list)
    manifest_real_paths: set[str] = set()
    for row in rows:
        if int(row["label"]) != 0:
            continue
        source = str(row.get("real_source") or "unspecified")
        path = str((manifest_path.parent / row["path"]).resolve())
        if path not in score_by_path:
            raise RuntimeError(f"missing frozen prediction for {path}")
        if path in manifest_real_paths:
            raise RuntimeError(f"duplicate real manifest path: {path}")
        manifest_real_paths.add(path)
        by_source[source].append(score_by_path[path])

    predicted_real_paths = {
        path for path, label in zip(paths, labels) if label == 0
    }
    if manifest_real_paths != predicted_real_paths:
        raise RuntimeError("manifest and frozen real-prediction paths differ")
    if len(by_source) < 2:
        raise RuntimeError("at least two real sources are required")

    fake_scores = [score for score, label in zip(scores, labels) if label == 1]
    if not fake_scores:
        raise RuntimeError("at least one fake score is required")

    observed = _between_share(by_source)
    sizes = [len(values) for values in by_source.values()]
    pooled = [value for values in by_source.values() for value in values]
    rng = random.Random(seed)
    exceed = 0
    for _ in range(permutations):
        shuffled = pooled.copy()
        rng.shuffle(shuffled)
        offset = 0
        permuted: dict[str, list[float]] = {}
        for source, size in zip(by_source, sizes):
            permuted[source] = shuffled[offset : offset + size]
            offset += size
        exceed += _between_share(permuted) >= observed
    permutation_p = (exceed + 1) / (permutations + 1)

    def auc_for_real_source(real_values: list[float]) -> float:
        # Fake is the positive class, so a lower authentic score is a win.
        wins = sum(
            float(fake > real) + 0.5 * float(fake == real)
            for fake in fake_scores
            for real in real_values
        )
        return wins / (len(fake_scores) * len(real_values))

    ranked = []
    for source, values in by_source.items():
        ranked.append(
            {
                "real_source": source,
                "count": len(values),
                "mean_ai_score": statistics.fmean(values),
                "median_ai_score": statistics.median(values),
                "minimum_ai_score": min(values),
                "maximum_ai_score": max(values),
                "auc_against_all_fakes": auc_for_real_source(values),
                "fraction_false_positive_at_0.5": sum(value >= 0.5 for value in values)
                / len(values),
            }
        )
    ranked.sort(key=lambda row: float(row["mean_ai_score"]), reverse=True)

    signature = progress["signature"]
    return {
        "purpose": "Frozen clean-score real-source dependence diagnosis; not model selection.",
        "checkpoint": signature["checkpoint"],
        "checkpoint_sha256": signature["checkpoint_sha256"],
        "model": signature["model"],
        "inference_policy": {
            key: signature[key]
            for key in ("preprocess_mode", "codec_normalization", "inference_policy")
        },
        "rows": {
            "real": len(manifest_real_paths),
            "fake": len(fake_scores),
            "real_sources": len(by_source),
        },
        "real_source_score_variance_share": observed,
        "permutation_test": {
            "permutations": permutations,
            "seed": seed,
            "p_value_plus_one_correction": permutation_p,
        },
        "interpretation": [
            "The variance share measures how much authentic-score variation is between named real sources after identical evaluator preprocessing.",
            "A high share proves source-dependent scores, not that the model reads file metadata or that source is the causal shortcut.",
            "The 0.5 false-positive fractions are illustrative because no deployment threshold is calibrated.",
            "AUC compares each real source with every frozen fake in the same gate.",
        ],
        "real_sources_ranked_high_to_low_ai_score": ranked,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    result = analyze(
        args.manifest,
        args.progress,
        permutations=args.permutations,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
