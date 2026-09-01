#!/usr/bin/env python3
"""Measure prompt versus generator dependence in a balanced fake-image grid."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def analyze(manifest_path: Path, progress_path: Path) -> dict[str, Any]:
    rows = read_jsonl(manifest_path)
    progress = json.loads(progress_path.read_text())
    clean = progress["predictions"]["clean"]
    paths = [str(Path(path).resolve()) for path in clean["paths"]]
    scores = [float(score) for score in clean["scores"]]
    labels = [int(label) for label in clean["labels"]]
    if len(paths) != len(scores) or len(paths) != len(labels) or len(paths) != len(set(paths)):
        raise RuntimeError("prediction paths must be unique and align with scores")
    score_by_path = dict(zip(paths, scores))
    real_scores = [score for score, label in zip(scores, labels) if label == 0]
    if not real_scores:
        raise RuntimeError("at least one real score is required for prompt-specific AUC")

    cells: list[dict[str, Any]] = []
    for row in rows:
        if int(row["label"]) != 1:
            continue
        path = str((manifest_path.parent / row["path"]).resolve())
        if path not in score_by_path:
            raise RuntimeError(f"missing frozen prediction for {path}")
        cells.append(
            {
                "prompt_id": int(row["prompt_id"]),
                "generator": str(row["generator_model"]),
                "score": score_by_path[path],
            }
        )

    by_prompt: dict[int, list[float]] = defaultdict(list)
    by_generator: dict[str, list[float]] = defaultdict(list)
    cell_keys = set()
    for cell in cells:
        key = (cell["prompt_id"], cell["generator"])
        if key in cell_keys:
            raise RuntimeError(f"duplicate prompt/generator cell: {key}")
        cell_keys.add(key)
        by_prompt[cell["prompt_id"]].append(cell["score"])
        by_generator[cell["generator"]].append(cell["score"])

    prompt_counts = {len(values) for values in by_prompt.values()}
    generator_counts = {len(values) for values in by_generator.values()}
    if len(prompt_counts) != 1 or len(generator_counts) != 1:
        raise RuntimeError("prompt/generator grid is not balanced")
    expected = len(by_prompt) * len(by_generator)
    if len(cells) != expected:
        raise RuntimeError(f"incomplete grid: {len(cells)} != {expected}")

    all_scores = [cell["score"] for cell in cells]
    grand = statistics.fmean(all_scores)
    prompt_means = {key: statistics.fmean(values) for key, values in by_prompt.items()}
    generator_means = {
        key: statistics.fmean(values) for key, values in by_generator.items()
    }
    total_ss = sum((score - grand) ** 2 for score in all_scores)
    prompt_ss = len(by_generator) * sum((mean - grand) ** 2 for mean in prompt_means.values())
    generator_ss = len(by_prompt) * sum(
        (mean - grand) ** 2 for mean in generator_means.values()
    )
    residual_ss = max(0.0, total_ss - prompt_ss - generator_ss)

    def auc_against_reals(fake_scores: list[float]) -> float:
        wins = sum(
            float(fake > real) + 0.5 * float(fake == real)
            for fake in fake_scores
            for real in real_scores
        )
        return wins / (len(fake_scores) * len(real_scores))

    def rows_for(groups: dict[Any, list[float]]) -> list[dict[str, Any]]:
        result = []
        for key, values in groups.items():
            result.append(
                {
                    "name": key,
                    "count": len(values),
                    "mean_score": statistics.fmean(values),
                    "median_score": statistics.median(values),
                    "minimum_score": min(values),
                    "maximum_score": max(values),
                    "auc_against_all_reals": auc_against_reals(values),
                    "fraction_at_or_above_0.5": sum(value >= 0.5 for value in values)
                    / len(values),
                }
            )
        return sorted(result, key=lambda row: float(row["mean_score"]))

    shares = {
        "prompt": prompt_ss / total_ss if total_ss else 0.0,
        "generator": generator_ss / total_ss if total_ss else 0.0,
        "unexplained_prompt_generator_interaction": residual_ss / total_ss
        if total_ss
        else 0.0,
    }
    return {
        "purpose": "Frozen clean-score prompt/content dependence diagnosis; not model selection.",
        "checkpoint": progress["signature"]["checkpoint"],
        "checkpoint_sha256": progress["signature"]["checkpoint_sha256"],
        "model": progress["signature"]["model"],
        "inference_policy": {
            key: progress["signature"][key]
            for key in ("preprocess_mode", "codec_normalization", "inference_policy")
        },
        "grid": {
            "fake_images": len(cells),
            "prompts": len(by_prompt),
            "generators": len(by_generator),
            "complete_balanced_grid": True,
        },
        "fake_score_mean": grand,
        "sum_of_squares_share": shares,
        "interpretation": [
            "Prompt share measures score dependence on repeated prompt identity across generators.",
            "Generator share measures score dependence on generator identity across prompts.",
            "The residual includes prompt-generator interaction because there is one image per cell.",
            "Scores are AI-positive probabilities; the 0.5 fractions are illustrative, not calibrated.",
            "Prompt-specific AUC compares 18 same-prompt fakes with every frozen real in the gate.",
            "Low prompt share weakens but cannot eliminate the content-shortcut hypothesis.",
        ],
        "prompts_ranked_low_to_high": rows_for(by_prompt),
        "generators_ranked_low_to_high": rows_for(by_generator),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.manifest, args.progress)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
