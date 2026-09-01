#!/usr/bin/env python3
"""Audit clean/noise failures by generator, real source and their pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from sklearn.metrics import roc_auc_score


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def generator_name(row: dict[str, Any]) -> str:
    return str(row.get("generator_model") or row.get("generator") or "unspecified")


def auc(rows: list[dict[str, Any]], score_key: str) -> float:
    labels = [int(row["label"]) for row in rows]
    if len(set(labels)) != 2:
        raise ValueError("AUC requires both labels")
    return float(roc_auc_score(labels, [float(row[score_key]) for row in rows]))


def grouped_auc(
    rows: list[dict[str, Any]],
    fake_key: Callable[[dict[str, Any]], str],
    real_key: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    fakes = [row for row in rows if int(row["label"]) == 1]
    reals = [row for row in rows if int(row["label"]) == 0]
    fake_groups = sorted({fake_key(row) for row in fakes})
    real_groups = sorted({real_key(row) for row in reals})

    def metrics(sample: list[dict[str, Any]]) -> dict[str, float | int]:
        sample_reals = [row for row in sample if int(row["label"]) == 0]
        sample_fakes = [row for row in sample if int(row["label"]) == 1]
        v6 = auc(sample, "v6_score")
        blend = auc(sample, "score")
        return {
            "rows": len(sample),
            "real_rows": len(sample_reals),
            "fake_rows": len(sample_fakes),
            "v6_auc": v6,
            "blend_auc": blend,
            "blend_delta": blend - v6,
            "v6_real_mean": sum(float(row["v6_score"]) for row in sample_reals) / len(sample_reals),
            "v6_fake_mean": sum(float(row["v6_score"]) for row in sample_fakes) / len(sample_fakes),
            "blend_real_mean": sum(float(row["score"]) for row in sample_reals) / len(sample_reals),
            "blend_fake_mean": sum(float(row["score"]) for row in sample_fakes) / len(sample_fakes),
        }

    by_fake = {
        fake: metrics(reals + [row for row in fakes if fake_key(row) == fake])
        for fake in fake_groups
    }
    by_real = {
        real: metrics(fakes + [row for row in reals if real_key(row) == real])
        for real in real_groups
    }
    pairs = {
        f"{fake} || {real}": metrics(
            [row for row in fakes if fake_key(row) == fake]
            + [row for row in reals if real_key(row) == real]
        )
        for fake in fake_groups
        for real in real_groups
    }
    return {
        "overall": metrics(rows),
        "by_fake_generator": by_fake,
        "by_real_source": by_real,
        "generator_real_source_pairs": pairs,
    }


def audit_gate(clean_path: Path, noise_path: Path) -> dict[str, Any]:
    clean = read_jsonl(clean_path)
    noise = read_jsonl(noise_path)
    clean_keys = [(row["image_sha256"], int(row["label"])) for row in clean]
    noise_keys = [(row["image_sha256"], int(row["label"])) for row in noise]
    if clean_keys != noise_keys:
        raise ValueError("clean and noise rows are not identical and ordered")

    real_key = lambda row: str(row.get("real_source") or "unspecified")
    clean_metrics = grouped_auc(clean, generator_name, real_key)
    noise_metrics = grouped_auc(noise, generator_name, real_key)

    pair_rows = []
    for pair, clean_pair in clean_metrics["generator_real_source_pairs"].items():
        noise_pair = noise_metrics["generator_real_source_pairs"][pair]
        pair_rows.append(
            {
                "pair": pair,
                "clean_blend_auc": clean_pair["blend_auc"],
                "noise_blend_auc": noise_pair["blend_auc"],
                "noise_drop": noise_pair["blend_auc"] - clean_pair["blend_auc"],
                "clean_blend_delta": clean_pair["blend_delta"],
                "noise_blend_delta": noise_pair["blend_delta"],
            }
        )

    return {
        "rows": len(clean),
        "clean": clean_metrics,
        "noise_sigma_0.10": noise_metrics,
        "worst_clean_blend_pairs": sorted(pair_rows, key=lambda row: row["clean_blend_auc"])[:10],
        "worst_noise_blend_pairs": sorted(pair_rows, key=lambda row: row["noise_blend_auc"])[:10],
        "largest_noise_pair_drops": sorted(pair_rows, key=lambda row: row["noise_drop"])[:10],
        "smallest_clean_blend_deltas": sorted(pair_rows, key=lambda row: row["clean_blend_delta"])[:10],
        "smallest_noise_blend_deltas": sorted(pair_rows, key=lambda row: row["noise_blend_delta"])[:10],
    }


def parse_gate(value: str) -> tuple[str, Path, Path]:
    name, separator, paths = value.partition("=")
    clean, separator2, noise = paths.partition(",")
    if not separator or not separator2 or not name or not clean or not noise:
        raise argparse.ArgumentTypeError("expected NAME=CLEAN_JSONL,NOISE_JSONL")
    return name, Path(clean), Path(noise)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="append", type=parse_gate, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "purpose": "Frozen prediction audit only; no training, thresholding or model selection.",
        "gates": {name: audit_gate(clean, noise) for name, clean, noise in args.gate},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
