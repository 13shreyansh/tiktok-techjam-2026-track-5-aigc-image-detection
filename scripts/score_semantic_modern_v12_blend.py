#!/usr/bin/env python3
"""Score the predeclared 50/50 v12 blend on the frozen semantic gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import roc_auc_score

import kaggle_evaluate_semantic_modern_v12_gate as semantic
import score_fixed_v12_blend as fixed


def score(pe_root: Path, dino_root: Path, manifest: Path) -> dict:
    rows = semantic.read_rows(manifest)
    compliance = semantic.validate_gate_rows(rows)
    integrity = {
        "pe_core": fixed.validate_progress(pe_root, "pe_core"),
        "dinov2_control": fixed.validate_progress(dino_root, "dinov2_control"),
    }
    condition_auc: dict[str, float] = {}
    prediction_hashes: dict[str, dict[str, str]] = {}
    pooled_labels: list[int] = []
    pooled_scores: list[float] = []
    clean_predictions = None
    for condition in fixed.CONDITIONS:
        pe_path = pe_root / f"{condition}_predictions.jsonl"
        dino_path = dino_root / f"{condition}_predictions.jsonl"
        pe_rows = fixed.read_jsonl(pe_path)
        dino_rows = fixed.read_jsonl(dino_path)
        if len(pe_rows) != len(rows) or len(dino_rows) != len(rows):
            raise RuntimeError(f"row-count mismatch: {condition}")
        if [fixed.identity(row) for row in pe_rows] != [fixed.identity(row) for row in dino_rows]:
            raise RuntimeError(f"prediction identity mismatch: {condition}")
        labels = [int(row["label"]) for row in pe_rows]
        scores = [
            0.5 * float(pe["score"]) + 0.5 * float(dino["score"])
            for pe, dino in zip(pe_rows, dino_rows)
        ]
        condition_auc[condition] = float(roc_auc_score(labels, scores))
        prediction_hashes[condition] = {
            "pe_core": fixed.sha256(pe_path),
            "dinov2_control": fixed.sha256(dino_path),
        }
        if condition == "clean":
            clean_predictions = [
                {"index": int(row["index"]), "label": int(row["label"]), "score": score}
                for row, score in zip(pe_rows, scores)
            ]
        else:
            pooled_labels.extend(labels)
            pooled_scores.extend(scores)
    if clean_predictions is None:
        raise RuntimeError("clean blend predictions missing")
    clean_semantic = semantic.semantic_metrics(rows, clean_predictions)
    pooled = float(roc_auc_score(pooled_labels, pooled_scores))
    official = 0.5 * clean_semantic["overall_auc"] + 0.5 * pooled
    worst = min(condition_auc.values())
    return {
        "status": "completed",
        "rule": "0.5 * pe_core_probability + 0.5 * dinov2_control_probability",
        "weights_searched": False,
        "gate_compliance": compliance,
        "checkpoint_sha256": fixed.EXPECTED,
        "rows_per_condition": len(rows),
        "conditions": condition_auc,
        "clean_semantic_metrics": clean_semantic,
        "clean_auc": clean_semantic["overall_auc"],
        "pooled_robust_auc": pooled,
        "official_style_score": official,
        "worst_individual_condition_auc": worst,
        "frozen_gate_decision": semantic.gate_decision(clean_semantic, official, worst),
        "integrity": integrity,
        "prediction_sha256": prediction_hashes,
        "boundary": "The blend was fixed before this one-shot audit. Passing is not hidden-set proof and these rows remain forbidden for tuning.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pe-root", type=Path, required=True)
    parser.add_argument("--dino-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = score(args.pe_root, args.dino_root, args.manifest)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
