#!/usr/bin/env python3
"""Score a frozen equal-probability blend from two saved prediction matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sklearn.metrics import roc_auc_score


CONDITIONS = (
    "clean",
    "jpeg_q90",
    "jpeg_q70",
    "jpeg_q50",
    "jpeg_q30",
    "blur_sigma_0.5",
    "blur_sigma_1",
    "blur_sigma_2",
    "resize_0.5",
    "resize_0.25",
    "noise_sigma_0.02",
    "noise_sigma_0.05",
    "noise_sigma_0.10",
    "brightness_0.8",
    "brightness_1.2",
    "contrast_0.8",
    "contrast_1.2",
    "saturation_0.8",
    "saturation_1.2",
    "center_crop_80",
)
EXPECTED = {
    "pe_core": "f37bd6b445b12257ff29a9e54946c5bf9a9184e86a45dc5490537b9ea325ddd2",
    "dinov2_control": "db07f30cbc94e4972f4a8c72c95bbe5df0dcd40b2dbd494ad82dc324cc1e2b5b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def identity(row: dict) -> tuple[int, int, str]:
    return int(row["index"]), int(row["label"]), str(row["image_sha256"])


def validate_progress(root: Path, candidate: str) -> dict:
    path = root / "progress.json"
    progress = json.loads(path.read_text())
    signature = progress.get("signature", {})
    if not progress.get("completed"):
        raise RuntimeError(f"incomplete prediction matrix: {root}")
    if signature.get("candidate") != candidate:
        raise RuntimeError(f"candidate mismatch: {root}")
    if signature.get("checkpoint_sha256") != EXPECTED[candidate]:
        raise RuntimeError(f"checkpoint mismatch: {root}")
    if tuple(signature.get("conditions", ())) != CONDITIONS:
        raise RuntimeError(f"condition order mismatch: {root}")
    return {"path": str(path), "sha256": sha256(path), "signature": signature}


def score(pe_root: Path, dino_root: Path) -> dict:
    integrity = {
        "pe_core": validate_progress(pe_root, "pe_core"),
        "dinov2_control": validate_progress(dino_root, "dinov2_control"),
    }
    condition_auc: dict[str, float] = {}
    pooled_labels: list[int] = []
    pooled_scores: list[float] = []
    prediction_hashes: dict[str, dict[str, str]] = {}
    rows_per_condition = None
    for condition in CONDITIONS:
        pe_path = pe_root / f"{condition}_predictions.jsonl"
        dino_path = dino_root / f"{condition}_predictions.jsonl"
        pe_rows = read_jsonl(pe_path)
        dino_rows = read_jsonl(dino_path)
        if len(pe_rows) != len(dino_rows) or not pe_rows:
            raise RuntimeError(f"row-count mismatch: {condition}")
        if rows_per_condition is None:
            rows_per_condition = len(pe_rows)
        elif rows_per_condition != len(pe_rows):
            raise RuntimeError(f"condition row-count mismatch: {condition}")
        if [identity(row) for row in pe_rows] != [identity(row) for row in dino_rows]:
            raise RuntimeError(f"prediction identity mismatch: {condition}")
        labels = [int(row["label"]) for row in pe_rows]
        scores = [
            0.5 * float(pe["score"]) + 0.5 * float(dino["score"])
            for pe, dino in zip(pe_rows, dino_rows)
        ]
        if any(not 0.0 <= value <= 1.0 for value in scores):
            raise RuntimeError(f"probability out of range: {condition}")
        condition_auc[condition] = float(roc_auc_score(labels, scores))
        prediction_hashes[condition] = {
            "pe_core": sha256(pe_path),
            "dinov2_control": sha256(dino_path),
        }
        if condition != "clean":
            pooled_labels.extend(labels)
            pooled_scores.extend(scores)
    clean = condition_auc["clean"]
    pooled = float(roc_auc_score(pooled_labels, pooled_scores))
    worst = min(condition_auc.values())
    official = 0.5 * clean + 0.5 * pooled
    return {
        "status": "completed",
        "rule": "0.5 * pe_probability + 0.5 * dinov2_probability",
        "weights_searched": False,
        "checkpoint_sha256": EXPECTED,
        "rows_per_condition": rows_per_condition,
        "conditions": condition_auc,
        "clean_auc": clean,
        "pooled_robust_auc": pooled,
        "official_style_score": official,
        "worst_individual_condition_auc": worst,
        "integrity": integrity,
        "prediction_sha256": prediction_hashes,
        "boundary": (
            "This reuses opened diagnostic predictions. It measures fixed-blend "
            "complementarity but is not fresh hidden-transfer evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pe-root", type=Path, required=True)
    parser.add_argument("--dino-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = score(args.pe_root, args.dino_root)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
