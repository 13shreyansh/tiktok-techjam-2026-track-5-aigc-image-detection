#!/usr/bin/env python3
"""Audit saved v6/ensemble scores on an exact metadata-matched subset.

This is a diagnosis-only analysis of already-computed predictions. It does not
run a model, change a score, or select a new blend weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from sklearn.metrics import roc_auc_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def metadata_matched_rows(manifest: Path, seed: int) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple, dict[int, list[dict]]] = defaultdict(
        lambda: {0: [], 1: []}
    )
    for row in read_jsonl(manifest):
        path = Path(str(row["path"]))
        if not path.is_absolute():
            path = (manifest.parent / path).resolve()
        with Image.open(path) as image:
            key = (
                str(image.format or "unknown"),
                str(image.mode),
                int(image.width),
                int(image.height),
            )
        groups[key][int(row["label"])].append({**row, "metadata_stratum": key})

    rng = random.Random(seed)
    selected: list[dict] = []
    strata: list[dict] = []
    for key, by_label in sorted(groups.items()):
        count = min(len(by_label[0]), len(by_label[1]))
        if count == 0:
            continue
        for label in (0, 1):
            pool = list(by_label[label])
            rng.shuffle(pool)
            selected.extend(pool[:count])
        strata.append(
            {
                "format": key[0],
                "mode": key[1],
                "width": key[2],
                "height": key[3],
                "selected_per_label": count,
                "available": {"real": len(by_label[0]), "fake": len(by_label[1])},
            }
        )
    rng.shuffle(selected)
    if not selected:
        raise RuntimeError("no metadata stratum contains both labels")
    return selected, strata


def aligned_predictions(path: Path, selected: list[dict]) -> list[dict]:
    predictions = read_jsonl(path)
    by_hash = {str(row["image_sha256"]): row for row in predictions}
    if len(by_hash) != len(predictions):
        raise RuntimeError(f"duplicate prediction hash: {path}")
    result = []
    for source in selected:
        digest = str(source["image_sha256"])
        if digest not in by_hash:
            raise RuntimeError(f"missing prediction for {digest}: {path}")
        row = by_hash[digest]
        if int(row["label"]) != int(source["label"]):
            raise RuntimeError(f"label mismatch for {digest}: {path}")
        result.append(row)
    return result


def auc_for(rows: list[dict], key: str) -> float:
    return float(
        roc_auc_score(
            [int(row["label"]) for row in rows],
            [float(row[key]) for row in rows],
        )
    )


def metric_block(conditions: dict[str, list[dict]], key: str) -> dict:
    clean_auc = auc_for(conditions["clean"], key)
    robust_rows = [
        row
        for name, rows in conditions.items()
        if name != "clean"
        for row in rows
    ]
    pooled_robust_auc = auc_for(robust_rows, key)
    per_condition = {
        name: auc_for(rows, key) for name, rows in sorted(conditions.items())
    }
    return {
        "clean_auc": clean_auc,
        "pooled_robust_auc": pooled_robust_auc,
        "official_score": 0.5 * (clean_auc + pooled_robust_auc),
        "worst_condition": min(per_condition, key=per_condition.get),
        "worst_condition_auc": min(per_condition.values()),
        "noise_sigma_0.10_auc": per_condition["noise_sigma_0.10"],
        "per_condition_auc": per_condition,
    }


def delta(after: dict, before: dict) -> dict:
    keys = (
        "clean_auc",
        "pooled_robust_auc",
        "official_score",
        "worst_condition_auc",
        "noise_sigma_0.10_auc",
    )
    return {key: float(after[key] - before[key]) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-source-rows", type=int, required=True)
    parser.add_argument("--expected-matched-rows", type=int, required=True)
    parser.add_argument("--expected-condition-files", type=int, default=20)
    args = parser.parse_args()

    observed_manifest_sha256 = sha256(args.manifest)
    if observed_manifest_sha256 != args.expected_manifest_sha256:
        raise RuntimeError(
            f"manifest SHA-256 mismatch: {observed_manifest_sha256}"
        )
    source_rows = read_jsonl(args.manifest)
    if len(source_rows) != args.expected_source_rows:
        raise RuntimeError(f"source row mismatch: {len(source_rows)}")
    selected, strata = metadata_matched_rows(args.manifest, args.seed)
    if len(selected) != args.expected_matched_rows:
        raise RuntimeError(f"metadata-matched row mismatch: {len(selected)}")
    label_counts = Counter(int(row["label"]) for row in selected)
    if label_counts[0] != label_counts[1]:
        raise RuntimeError(f"metadata-matched labels are unbalanced: {label_counts}")

    files = sorted(args.predictions_root.glob("*_predictions.jsonl"))
    if len(files) != args.expected_condition_files:
        raise RuntimeError(f"prediction file mismatch: {len(files)}")
    conditions: dict[str, list[dict]] = {}
    inventory = []
    for path in files:
        name = path.name.removesuffix("_predictions.jsonl")
        conditions[name] = aligned_predictions(path, selected)
        inventory.append(f"{name}\t{sha256(path)}")
    if "clean" not in conditions or "noise_sigma_0.10" not in conditions:
        raise RuntimeError("required conditions are absent")

    subset_inventory = "\n".join(
        sorted(f"{int(row['label'])}\t{row['image_sha256']}" for row in selected)
    ) + "\n"
    v6 = metric_block(conditions, "v6_score")
    blend = metric_block(conditions, "score")
    report = {
        "completed": True,
        "scope": "saved predictions on exact format/mode/width/height matched NTIRE rows",
        "diagnosis_only": True,
        "source_manifest_sha256": observed_manifest_sha256,
        "source_rows": len(source_rows),
        "matched_rows": len(selected),
        "matched_label_counts": {str(k): v for k, v in sorted(label_counts.items())},
        "matched_strata": len(strata),
        "matched_subset_inventory_sha256": hashlib.sha256(
            subset_inventory.encode()
        ).hexdigest(),
        "prediction_files": len(files),
        "prediction_inventory_sha256": hashlib.sha256(
            ("\n".join(inventory) + "\n").encode()
        ).hexdigest(),
        "v6": v6,
        "blend": blend,
        "delta_blend_minus_v6": delta(blend, v6),
        "interpretation_boundary": (
            "Exact container and geometry are matched inside retained strata. "
            "This does not remove semantic, acquisition-source or generator shortcuts, "
            "and it cannot estimate the unpublished organizer hidden set."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
