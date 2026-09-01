#!/usr/bin/env python3
"""Reconcile new CIFAR-100 real scores with frozen Qwen fake scores.

Only the authentic side is new. The Qwen fake predictions are checksum-pinned
outputs from the already-consumed first promotion gate. This analysis changes
no model, threshold, calibration, blend weight or training row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


QWEN_HASHES = {
    "clean": "26372ace7f11ae079d0fe88c8cc0eefd91aa975992b3b51a6f75e595de7da8be",
    "noise_sigma_0.10": "58e9768f356ccd3f7ca078c84002aa41ade8eeb5934cd040a23001e38818e931",
}
EXPECTED_REAL_ROWS = 1000
EXPECTED_FAKE_ROWS = 288
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260831


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def auc(real: np.ndarray, fake: np.ndarray) -> float:
    labels = np.concatenate(
        [np.zeros(real.size, dtype=np.int8), np.ones(fake.size, dtype=np.int8)]
    )
    scores = np.concatenate([real, fake])
    return float(roc_auc_score(labels, scores))


def bootstrap_pair(
    real_v6: np.ndarray,
    fake_v6: np.ndarray,
    real_blend: np.ndarray,
    fake_blend: np.ndarray,
) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty((BOOTSTRAP_REPLICATES, 3), dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        real_indices = rng.integers(0, real_v6.size, real_v6.size)
        fake_indices = rng.integers(0, fake_v6.size, fake_v6.size)
        v6_value = auc(real_v6[real_indices], fake_v6[fake_indices])
        blend_value = auc(real_blend[real_indices], fake_blend[fake_indices])
        values[index] = (v6_value, blend_value, blend_value - v6_value)
    names = ("v6_auc", "blend_auc", "delta_blend_minus_v6")
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        **{
            name: {
                "q025": float(np.quantile(values[:, column], 0.025)),
                "median": float(np.quantile(values[:, column], 0.5)),
                "q975": float(np.quantile(values[:, column], 0.975)),
            }
            for column, name in enumerate(names)
        },
    }


def condition_metrics(real_rows: list[dict], fake_rows: list[dict]) -> dict:
    if len(real_rows) != EXPECTED_REAL_ROWS or len(fake_rows) != EXPECTED_FAKE_ROWS:
        raise ValueError(
            f"row mismatch: real={len(real_rows)}, fake={len(fake_rows)}"
        )
    if any(int(row["label"]) != 0 for row in real_rows):
        raise ValueError("new gate contains a non-authentic row")
    if any(int(row["label"]) != 1 for row in fake_rows):
        raise ValueError("frozen reference contains a non-fake row")
    real_hashes = [row["image_sha256"] for row in real_rows]
    fake_hashes = [row["image_sha256"] for row in fake_rows]
    if len(set(real_hashes)) != len(real_hashes) or len(set(fake_hashes)) != len(fake_hashes):
        raise ValueError("duplicate hashes in a gate side")
    if set(real_hashes).intersection(fake_hashes):
        raise ValueError("real/fake byte overlap")

    arrays = {
        "real_v6": np.asarray([float(row["v6_score"]) for row in real_rows]),
        "fake_v6": np.asarray([float(row["v6_score"]) for row in fake_rows]),
        "real_blend": np.asarray([float(row["score"]) for row in real_rows]),
        "fake_blend": np.asarray([float(row["score"]) for row in fake_rows]),
    }
    v6_auc = auc(arrays["real_v6"], arrays["fake_v6"])
    blend_auc = auc(arrays["real_blend"], arrays["fake_blend"])
    per_fine_class = []
    for fine_label in sorted({int(row["fine_label"]) for row in real_rows}):
        class_rows = [row for row in real_rows if int(row["fine_label"]) == fine_label]
        if len(class_rows) != 10:
            raise ValueError(f"fine class {fine_label} has {len(class_rows)} rows")
        per_fine_class.append(
            {
                "fine_label": fine_label,
                "real_rows": len(class_rows),
                "v6_auc": auc(
                    np.asarray([float(row["v6_score"]) for row in class_rows]),
                    arrays["fake_v6"],
                ),
                "blend_auc": auc(
                    np.asarray([float(row["score"]) for row in class_rows]),
                    arrays["fake_blend"],
                ),
                "blend_real_mean": float(
                    np.mean([float(row["score"]) for row in class_rows])
                ),
            }
        )
    return {
        "rows": {"real": len(real_rows), "fake": len(fake_rows)},
        "v6_auc": v6_auc,
        "blend_auc": blend_auc,
        "delta_blend_minus_v6": blend_auc - v6_auc,
        "means": {
            key: float(value.mean()) for key, value in arrays.items()
        },
        "mean_score_inversion": bool(
            arrays["real_blend"].mean() > arrays["fake_blend"].mean()
        ),
        "illustrative_fraction_real_at_or_above_0.5": {
            "v6": float(np.mean(arrays["real_v6"] >= 0.5)),
            "blend": float(np.mean(arrays["real_blend"] >= 0.5)),
        },
        "bootstrap_95_percentile_interval": bootstrap_pair(**arrays),
        "worst_fine_classes_by_blend_auc": sorted(
            per_fine_class, key=lambda row: (row["blend_auc"], row["fine_label"])
        )[:10],
        "best_fine_classes_by_blend_auc": sorted(
            per_fine_class,
            key=lambda row: (-row["blend_auc"], row["fine_label"]),
        )[:10],
    }


def interpretation(noise: dict, clean: dict) -> dict:
    score = float(noise["blend_auc"])
    if score < 0.50:
        band = "general low-resolution authentic ranking inversion confirmed on the new source"
    elif score < 0.70:
        band = "severe general low-resolution/noise failure"
    elif score < 0.80:
        band = "material risk; inconclusive between general failure and strong source dependence"
    else:
        band = "CIFAKE-specific or source-interaction explanation becomes more likely; broad robustness remains unproved"
    flags = {
        "clean_auc_below_0.80": clean["blend_auc"] < 0.80,
        "noise_authentic_mean_above_fake_mean": noise["mean_score_inversion"],
        "blend_noise_auc_more_than_0.01_below_v6": (
            noise["delta_blend_minus_v6"] < -0.01
        ),
    }
    return {
        "predeclared_noise_band": band,
        "flags": flags,
        "repair_priority": score < 0.70 or any(flags.values()),
        "decision": "Diagnosis consumed. Do not train, tune, calibrate, select a threshold or select a blend using these rows.",
    }


def analyze(
    report_path: Path,
    real_clean_path: Path,
    real_noise_path: Path,
    qwen_clean_path: Path,
    qwen_noise_path: Path,
) -> dict:
    report = json.loads(report_path.read_text())
    if report.get("completed") is not True:
        raise ValueError("Kaggle report is not complete")
    paths = {
        "clean": (real_clean_path, qwen_clean_path),
        "noise_sigma_0.10": (real_noise_path, qwen_noise_path),
    }
    conditions = {}
    inputs = {"kaggle_report_sha256": sha256_file(report_path)}
    for name, (real_path, qwen_path) in paths.items():
        expected_real_hash = report["outputs"][name]["sha256"]
        if sha256_file(real_path) != expected_real_hash:
            raise ValueError(f"new real prediction hash mismatch for {name}")
        if sha256_file(qwen_path) != QWEN_HASHES[name]:
            raise ValueError(f"frozen Qwen prediction hash mismatch for {name}")
        real_rows = read_jsonl(real_path)
        qwen_rows = read_jsonl(qwen_path)
        fake_rows = [row for row in qwen_rows if int(row["label"]) == 1]
        conditions[name] = condition_metrics(real_rows, fake_rows)
        inputs[f"real_{name}_sha256"] = expected_real_hash
        inputs[f"qwen_{name}_sha256"] = QWEN_HASHES[name]
    return {
        "completed": True,
        "purpose": "New CIFAR-100 authentic-side diagnosis against unchanged frozen Qwen fakes; no new fake-family evidence.",
        "inputs": inputs,
        "conditions": conditions,
        "interpretation": interpretation(
            conditions["noise_sigma_0.10"], conditions["clean"]
        ),
        "training_or_selection_change": "none",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--real-clean", type=Path, required=True)
    parser.add_argument("--real-noise", type=Path, required=True)
    parser.add_argument("--qwen-clean", type=Path, required=True)
    parser.add_argument("--qwen-noise", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.report,
        args.real_clean,
        args.real_noise,
        args.qwen_clean,
        args.qwen_noise,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
