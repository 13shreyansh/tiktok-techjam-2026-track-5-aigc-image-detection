#!/usr/bin/env python3
"""Ablate v8's frontier sampling mass while preserving every other factor.

V8 gave each named fake generator equal mass.  Because the 576 Qwen Image
Bench rows introduce 18 new generator names, those rows collectively received
about 60% of the fake-label draws despite being only 5.7% of the unique fake
training images.  V9 keeps the same images, frozen gates, seed, model,
optimizer, and augmentation but caps the frontier block at 15% of fake draws.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import WeightedRandomSampler

import kaggle_train_v3 as runner
import kaggle_train_v8_frontier as v8


OUTPUT_ROOT = Path("/kaggle/working/track5-v9-frontier-capped-candidate")
FRONTIER_FAKE_MASS = 0.15
LEGACY_FAKE_MASS = 1.0 - FRONTIER_FAKE_MASS
FRONTIER_FAMILY = "frontier-2026-image-generation"


def frontier_capped_sampler(
    rows: list[dict],
) -> tuple[WeightedRandomSampler, dict]:
    real_groups = Counter(
        str(row.get("real_source", "unknown"))
        for row in rows
        if int(row["label"]) == 0
    )
    legacy_groups = Counter(
        str(row.get("generator", "unknown"))
        for row in rows
        if int(row["label"]) == 1 and row.get("family") != FRONTIER_FAMILY
    )
    frontier_groups = Counter(
        str(row.get("generator", "unknown"))
        for row in rows
        if int(row["label"]) == 1 and row.get("family") == FRONTIER_FAMILY
    )
    if not real_groups or not legacy_groups or len(frontier_groups) != 18:
        raise RuntimeError(
            "unexpected sampler composition: "
            f"real={real_groups}, legacy={legacy_groups}, frontier={frontier_groups}"
        )
    if "unknown" in real_groups or "unknown" in legacy_groups or "unknown" in frontier_groups:
        raise RuntimeError("unnamed source group in v9 sampler")

    weights: list[float] = []
    for row in rows:
        label = int(row["label"])
        if label == 0:
            group = str(row["real_source"])
            weight = 0.5 / (len(real_groups) * real_groups[group])
        elif row.get("family") == FRONTIER_FAMILY:
            group = str(row["generator"])
            weight = (
                0.5
                * FRONTIER_FAKE_MASS
                / (len(frontier_groups) * frontier_groups[group])
            )
        else:
            group = str(row["generator"])
            weight = (
                0.5
                * LEGACY_FAKE_MASS
                / (len(legacy_groups) * legacy_groups[group])
            )
        weights.append(weight)

    observed_mass = {
        "real": sum(
            weight for row, weight in zip(rows, weights) if int(row["label"]) == 0
        ),
        "legacy_fake": sum(
            weight
            for row, weight in zip(rows, weights)
            if int(row["label"]) == 1 and row.get("family") != FRONTIER_FAMILY
        ),
        "frontier_fake": sum(
            weight
            for row, weight in zip(rows, weights)
            if int(row["label"]) == 1 and row.get("family") == FRONTIER_FAMILY
        ),
    }
    expected_mass = {
        "real": 0.5,
        "legacy_fake": 0.5 * LEGACY_FAKE_MASS,
        "frontier_fake": 0.5 * FRONTIER_FAKE_MASS,
    }
    if any(abs(observed_mass[key] - expected_mass[key]) > 1e-9 for key in expected_mass):
        raise RuntimeError(f"sampler mass mismatch: {observed_mass}")

    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(rows),
        replacement=True,
        generator=torch.Generator().manual_seed(runner.SEED),
    )
    report = {
        "policy": (
            "equal labels; equal real sources; frontier block capped to 15% "
            "of fake draws; equal named sources within legacy/frontier blocks"
        ),
        "target_mass": expected_mass,
        "observed_weight_mass": observed_mass,
        "real_group_counts": dict(sorted(real_groups.items())),
        "legacy_fake_group_counts": dict(sorted(legacy_groups.items())),
        "frontier_fake_group_counts": dict(sorted(frontier_groups.items())),
    }
    return sampler, report


def main() -> None:
    v8.OUTPUT_ROOT = OUTPUT_ROOT
    runner.source_balanced_sampler = frontier_capped_sampler
    v8.main()
    policy = {
        "candidate": "v9-frontier-capped",
        "parent": "v8-frontier-balanced",
        "single_changed_factor": "aggregate frontier sampling mass",
        "frontier_share_of_fake_draws": FRONTIER_FAKE_MASS,
        "legacy_share_of_fake_draws": LEGACY_FAKE_MASS,
        "predeclared_internal_rejection": {
            "maximum_clean_auc_drop_from_v6": 0.002,
            "maximum_worst_pair_auc_drop_from_v6": 0.01,
        },
        "promotion_requires": [
            "internal frozen clean gate passes",
            "sealed Qwen frontier holdout improves meaningfully",
            "full NTIRE independent gate does not regress materially",
            "workshop individual-transform robustness does not regress materially",
        ],
    }
    (OUTPUT_ROOT / "ablation-policy.json").write_text(
        json.dumps(policy, indent=2) + "\n"
    )
    print(json.dumps(policy, indent=2), flush=True)


if __name__ == "__main__":
    main()
