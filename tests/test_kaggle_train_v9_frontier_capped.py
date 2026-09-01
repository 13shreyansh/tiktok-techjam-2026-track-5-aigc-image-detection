from __future__ import annotations

import math
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import kaggle_train_v9_frontier_capped as v9  # noqa: E402


def test_frontier_sampler_preserves_declared_block_mass() -> None:
    rows = []
    for source in ("real-a", "real-b"):
        rows.extend(
            {"label": 0, "real_source": source, "family": "real"}
            for _ in range(3)
        )
    for generator in ("legacy-a", "legacy-b", "legacy-c"):
        rows.extend(
            {"label": 1, "generator": generator, "family": "legacy"}
            for _ in range(4)
        )
    for index in range(18):
        rows.extend(
            {
                "label": 1,
                "generator": f"frontier-{index:02d}",
                "family": v9.FRONTIER_FAMILY,
            }
            for _ in range(2)
        )

    sampler, report = v9.frontier_capped_sampler(rows)

    assert sampler.num_samples == len(rows)
    assert math.isclose(report["observed_weight_mass"]["real"], 0.5)
    assert math.isclose(report["observed_weight_mass"]["legacy_fake"], 0.425)
    assert math.isclose(report["observed_weight_mass"]["frontier_fake"], 0.075)
    assert len(report["frontier_fake_group_counts"]) == 18
