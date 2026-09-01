import math
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import kaggle_train_v10_lowres_paired as v10  # noqa: E402


def test_v10_mass_contract_and_limit():
    assert v10.LOWRES_MASS_WITHIN_LABEL == 0.25
    assert v10.FRONTIER_MASS_WITHIN_FAKE == 0.15
    assert v10.OTHER_REAL_MASS_WITHIN_LABEL == 0.75
    assert v10.OTHER_LEGACY_FAKE_MASS_WITHIN_LABEL == 0.60
    assert v10.EXPECTED_TRAIN_ROWS == 29_534


def test_v10_lowres_classifier_is_label_symmetric():
    assert v10.is_lowres({"label": 0, "real_source": "CIFAKE-CIFAR10"})
    assert v10.is_lowres(
        {"label": 1, "generator": "CIFAKE-Stable-Diffusion", "family": "latent-diffusion"}
    )
    assert not v10.is_lowres({"label": 0, "real_source": "AFHQ-v2"})
    assert not v10.is_lowres(
        {"label": 1, "generator": "FLUX.1-schnell", "family": "diffusion-transformer"}
    )


def test_v10_sampler_preserves_predeclared_block_mass():
    rows = []
    for source in ("CIFAKE-CIFAR10", "CIFAKE-CIFAR10-v10-supplement"):
        rows.extend({"label": 0, "real_source": source, "family": "real"} for _ in range(2))
    for source in ("AFHQ-v2", "FFHQ-train"):
        rows.extend({"label": 0, "real_source": source, "family": "real"} for _ in range(2))
    for generator in (
        "CIFAKE-Stable-Diffusion",
        "CIFAKE-Stable-Diffusion-v10-supplement",
    ):
        rows.extend(
            {"label": 1, "generator": generator, "family": "latent-diffusion"}
            for _ in range(2)
        )
    for generator in ("legacy-a", "legacy-b"):
        rows.extend(
            {"label": 1, "generator": generator, "family": "legacy"}
            for _ in range(2)
        )
    for index in range(18):
        rows.extend(
            {
                "label": 1,
                "generator": f"frontier-{index:02d}",
                "family": v10.FRONTIER_FAMILY,
            }
            for _ in range(2)
        )
    sampler, report = v10.lowres_paired_sampler(rows)
    assert sampler.num_samples == len(rows)
    expected = {
        "real_lowres": 0.125,
        "real_other": 0.375,
        "fake_lowres": 0.125,
        "fake_frontier": 0.075,
        "fake_other": 0.30,
    }
    for key, value in expected.items():
        assert math.isclose(report["observed_weight_mass"][key], value)
