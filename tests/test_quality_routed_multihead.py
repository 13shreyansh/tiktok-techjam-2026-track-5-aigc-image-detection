from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import quality_routed_multihead as routed  # noqa: E402


def test_haar_estimator_tracks_gaussian_noise() -> None:
    generator = torch.Generator().manual_seed(7)
    clean = torch.full((16, 3, 224, 224), 0.5)
    noisy = (
        clean
        + torch.randn(clean.shape, generator=generator, dtype=clean.dtype) * 0.10
    ).clamp(0.0, 1.0)
    clean_estimate = routed.haar_noise_estimate(clean)
    noisy_estimate = routed.haar_noise_estimate(noisy)
    assert float(clean_estimate.max()) == 0.0
    assert float(noisy_estimate.min()) > routed.ROUTING_THRESHOLD
    assert float(noisy_estimate.median()) == pytest.approx(0.10, abs=0.005)


def test_quality_route_preserves_general_path_and_switches_noise() -> None:
    estimates = torch.tensor([0.01, routed.ROUTING_THRESHOLD, 0.10])
    v6 = torch.tensor([0.2, 0.3, 0.4])
    v9 = torch.tensor([0.6, 0.7, 0.8])
    v10 = torch.tensor([0.9, 0.1, 0.2])
    score, general, mask = routed.quality_routed_scores(estimates, v6, v9, v10)
    assert general.tolist() == pytest.approx([0.3, 0.4, 0.5])
    assert mask.tolist() == [False, True, True]
    assert score.tolist() == pytest.approx([0.3, 0.1, 0.2])


def test_quality_route_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="one shape"):
        routed.quality_routed_scores(
            torch.zeros(2), torch.zeros(2), torch.zeros(3), torch.zeros(2)
        )

