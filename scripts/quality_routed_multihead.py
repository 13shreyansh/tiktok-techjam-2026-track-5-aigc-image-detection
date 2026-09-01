#!/usr/bin/env python3
"""Shared-backbone, label-blind quality routing primitives for v11.

The routing statistic is computed from pixels before normalization.  It does
not use filenames, dimensions, labels, model scores or evaluation outcomes.
The fixed threshold was frozen from clean versus sigma-0.10 transformations on
training-only images before any v11 gate was scored.
"""

from __future__ import annotations

import torch


MAD_NORMAL_SCALE = 0.67448975
ROUTING_THRESHOLD = 0.055
GENERAL_V6_WEIGHT = 0.75
GENERAL_V9_WEIGHT = 0.25


def haar_noise_estimate(images: torch.Tensor) -> torch.Tensor:
    """Estimate per-image high-frequency noise from an RGB BCHW tensor.

    The diagonal 2-by-2 Haar coefficient of independent Gaussian noise with
    standard deviation sigma also has standard deviation sigma.  Median
    absolute deviation makes the estimate less sensitive to natural edges.
    Inputs must be unnormalized float pixels in the unit interval.
    """

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("expected an RGB BCHW tensor")
    if not torch.is_floating_point(images):
        raise ValueError("expected floating-point pixels")
    height = int(images.shape[-2]) // 2 * 2
    width = int(images.shape[-1]) // 2 * 2
    if height < 2 or width < 2:
        raise ValueError("images must be at least 2 by 2 pixels")
    cropped = images[..., :height, :width]
    diagonal = (
        cropped[..., 0::2, 0::2]
        - cropped[..., 0::2, 1::2]
        - cropped[..., 1::2, 0::2]
        + cropped[..., 1::2, 1::2]
    ) / 2.0
    return diagonal.abs().flatten(1).median(dim=1).values / MAD_NORMAL_SCALE


def quality_routed_scores(
    noise_estimates: torch.Tensor,
    v6_scores: torch.Tensor,
    v9_scores: torch.Tensor,
    v10_scores: torch.Tensor,
    threshold: float = ROUTING_THRESHOLD,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return routed, general-path and route-mask tensors.

    The normal path preserves the already-frozen 75/25 v6/v9 probability
    blend.  Only images whose label-blind noise estimate crosses the frozen
    threshold use v10.  All tensors must contain one scalar per image.
    """

    shapes = {
        tuple(value.shape)
        for value in (noise_estimates, v6_scores, v9_scores, v10_scores)
    }
    if len(shapes) != 1:
        raise ValueError(f"score tensors must share one shape, got {shapes}")
    general = GENERAL_V6_WEIGHT * v6_scores + GENERAL_V9_WEIGHT * v9_scores
    route_mask = noise_estimates >= float(threshold)
    return torch.where(route_mask, v10_scores, general), general, route_mask

