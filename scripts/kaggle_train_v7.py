#!/usr/bin/env python3
"""Train the fixed v6 PE head with a noise-weighted single-transform policy.

This is a controlled response to the same candidate's completed failure-mode
matrix: Gaussian noise is the weakest transform both on the fixed v6 gate and
on the audit-only 78-model external gate.  The data, split, source-balanced
sampler, model, optimizer and one-epoch budget remain unchanged.
"""

from __future__ import annotations

import random
from pathlib import Path

import kaggle_train_v3 as runner
import kaggle_train_v6  # noqa: F401  # apply checksum-pinned v6 constants


class NoiseWeightedSingleOfficialTransform:
    """Apply no more than one listed transform, with extra noise exposure."""

    def __init__(self, probability: float = 0.9) -> None:
        self.probability = probability
        self.transforms = (
            [runner.JpegCompression(quality) for quality in (90, 70, 50, 30)]
            + [runner.GaussianBlurPIL(sigma) for sigma in (0.5, 1.0, 2.0)]
            + [runner.DownUpResize(scale) for scale in (0.5, 0.25)]
            + [runner.GaussianNoisePIL(sigma) for sigma in (0.02, 0.05, 0.10)]
            + [
                runner.FixedEnhancement(kind, factor)
                for kind in ("brightness", "contrast", "saturation")
                for factor in (0.8, 1.2)
            ]
            + [runner.CenterCropFraction(0.8)]
        )
        # Conditional on applying a transform, 40% of draws are Gaussian
        # noise.  Every other workshop category remains represented.  No draw
        # ever chains two listed transformations.
        self.weights = (
            [0.0375] * 4
            + [0.1 / 3] * 3
            + [0.05] * 2
            + [0.15, 0.15, 0.10]
            + [0.025] * 6
            + [0.10]
        )
        if len(self.transforms) != len(self.weights):
            raise RuntimeError("transform/weight length mismatch")
        if abs(sum(self.weights) - 1.0) > 1e-9:
            raise RuntimeError(f"transform weights do not sum to one: {self.weights}")

    def __call__(self, image):
        if random.random() >= self.probability:
            return image
        transform = random.choices(self.transforms, weights=self.weights, k=1)[0]
        return transform(image)


runner.RandomSingleOfficialTransform = NoiseWeightedSingleOfficialTransform
runner.AUGMENTATION_DESCRIPTION = (
    "at_most_one_workshop_transform_noise_weighted_40pct_conditional_"
    "90pct_application"
)
runner.OUTPUT_ROOT = Path("/kaggle/working/track5-v7-noise-weighted-candidate")


if __name__ == "__main__":
    runner.main()
