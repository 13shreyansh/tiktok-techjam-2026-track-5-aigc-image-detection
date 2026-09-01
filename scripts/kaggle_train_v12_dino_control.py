#!/usr/bin/env python3
"""Run the frozen DINOv2-L representation control beside v12 PE-Core-L."""

from pathlib import Path

import kaggle_train_v12_permissive as v12


MODEL_NAME = "vit_large_patch14_dinov2.lvd142m"
OUTPUT_ROOT = Path("/kaggle/working/track5-v12-dino-control")


def configure() -> None:
    v12.runner.MODEL_NAMES = (MODEL_NAME,)
    v12.runner.OUTPUT_ROOT = OUTPUT_ROOT


if __name__ == "__main__":
    configure()
    v12.main()
