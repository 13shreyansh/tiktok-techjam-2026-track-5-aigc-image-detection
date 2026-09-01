#!/usr/bin/env python3
"""Entry point for the checksum-pinned v5 Kaggle comparison."""

from pathlib import Path

import kaggle_train_v3 as runner


runner.PACKAGE_NAME = "family-mixture-v5-dedup.zip"
runner.EXPECTED_ZIP_SHA256 = (
    "31494bf4d8d345a26d838b35012ab1cfce827a7a892c8e5844880effaf0a6ae4"
)
runner.EXPECTED_INVENTORY_SHA256 = (
    "9b9a2a6c21d40d9bf539bc3a0440d402526c3698f8f6faaac5b62901a2bd76b8"
)
runner.WORK_ROOT = Path("/kaggle/working/family-mixture-v5")
runner.OUTPUT_ROOT = Path("/kaggle/working/track5-v5-candidates")
runner.EXPECTED_TRAIN_ROWS = 16910
runner.EXPECTED_EVAL_ROWS = 13350
runner.EXPECTED_CONTENT_EVAL_ROWS = 8940


if __name__ == "__main__":
    runner.main()
