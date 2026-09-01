#!/usr/bin/env python3
"""Entry point for the checksum-pinned v6 Kaggle comparison."""

from pathlib import Path

import kaggle_train_v3 as runner


runner.PACKAGE_NAME = "family-mixture-v6-dedup.zip"
runner.EXPECTED_ZIP_SHA256 = (
    "1655bf6350fc60e18e43ad74fafe69df7954fab2229eaf0962033b72cf8547f3"
)
runner.EXPECTED_INVENTORY_SHA256 = (
    "30d8f4791b125c208a7dcd1d2b3915098b932dd0e0127363149c64c3ce41428d"
)
runner.WORK_ROOT = Path("/kaggle/working/family-mixture-v6")
runner.OUTPUT_ROOT = Path("/kaggle/working/track5-v6-candidates")
runner.MODEL_NAMES = ("vit_pe_core_large_patch14_336",)
runner.EXCLUDED_EVAL_SHA256 = {
    # Manual review confirmed this held-out LAION photograph is a near-duplicate
    # of training SHA-256 e8c7de2e490b3830403178cbdb26fcd06a6ed58e84f39cc60b507b1b213a1c0c.
    "ff3e8968b04eaa4d95b3d9b8bd88e61a673bf0589a7be761cd71470559cd3ab1",
}
runner.EXPECTED_TRAIN_ROWS = 18958
runner.EXPECTED_EVAL_ROWS = 13350
runner.EXPECTED_CONTENT_EVAL_ROWS = 8940


if __name__ == "__main__":
    runner.main()
