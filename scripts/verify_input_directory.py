#!/usr/bin/env python3
"""Read-only validation of the organizer's image-directory input boundary."""

from __future__ import annotations

import argparse
from pathlib import Path

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("--require-images", action="store_true")
    args = parser.parse_args()

    root = args.image_directory.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"not a readable directory: {root}")
    images = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    print(f"input_directory={root}")
    print(f"supported_image_files={len(images)}")
    if images:
        print(f"first_relative_path={images[0].relative_to(root)}")
        print(f"last_relative_path={images[-1].relative_to(root)}")
    if args.require_images and not images:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
