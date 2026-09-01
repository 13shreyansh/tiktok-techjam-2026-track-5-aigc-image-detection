#!/usr/bin/env python3
"""Audit RRDataset real images against the forbidden COCO validation source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def dhash(path: Path) -> int:
    with Image.open(path) as handle:
        pixels = np.asarray(
            ImageOps.exif_transpose(handle)
            .convert("L")
            .resize((9, 8), Image.Resampling.LANCZOS)
        )
    bits = (pixels[:, :-1] > pixels[:, 1:]).reshape(-1)
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rr-root",
        type=Path,
        default=Path("datasets/rrdataset/extracted/RRDataset_original_train_val"),
    )
    parser.add_argument(
        "--coco-root", type=Path, default=Path("datasets/demo_only/coco_val2017")
    )
    parser.add_argument("--max-distance", type=int, default=6)
    args = parser.parse_args()
    rr_images = sorted(args.rr_root.glob("*/real/*"))
    coco_images = sorted(path for path in args.coco_root.rglob("*") if path.is_file())
    exact = {sha256(path): path for path in coco_images}
    exact_matches = [
        {"rr": str(path), "coco": str(exact[sha256(path)])}
        for path in rr_images
        if sha256(path) in exact
    ]

    coco_hashes = np.array([dhash(path) for path in coco_images], dtype=np.uint64)
    near_matches = []
    for rr_path in rr_images:
        value = np.uint64(dhash(rr_path))
        xor = np.bitwise_xor(coco_hashes, value).view(np.uint8).reshape(-1, 8)
        distances = POPCOUNT[xor].sum(axis=1)
        index = int(distances.argmin())
        distance = int(distances[index])
        if distance <= args.max_distance:
            near_matches.append(
                {"distance": distance, "rr": str(rr_path), "coco": str(coco_images[index])}
            )
    payload = {
        "rr_real_images": len(rr_images),
        "coco_demo_source_images": len(coco_images),
        "exact_sha256_matches": exact_matches,
        "dhash_max_distance": args.max_distance,
        "near_matches": near_matches,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
