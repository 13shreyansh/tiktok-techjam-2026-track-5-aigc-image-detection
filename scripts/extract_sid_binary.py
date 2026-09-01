#!/usr/bin/env python3
"""Extract only Track-eligible real/full-synthetic SID_Set rows from pinned Parquet shards."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

PINNED_SHARDS = {
    "train-00000-of-00249.parquet": {
        "bytes": 489_780_970,
        "sha256": "82e62f400fbb168e0b69ba5104e8109c312fe8a02ee07f06a82ab58208a6fb4a",
    },
    "validation-00000-of-00034.parquet": {
        "bytes": 477_663_216,
        "sha256": "56cf2dd5c6a72a158f91aee4c5e06154f5d0a0903eb258a3de11eedded82c2a6",
    },
    "validation-00001-of-00034.parquet": {
        "bytes": 505_844_042,
        "sha256": "1447bbd98adf7eda68fca5615560c6b1de34c8e30157ff6b34ebd1e015a18042",
    },
}
LABEL_DIR = {0: "REAL", 1: "FAKE"}
FORMAT_SUFFIX = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp", "TIFF": ".tiff"}


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def verify_shard(path: Path) -> None:
    expected = PINNED_SHARDS.get(path.name)
    if expected is None:
        raise SystemExit(f"shard is not pinned in this script: {path.name}")
    observed_size = path.stat().st_size
    observed_sha256 = digest(path)
    if observed_size != expected["bytes"] or observed_sha256 != expected["sha256"]:
        raise SystemExit(
            f"verification failed for {path}: bytes={observed_size}, sha256={observed_sha256}"
        )


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "unnamed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-split", choices=("train", "test"), required=True)
    args = parser.parse_args()
    verify_shard(args.shard)

    table = pq.read_table(args.shard, columns=["img_id", "image", "width", "height", "label"])
    counts: Counter[int] = Counter()
    records: list[dict] = []
    rows = table.to_pylist()
    for row_index, row in enumerate(rows):
        label = int(row["label"])
        counts[label] += 1
        if label not in LABEL_DIR:
            continue
        image_bytes = row["image"]["bytes"]
        if not image_bytes:
            raise SystemExit(f"missing embedded image bytes at row {row_index}")
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
            image_format = image.format
        suffix = FORMAT_SUFFIX.get(image_format, ".img")
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        destination = (
            args.output_root
            / args.output_split
            / LABEL_DIR[label]
            / f"{safe_stem(str(row['img_id']))}-{image_sha256[:12]}{suffix}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if digest(destination) != image_sha256:
                raise SystemExit(f"existing file has different content: {destination}")
        else:
            destination.write_bytes(image_bytes)
        records.append(
            {
                "path": str(destination.relative_to(args.output_root)),
                "label": label,
                "img_id": row["img_id"],
                "width": int(row["width"]),
                "height": int(row["height"]),
                "image_sha256": image_sha256,
                "source_shard": args.shard.name,
                "source_row": row_index,
            }
        )

    manifest = args.output_root / f"manifest-{args.output_split}-{args.shard.stem}.jsonl"
    manifest.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    print(
        json.dumps(
            {
                "source_rows": len(rows),
                "source_label_counts": dict(sorted(counts.items())),
                "eligible_extracted": len(records),
                "excluded_tampered": counts[2],
                "manifest": str(manifest),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
