#!/usr/bin/env python3
"""Freeze a hash-selected CIFAR-100 test gate from the pinned Parquet mirror."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image


EXPECTED_SOURCE_BYTES = 23_772_751
EXPECTED_SOURCE_SHA256 = "98776c529bb146a9c791229df74a5cf076be9b43d82dbbd334b6a7788d73dc68"
PINNED_REVISION = "aadb3af77e9048adbea6b47c21a81e47dd092ae5"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_by_hash(
    rows: list[dict[str, Any]], per_class: int, rank_start: int = 0
) -> list[dict[str, Any]]:
    if rank_start < 0:
        raise ValueError("rank_start must be non-negative")
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(int(row["fine_label"]), []).append(row)
    if len(groups) != 100:
        raise ValueError(f"expected 100 fine classes, found {len(groups)}")
    selected = []
    for fine_label in sorted(groups):
        ordered = sorted(groups[fine_label], key=lambda row: sha256_bytes(row["img"]["bytes"]))
        rank_end = rank_start + per_class
        if len(ordered) < rank_end:
            raise ValueError(
                f"fine class {fine_label} has only {len(ordered)} rows; "
                f"need hash ranks [{rank_start}, {rank_end})"
            )
        selected.extend(ordered[rank_start:rank_end])
    return selected


def prepare(
    source: Path, output_dir: Path, per_class: int, rank_start: int = 0
) -> dict[str, Any]:
    if source.stat().st_size != EXPECTED_SOURCE_BYTES:
        raise ValueError("source byte count does not match pinned mirror")
    source_hash = sha256_file(source)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError("source SHA-256 does not match pinned mirror")

    rows = pq.read_table(source).to_pylist()
    if len(rows) != 10_000:
        raise ValueError(f"expected 10,000 test rows, found {len(rows)}")
    selected = select_by_hash(rows, per_class, rank_start)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    hashes = set()
    for row in selected:
        payload = row["img"]["bytes"]
        image_hash = sha256_bytes(payload)
        if image_hash in hashes:
            raise ValueError(f"duplicate selected image hash: {image_hash}")
        hashes.add(image_hash)
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            if image.size != (32, 32) or image.convert("RGB").mode != "RGB":
                raise ValueError("unexpected CIFAR-100 image geometry or mode")
        fine = int(row["fine_label"])
        coarse = int(row["coarse_label"])
        relative_image = Path("images") / f"{fine:03d}" / f"{image_hash}.png"
        destination = output_dir / relative_image
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        manifest_rows.append(
            {
                "path": str(relative_image),
                "label": 0,
                "family": "authentic-low-resolution-independent",
                "real_source": "CIFAR100-test",
                "fine_label": fine,
                "coarse_label": coarse,
                "image_sha256": image_hash,
            }
        )

    manifest_rows.sort(key=lambda row: (row["fine_label"], row["image_sha256"]))
    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows))
    inventory_hash = hashlib.sha256(
        "\n".join(row["image_sha256"] for row in manifest_rows).encode()
    ).hexdigest()
    result = {
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "source_sha256": source_hash,
        "mirror_revision": PINNED_REVISION,
        "source_rows": len(rows),
        "selection": "contiguous image SHA-256 rank window within each fine class",
        "hash_rank_start_inclusive": rank_start,
        "hash_rank_end_exclusive": rank_start + per_class,
        "per_fine_class": per_class,
        "selected_rows": len(manifest_rows),
        "fine_class_counts": dict(sorted(Counter(row["fine_label"] for row in manifest_rows).items())),
        "coarse_class_counts": dict(sorted(Counter(row["coarse_label"] for row in manifest_rows).items())),
        "selected_inventory_sha256": inventory_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "training_allowed": False,
        "boundary": "Evaluation-only authentic source; no training, tuning, thresholding or redistribution.",
    }
    (output_dir / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--rank-start", type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.source, args.output_dir, args.per_class, args.rank_start),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
