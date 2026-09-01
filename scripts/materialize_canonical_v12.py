#!/usr/bin/env python3
"""Materialize label-independent square JPEG views for the v12 GPU package.

The source manifests remain the provenance authority.  This derivative removes
container and geometry shortcuts before upload by applying exactly the same
decode, EXIF transpose, centre crop, resize, and JPEG encoding to both labels.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps


SIZE = 336
JPEG_QUALITY = 96
JPEG_SUBSAMPLING = 0
POLICY = "exif_transpose_center_square_resize336_jpeg_q96_subsampling0"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(path: Path) -> tuple[bytes, tuple[int, int]]:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        original_size = image.size
        side = min(image.size)
        left = (image.width - side) // 2
        top = (image.height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        image = image.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=JPEG_QUALITY,
            subsampling=JPEG_SUBSAMPLING,
            optimize=False,
            progressive=False,
        )
    return buffer.getvalue(), original_size


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_rows(path: Path, rows: list[dict]) -> str:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return file_sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    manifests = {"train": args.train.resolve(), "eval_frozen": args.eval.resolve()}
    args.output.mkdir(parents=True)
    image_root = args.output / "images"
    image_root.mkdir()
    all_seen: dict[str, tuple[str, int]] = {}
    rewritten: dict[str, list[dict]] = {}
    source_rows = sum(len(read_rows(path)) for path in manifests.values())
    processed = 0
    source_bytes = 0
    derivative_bytes = 0

    for role, manifest in manifests.items():
        output_rows = []
        for row in read_rows(manifest):
            source = Path(row["path"])
            if not source.is_absolute():
                source = (manifest.parent / source).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            if row.get("organizer_demo_row") is not False:
                raise ValueError(f"{role}: organizer demo status is not false: {source}")
            source_digest = file_sha256(source)
            recorded_digest = row.get("sha256") or row.get("image_sha256")
            if recorded_digest and recorded_digest != source_digest:
                raise ValueError(f"{role}: source checksum mismatch: {source}")
            data, original_size = canonical_bytes(source)
            digest = hashlib.sha256(data).hexdigest()
            previous = all_seen.get(digest)
            label = int(row["label"])
            if previous and previous[1] != label:
                raise ValueError(
                    f"canonical cross-label collision {digest}: {previous} vs {(role, label)}"
                )
            all_seen[digest] = (role, label)
            destination = image_root / digest[:2] / f"{digest}.jpg"
            destination.parent.mkdir(exist_ok=True)
            if not destination.exists():
                destination.write_bytes(data)
                derivative_bytes += len(data)
            source_bytes += source.stat().st_size
            output_rows.append(
                {
                    **row,
                    "path": str(destination.resolve()),
                    "source_image_sha256": source_digest,
                    "sha256": digest,
                    "canonicalization": POLICY,
                    "canonical_width": SIZE,
                    "canonical_height": SIZE,
                    "canonical_format": "JPEG",
                    "original_width": original_size[0],
                    "original_height": original_size[1],
                }
            )
            processed += 1
            if processed % 500 == 0 or processed == source_rows:
                print(f"canonicalized {processed}/{source_rows}", flush=True)
        rewritten[role] = output_rows

    train_hashes = {row["sha256"] for row in rewritten["train"]}
    eval_hashes = {row["sha256"] for row in rewritten["eval_frozen"]}
    overlap = train_hashes & eval_hashes
    if overlap:
        raise ValueError(f"canonical train/eval overlap: {len(overlap)}")

    manifest_hashes = {
        role: write_rows(args.output / f"{role}.jsonl", rows)
        for role, rows in rewritten.items()
    }
    inventory = hashlib.sha256(
        "".join(sorted(all_seen)).encode()
    ).hexdigest()
    report = {
        "policy": POLICY,
        "size": SIZE,
        "jpeg_quality": JPEG_QUALITY,
        "jpeg_subsampling": JPEG_SUBSAMPLING,
        "source_rows": source_rows,
        "unique_derivative_images": len(all_seen),
        "source_bytes_counting_manifest_rows": source_bytes,
        "derivative_bytes": derivative_bytes,
        "manifest_sha256": manifest_hashes,
        "derivative_inventory_sha256": inventory,
        "train_eval_overlap": 0,
        "label_counts": {
            role: dict(sorted(Counter(int(row["label"]) for row in rows).items()))
            for role, rows in rewritten.items()
        },
        "organizer_demo_rows": 0,
        "interpretation": (
            "This removes label-correlated source containers and original aspect ratios "
            "from the model input; it does not prove hidden-set generalization."
        ),
    }
    (args.output / "canonicalization.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
