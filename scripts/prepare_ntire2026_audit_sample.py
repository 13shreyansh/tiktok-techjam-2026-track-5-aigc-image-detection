#!/usr/bin/env python3
"""Extract a deterministic, label-balanced audit sample from NTIRE shard 5.

The sample is audit-only by default.  It is selected by a filename hash before
any model inference and remains inside ignored dataset storage.  The source
does not publish generator identities, so this is an independent binary gate,
not a generator-family holdout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image


EXPECTED_ARCHIVE_BYTES = 11_370_161_676
EXPECTED_ARCHIVE_SHA256 = (
    "6d6628c983c43f1de44589151e2b3b9d33726691efbd9b0208e9f015ded9af8f"
)
REVISION = "700b6d08a3268b1e7a191306dec7321dd953b12f"
SEED = 20260830


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank(filename: str) -> str:
    return hashlib.sha256(f"{SEED}:{filename}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=Path("datasets/ntire2026/shard_5.zip")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/ntire2026/shard_5_audit_sample"),
    )
    parser.add_argument("--per-label", type=int, default=256)
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_root}")
    if args.archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise SystemExit("archive byte count mismatch")
    observed_archive_sha256 = sha256(args.archive)
    if observed_archive_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise SystemExit(f"archive SHA-256 mismatch: {observed_archive_sha256}")

    args.output_root.mkdir(parents=True)
    output_rows = []
    dimensions: Counter[str] = Counter()
    with zipfile.ZipFile(args.archive) as archive:
        label_members = [name for name in archive.namelist() if name.endswith("labels.csv")]
        if len(label_members) != 1:
            raise RuntimeError(f"expected one labels.csv, found {label_members}")
        labels_text = archive.read(label_members[0]).decode("utf-8-sig")
        labels = list(csv.DictReader(io.StringIO(labels_text)))
        grouped = {
            label: sorted(
                (row for row in labels if int(row["label"]) == label),
                key=lambda row: rank(row["image_name"]),
            )
            for label in (0, 1)
        }
        if any(len(rows) < args.per_label for rows in grouped.values()):
            raise RuntimeError("insufficient rows for requested balanced sample")
        selected = [row for label in (0, 1) for row in grouped[label][: args.per_label]]
        for row in selected:
            label = int(row["label"])
            filename = str(row["image_name"])
            member = f"shard_5/images/{filename}"
            data = archive.read(member)
            digest = hashlib.sha256(data).hexdigest()
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                decoded_format = image.format
                width, height = image.size
                mode = image.mode
            if decoded_format != "JPEG":
                raise RuntimeError(f"unexpected format for {member}: {decoded_format}")
            destination = args.output_root / "images" / str(label) / f"{digest}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            dimensions[f"{width}x{height}"] += 1
            output_rows.append(
                {
                    "path": str(destination.relative_to(args.output_root)),
                    "label": label,
                    "real_source": "NTIRE-2026-shard-5" if label == 0 else None,
                    "generator": "NTIRE-2026-shard-5-undisclosed" if label == 1 else None,
                    "family": "undisclosed",
                    "source_filename": filename,
                    "image_sha256": digest,
                    "source_archive_sha256": EXPECTED_ARCHIVE_SHA256,
                }
            )

    output_rows.sort(key=lambda row: (int(row["label"]), str(row["image_sha256"])))
    manifest = args.output_root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    canonical_source_names = "\n".join(sorted(row["source_filename"] for row in output_rows)) + "\n"
    report = {
        "purpose": "audit_only_no_training_selection_or_calibration",
        "source_revision": REVISION,
        "source_archive": str(args.archive),
        "source_archive_bytes": EXPECTED_ARCHIVE_BYTES,
        "source_archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "source_label_counts": {
            str(label): len(rows) for label, rows in grouped.items()
        },
        "selection_seed": SEED,
        "selection_policy": "lowest SHA-256 rank of seed plus randomized source filename per label",
        "selected_per_label": args.per_label,
        "selected_rows": len(output_rows),
        "selected_source_filename_inventory_sha256": hashlib.sha256(
            canonical_source_names.encode()
        ).hexdigest(),
        "manifest_sha256": sha256(manifest),
        "formats": {"JPEG": len(output_rows)},
        "dimension_counts": dict(dimensions.most_common()),
        "generator_identity_available": False,
        "limitations": [
            "Generator identities are not published for this shard.",
            "The sample can test independent binary transfer but cannot prove unseen-generator transfer.",
            "The dataset was designed for a different competition and is not evidence of the TechJam hidden-set distribution.",
        ],
    }
    (args.output_root / "audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
