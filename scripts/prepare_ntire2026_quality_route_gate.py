#!/usr/bin/env python3
"""Freeze a byte-disjoint one-shot NTIRE gate for the v11 quality router.

The source archive is already checksum pinned.  Rows from the earlier NTIRE
audit are excluded by source filename before selection.  The extracted files
remain ignored and evaluation-only because the pinned dataset card does not
state a reusable licence.
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
SEED = 20260831
PER_LABEL = 512


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank(filename: str) -> str:
    return hashlib.sha256(f"{SEED}:quality-route-v11:{filename}".encode()).hexdigest()


def read_excluded_source_names(path: Path) -> set[str]:
    return {
        str(json.loads(line)["source_filename"])
        for line in path.read_text().splitlines()
        if line
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=Path("datasets/ntire2026/shard_5.zip")
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        default=Path("datasets/ntire2026/shard_5_audit_sample/manifest.jsonl"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/ntire2026/shard_5_quality_route_gate"),
    )
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_root}")
    if args.archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise SystemExit("archive byte count mismatch")
    observed_archive_sha256 = sha256_file(args.archive)
    if observed_archive_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise SystemExit(f"archive SHA-256 mismatch: {observed_archive_sha256}")

    excluded = read_excluded_source_names(args.exclude_manifest)
    args.output_root.mkdir(parents=True)
    rows = []
    dimensions: Counter[str] = Counter()
    with zipfile.ZipFile(args.archive) as archive:
        label_members = [name for name in archive.namelist() if name.endswith("labels.csv")]
        if len(label_members) != 1:
            raise RuntimeError(f"expected one labels.csv, found {label_members}")
        labels = list(
            csv.DictReader(io.StringIO(archive.read(label_members[0]).decode("utf-8-sig")))
        )
        grouped = {
            label: sorted(
                (
                    row
                    for row in labels
                    if int(row["label"]) == label
                    and str(row["image_name"]) not in excluded
                ),
                key=lambda row: rank(str(row["image_name"])),
            )
            for label in (0, 1)
        }
        if any(len(group) < PER_LABEL for group in grouped.values()):
            raise RuntimeError("insufficient rows for the frozen balanced sample")
        selected = [row for label in (0, 1) for row in grouped[label][:PER_LABEL]]
        if any(str(row["image_name"]) in excluded for row in selected):
            raise RuntimeError("excluded source filename entered the fresh gate")

        for index, row in enumerate(selected, 1):
            label = int(row["label"])
            filename = str(row["image_name"])
            data = archive.read(f"shard_5/images/{filename}")
            digest = hashlib.sha256(data).hexdigest()
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                decoded_format = image.format
                width, height = image.size
                mode = image.mode
            if decoded_format != "JPEG":
                raise RuntimeError(f"unexpected format: {filename}: {decoded_format}")
            destination = args.output_root / "images" / str(label) / f"{digest}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            dimensions[f"{width}x{height}"] += 1
            rows.append(
                {
                    "path": str(destination.relative_to(args.output_root)),
                    "label": label,
                    "real_source": "NTIRE-2026-shard-5-v11-fresh" if label == 0 else None,
                    "generator": "NTIRE-2026-shard-5-v11-undisclosed" if label == 1 else None,
                    "family": "undisclosed",
                    "source_filename": filename,
                    "image_sha256": digest,
                    "source_archive_sha256": EXPECTED_ARCHIVE_SHA256,
                    "workflow_purpose": "one-shot-v11-promotion-gate",
                }
            )
            if index % 128 == 0 or index == len(selected):
                print(json.dumps({"extracted_rows": index}), flush=True)

    if len({row["image_sha256"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate image bytes in fresh gate")
    previous_hashes = {
        json.loads(line)["image_sha256"]
        for line in args.exclude_manifest.read_text().splitlines()
        if line
    }
    overlap = previous_hashes & {row["image_sha256"] for row in rows}
    if overlap:
        raise RuntimeError(f"fresh gate overlaps old NTIRE bytes: {len(overlap)}")

    rows.sort(key=lambda row: (int(row["label"]), str(row["image_sha256"])))
    manifest = args.output_root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    source_inventory = "\n".join(sorted(row["source_filename"] for row in rows)) + "\n"
    report = {
        "purpose": "one-shot-v11-promotion-evaluation-only",
        "training_tuning_calibration_allowed": False,
        "source_revision": REVISION,
        "source_archive_bytes": EXPECTED_ARCHIVE_BYTES,
        "source_archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "source_license_boundary": "no explicit reusable licence in pinned card; do not redistribute",
        "selection_seed": SEED,
        "selection_policy": "lowest SHA-256 rank of seed, v11 namespace and source filename per label after old-audit exclusion",
        "excluded_source_filenames": len(excluded),
        "selected_per_label": PER_LABEL,
        "selected_rows": len(rows),
        "old_source_filename_overlap": 0,
        "old_image_sha256_overlap": 0,
        "selected_source_filename_inventory_sha256": hashlib.sha256(
            source_inventory.encode()
        ).hexdigest(),
        "manifest_sha256": sha256_file(manifest),
        "formats": {"JPEG": len(rows)},
        "dimension_counts": dict(dimensions.most_common()),
        "generator_identity_available": False,
        "organizer_demo_rows": 0,
        "limitations": [
            "Generator identities are unpublished.",
            "The source is new rows from the same NTIRE shard as an earlier audit, not a new dataset family.",
            "This different-competition corpus does not estimate the TechJam hidden distribution.",
        ],
    }
    (args.output_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

