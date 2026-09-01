#!/usr/bin/env python3
"""Resume, verify, and optionally extract a pinned NTIRE 2026 training shard."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import zipfile
from collections import Counter
from pathlib import Path


REPO_ID = "deepfakesMSU/NTIRE-RobustAIGenDetection-train"
REVISION = "700b6d08a3268b1e7a191306dec7321dd953b12f"
SHARDS = {
    0: (20_589_999_903, "6aeafcace53555ea71bf47ec5048d51014987cde814148993836de4e68ad2755"),
    1: (20_833_978_590, "ac1deb604539941377823ac7446de4a224b6554df65884cc86b3c4220b4c7956"),
    2: (20_447_618_786, "f3d51a4845ae39d39e009305680f53fb1f60275c65dd8362ab87a85aae019050"),
    3: (20_615_488_046, "2ff92fb3867a0e9e0b104997360cf1e74e71e79188c7012a49f8ebe9a7950249"),
    4: (20_500_683_037, "d0621fc3708e844b26edf699fd205fb99c592034eef777e2250ab665c9c2fa5b"),
    5: (11_370_161_676, "6d6628c983c43f1de44589151e2b3b9d33726691efbd9b0208e9f015ded9af8f"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination_resolved = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if destination_resolved not in target.parents and target != destination_resolved:
            raise RuntimeError(f"unsafe archive member: {member.filename}")
    archive.extractall(destination)


def label_inventory(extracted: Path) -> dict:
    labels = list(extracted.rglob("labels.csv"))
    if len(labels) != 1:
        raise RuntimeError(f"expected one labels.csv, found {labels}")
    counts: Counter[int] = Counter()
    rows = 0
    with labels[0].open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            label = int(row["label"])
            if label not in (0, 1):
                raise RuntimeError(f"unexpected label: {label}")
            counts[label] += 1
            rows += 1
    images = [
        path
        for path in extracted.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    if len(images) != rows:
        raise RuntimeError(f"image/label mismatch: images={len(images)} labels={rows}")
    return {
        "labels_csv": str(labels[0]),
        "rows": rows,
        "labels": {str(label): counts[label] for label in sorted(counts)},
        "images": len(images),
        "extensions": dict(sorted(Counter(path.suffix.lower() for path in images).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, choices=sorted(SHARDS), default=5)
    parser.add_argument("--output-root", type=Path, default=Path("datasets/ntire2026"))
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()

    expected_bytes, expected_sha256 = SHARDS[args.shard]
    filename = f"shard_{args.shard}.zip"
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/{filename}?download=true"
    args.output_root.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_root / filename
    command = [
        "curl",
        "--location",
        "--fail",
        "--continue-at",
        "-",
        "--output",
        str(archive_path),
        url,
    ]
    if archive_path.is_file() and archive_path.stat().st_size == expected_bytes:
        print(
            json.dumps(
                {
                    "download_skipped": "archive already has expected byte count",
                    "archive": str(archive_path),
                }
            ),
            flush=True,
        )
    else:
        if archive_path.is_file() and archive_path.stat().st_size > expected_bytes:
            raise RuntimeError(
                f"refusing oversized partial archive: {archive_path.stat().st_size}"
            )
        print(json.dumps({"command": command}), flush=True)
        subprocess.run(command, check=True)

    observed_bytes = archive_path.stat().st_size
    observed_sha256 = sha256(archive_path)
    if observed_bytes != expected_bytes or observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"source mismatch: bytes={observed_bytes}, sha256={observed_sha256}"
        )
    report = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "source_url": url,
        "shard": args.shard,
        "archive": str(archive_path),
        "bytes": observed_bytes,
        "sha256": observed_sha256,
        "license": (
            "No explicit licence in the pinned Hugging Face card; restricted here "
            "to official challenge research/educational use with no redistribution"
        ),
        "extracted": False,
    }

    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"corrupt ZIP member: {bad_member}")
        report["zip_members"] = len(archive.infolist())
        if args.extract:
            destination = args.output_root / f"shard_{args.shard}"
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError(f"refusing to overwrite non-empty {destination}")
            destination.mkdir(parents=True, exist_ok=True)
            safe_extract(archive, destination)
            report["extracted"] = True
            report["extracted_root"] = str(destination)

    if args.extract:
        report["inventory"] = label_inventory(Path(report["extracted_root"]))
    report_path = args.output_root / f"shard_{args.shard}-acquisition.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
