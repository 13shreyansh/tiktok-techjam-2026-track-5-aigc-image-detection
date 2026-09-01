#!/usr/bin/env python3
"""Verify and extract a deterministic DDIM-vs-ImageNet WildFake diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import zipfile
from pathlib import Path


SOURCES = {
    "FAKE": {
        "archive": "datasets/wildfake/archives/DDIM.zip",
        "csv": "datasets/wildfake/metadata/ddim.csv",
        "archive_bytes": 6_054_264_809,
        "archive_sha256": "fa509e0ae546d91b2edd6dad91a1efe0ae3bd5c50d31609cb1db56a31d9f6e9c",
        "csv_bytes": 7_498_439,
        "csv_sha256": "6e8f3cadca7551e57f009f0f63ca5967ffd25d6639f42a5b1b081142da74cd11",
        "prefix": "./Diffusion_based/",
        "expected_label": "1",
    },
    "REAL": {
        "archive": "datasets/wildfake/archives/imagenet.zip",
        "csv": "datasets/wildfake/metadata/real_imagenet.csv",
        "archive_bytes": 1_378_959_009,
        "archive_sha256": "f80fe448c6f1dbfb9e8fd143c168d19697b1e38ee00e2d4cd040337592f515a7",
        "csv_bytes": 8_563_441,
        "csv_sha256": "31d887ea8bcf5b0aa9d6dd9a4024a98ba7283abe7f0f5c8b9464d7d63134df02",
        "prefix": "./Real/",
        "expected_label": "0",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"unexpected size for {path}")
    observed = sha256(path)
    if observed != expected_sha256:
        raise ValueError(f"unexpected SHA-256 for {path}: {observed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("datasets/wildfake_ddim_imagenet"))
    parser.add_argument("--per-class", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    if args.per_class < 1:
        raise SystemExit("--per-class must be positive")

    rng = random.Random(args.seed)
    manifest = {"seed": args.seed, "per_class": args.per_class, "classes": {}}
    for label, spec in SOURCES.items():
        archive = Path(spec["archive"])
        index = Path(spec["csv"])
        verify(archive, int(spec["archive_bytes"]), str(spec["archive_sha256"]))
        verify(index, int(spec["csv_bytes"]), str(spec["csv_sha256"]))

        with index.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows = [row for row in rows if row["IsFake"] == spec["expected_label"]]
        rng.shuffle(rows)
        selected = rows[: args.per_class]
        if len(selected) != args.per_class:
            raise ValueError(f"only {len(selected)} eligible {label} rows")

        destination = args.output_root / "test" / label
        destination.mkdir(parents=True, exist_ok=True)
        extracted = []
        with zipfile.ZipFile(archive) as source_zip:
            members = set(source_zip.namelist())
            for position, row in enumerate(selected):
                member = row["Image_path"].replace(str(spec["prefix"]), "", 1)
                if member not in members:
                    raise ValueError(f"indexed member missing from {archive}: {member}")
                suffix = Path(member).suffix.lower()
                output = destination / f"{position:06d}{suffix}"
                with source_zip.open(member) as source, output.open("wb") as target:
                    shutil.copyfileobj(source, target)
                extracted.append({"output": str(output), "source_member": member, "row": int(row["Num"])})
        manifest["classes"][label] = {
            "eligible_rows": len(rows),
            "archive": str(archive),
            "archive_sha256": spec["archive_sha256"],
            "index": str(index),
            "index_sha256": spec["csv_sha256"],
            "extracted": extracted,
        }

    manifest_path = args.output_root / "extraction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(args.output_root), "per_class": args.per_class}, indent=2))


if __name__ == "__main__":
    main()
