#!/usr/bin/env python3
"""Acquire a deterministic audit-only Open Images V7 validation pool.

The CVDF S3 mirror is public and exposes object checksums and sizes. Open
Images lists its images as CC BY 2.0 while warning users to verify each source
licence themselves. This script records that limitation and marks every row as
audit-only; it never upgrades the dataset-wide statement into a per-image
licence warranty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


SEED = 20260901
BUCKET = "https://open-images-dataset.s3.amazonaws.com"
PREFIX = "validation/"
OFFICIAL_DOWNLOAD = "https://storage.googleapis.com/openimages/web/download_v7.html"
OFFICIAL_DESCRIPTION = "https://storage.googleapis.com/openimages/web/factsfigures_v7.html"
LICENSE_URL = "https://creativecommons.org/licenses/by/2.0/"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_key(key: str, seed: int = SEED) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def parse_listing(payload: bytes) -> tuple[list[dict], str | None]:
    root = ET.fromstring(payload)
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    rows = []
    for item in root.findall("s3:Contents", namespace):
        key = item.findtext("s3:Key", namespaces=namespace)
        size = item.findtext("s3:Size", namespaces=namespace)
        etag = item.findtext("s3:ETag", namespaces=namespace)
        modified = item.findtext("s3:LastModified", namespaces=namespace)
        if key and key.lower().endswith(".jpg"):
            rows.append(
                {
                    "key": key,
                    "bytes": int(size or 0),
                    "etag": (etag or "").strip('"'),
                    "last_modified": modified,
                }
            )
    token = root.findtext("s3:NextContinuationToken", namespaces=namespace)
    return rows, token


def list_objects() -> tuple[list[dict], int]:
    objects: list[dict] = []
    token = None
    pages = 0
    while True:
        query = {"list-type": "2", "prefix": PREFIX, "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        url = f"{BUCKET}/?{urllib.parse.urlencode(query)}"
        with urllib.request.urlopen(url, timeout=30) as response:
            rows, token = parse_listing(response.read())
        objects.extend(rows)
        pages += 1
        if not token:
            break
    return objects, pages


def download_one(record: dict, image_root: Path) -> dict:
    image_id = Path(record["key"]).stem
    destination = image_root / f"{image_id}.jpg"
    if not destination.is_file() or destination.stat().st_size != record["bytes"]:
        url = f"{BUCKET}/{record['key']}"
        request = urllib.request.Request(url, headers={"User-Agent": "track5-audit/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        if len(data) != record["bytes"]:
            raise RuntimeError(f"size mismatch for {record['key']}")
        destination.write_bytes(data)
    digest = file_sha256(destination)
    with Image.open(destination) as image:
        image.verify()
    with Image.open(destination) as image:
        width, height = image.size
        mode = image.mode
    return {
        "path": str(destination.resolve()),
        "label": 0,
        "dataset": "Open-Images-V7-validation-CVDF",
        "real_source": "Open-Images-V7-validation-CVDF",
        "image_id": image_id,
        "source_url": f"{BUCKET}/{record['key']}",
        "source_key": record["key"],
        "source_etag": record["etag"],
        "source_last_modified": record["last_modified"],
        "source_bytes": record["bytes"],
        "source_image_sha256": digest,
        "sha256": digest,
        "width": width,
        "height": height,
        "mode": mode,
        "license_name": "CC BY 2.0 (dataset listing; not independently reverified per image)",
        "license_url": LICENSE_URL,
        "organizer_demo_row": False,
        "training_allowed": False,
        "workflow_purpose": "independent-real-source-audit",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.count < 144 or args.workers < 1:
        raise SystemExit("count must be at least 144 and workers must be positive")

    objects, pages = list_objects()
    if len(objects) < args.count:
        raise RuntimeError(f"only {len(objects)} validation images listed")
    selected = sorted(objects, key=lambda row: rank_key(row["key"], args.seed))[
        : args.count
    ]
    args.output.mkdir(parents=True)
    image_root = args.output / "images"
    image_root.mkdir()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda row: download_one(row, image_root), selected))
    rows.sort(key=lambda row: row["image_id"])
    if len({row["source_image_sha256"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate Open Images bytes in selected pool")
    manifest = args.output / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    key_inventory = hashlib.sha256(
        "\n".join(sorted(row["source_key"] for row in rows)).encode()
    ).hexdigest()
    content_inventory = hashlib.sha256(
        "\n".join(sorted(row["source_image_sha256"] for row in rows)).encode()
    ).hexdigest()
    report = {
        "status": "acquired_audit_only",
        "official_download_url": OFFICIAL_DOWNLOAD,
        "official_description_url": OFFICIAL_DESCRIPTION,
        "bucket": BUCKET,
        "prefix": PREFIX,
        "listing_pages": pages,
        "listed_images": len(objects),
        "selection": "lowest SHA-256 rank of seed:key",
        "seed": args.seed,
        "selected_rows": len(rows),
        "unique_content": len(rows),
        "source_bytes": sum(int(row["source_bytes"]) for row in rows),
        "key_inventory_sha256": key_inventory,
        "content_inventory_sha256": content_inventory,
        "manifest_sha256": file_sha256(manifest),
        "license_name": "CC BY 2.0",
        "license_url": LICENSE_URL,
        "license_boundary": "Open Images lists images under CC BY 2.0 but disclaims a per-image warranty; individual upstream landing-page verification was not performed.",
        "organizer_demo_rows": 0,
        "training_allowed_rows": 0,
    }
    (args.output / "acquisition.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
