#!/usr/bin/env python3
"""Acquire a deterministic commercial-compatible COCO train2017 real subset.

The organizer-prohibited val2017 split is rejected by member name, image ID,
file name and URL. Only licence IDs 4, 5, 7 and 8 are admitted; all
NonCommercial and NoDerivatives rows are excluded conservatively.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import heapq
import io
import json
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image


TRAIN_MEMBER = "annotations/instances_train2017.json"
VAL_MEMBER = "annotations/instances_val2017.json"
ALLOWED_LICENSE_IDS = {4, 5, 7, 8}
SEED = 20260831


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_named_array(stream: io.TextIOBase, key: str):
    """Incrementally decode objects from one top-level JSON array."""
    marker = f'"{key}"'
    prefix = ""
    while True:
        chunk = stream.read(64 * 1024)
        if not chunk:
            raise ValueError(f"top-level array not found: {key}")
        prefix += chunk
        marker_at = prefix.find(marker)
        if marker_at < 0:
            prefix = prefix[-len(marker) :]
            continue
        array_at = prefix.find("[", marker_at + len(marker))
        if array_at < 0:
            continue
        remainder = prefix[array_at + 1 :]
        break

    current: list[str] = []
    depth = 0
    in_string = False
    escaped = False
    chunks = iter((remainder,))
    while True:
        try:
            chunk = next(chunks)
        except StopIteration:
            chunk = stream.read(64 * 1024)
            if not chunk:
                raise ValueError(f"unterminated top-level array: {key}")
        for character in chunk:
            if depth == 0:
                if character in " \t\r\n,":
                    continue
                if character == "]":
                    return
                if character != "{":
                    raise ValueError(
                        f"expected object in {key}, found {character!r}"
                    )
                current = [character]
                depth = 1
                in_string = False
                escaped = False
                continue

            current.append(character)
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    value = json.loads("".join(current))
                    if not isinstance(value, dict):
                        raise ValueError(
                            f"expected object in {key}, got {type(value).__name__}"
                        )
                    yield value
                    current = []


def image_rows(archive: Path, member: str):
    if "val2017" in member and member != VAL_MEMBER:
        raise ValueError(f"unexpected validation member: {member}")
    with zipfile.ZipFile(archive) as bundle:
        with bundle.open(member) as raw:
            with io.TextIOWrapper(raw, encoding="utf-8") as text:
                yield from iter_named_array(text, "images")


def rank(seed: int, image_id: int) -> int:
    material = f"{seed}:coco-train2017-permissive:{image_id}".encode()
    return int.from_bytes(hashlib.sha256(material).digest(), "big")


def validate_train_row(row: dict, validation_ids: set[int]) -> None:
    image_id = int(row["id"])
    file_name = str(row["file_name"])
    url = str(row["coco_url"])
    if image_id in validation_ids:
        raise ValueError(f"train/validation image ID overlap: {image_id}")
    if "val2017" in file_name.lower() or "val2017" in url.lower():
        raise ValueError(f"organizer-prohibited val2017 locator: {image_id}")
    if "train2017" not in url.lower() or not file_name.lower().endswith(".jpg"):
        raise ValueError(f"unexpected train2017 locator: {image_id}")
    if int(row["license"]) not in ALLOWED_LICENSE_IDS:
        raise ValueError(f"non-permissive licence reached selected rows: {image_id}")


def select_rows(archive: Path, count: int, seed: int) -> tuple[list[dict], dict]:
    validation_ids = {int(row["id"]) for row in image_rows(archive, VAL_MEMBER)}
    heap: list[tuple[int, int, dict]] = []
    scanned = 0
    excluded = Counter()
    for row in image_rows(archive, TRAIN_MEMBER):
        scanned += 1
        licence_id = int(row["license"])
        if licence_id not in ALLOWED_LICENSE_IDS:
            excluded[licence_id] += 1
            continue
        validate_train_row(row, validation_ids)
        score = rank(seed, int(row["id"]))
        item = (-score, int(row["id"]), row)
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    if len(heap) != count:
        raise RuntimeError(f"requested {count} rows, selected {len(heap)}")
    selected = [item[2] for item in heap]
    selected.sort(key=lambda row: rank(seed, int(row["id"])))
    return selected, {
        "train_rows_scanned": scanned,
        "validation_ids_rejected_by_identity": len(validation_ids),
        "excluded_license_counts": dict(sorted(excluded.items())),
    }


def download_one(row: dict, image_dir: Path, retries: int = 3) -> dict:
    target = image_dir / str(row["file_name"])
    # The pinned annotation intentionally supplies the official HTTP endpoint.
    # Its HTTPS alias presented a hostname-mismatched certificate during the
    # observed run; integrity is established from decoded geometry plus the
    # per-file SHA-256 inventory rather than transport assumptions.
    url = str(row["coco_url"])
    error = None
    for attempt in range(retries):
        try:
            if not target.exists():
                request = urllib.request.Request(
                    url, headers={"User-Agent": "track5-public-data-audit/1.0"}
                )
                with urllib.request.urlopen(request, timeout=45) as response:
                    payload = response.read()
                target.write_bytes(payload)
            with Image.open(target) as image:
                image.load()
                observed_size = [int(image.width), int(image.height)]
                observed_mode = image.mode
            expected_size = [int(row["width"]), int(row["height"])]
            if observed_size != expected_size:
                raise ValueError(
                    f"geometry mismatch for {target.name}: {observed_size} != {expected_size}"
                )
            return {
                "path": str(target.resolve()),
                "label": 0,
                "real_source": "COCO-train2017-commercial-compatible",
                "generator": None,
                "coco_image_id": int(row["id"]),
                "source_url": url,
                "source_license_id": int(row["license"]),
                "source_license_url": row.get("license_url"),
                "source_license_name": row.get("license_name"),
                "width": observed_size[0],
                "height": observed_size[1],
                "mode": observed_mode,
                "sha256": sha256_file(target),
                "training_allowed": True,
                "organizer_demo_row": False,
            }
        except Exception as exc:  # acquisition reports exact terminal error
            error = exc
            if target.exists():
                target.unlink()
            if attempt + 1 < retries:
                time.sleep(1 + attempt)
    raise RuntimeError(f"failed to acquire {url}: {error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.count <= 0:
        raise SystemExit("count must be positive")
    if args.workers <= 0:
        raise SystemExit("workers must be positive")

    selected, selection_report = select_rows(args.annotations, args.count, args.seed)
    licences = {
        int(item["id"]): {"name": item["name"], "url": item["url"]}
        for item in [
            {"id": 4, "name": "Attribution License", "url": "https://creativecommons.org/licenses/by/2.0/"},
            {"id": 5, "name": "Attribution-ShareAlike License", "url": "https://creativecommons.org/licenses/by-sa/2.0/"},
            {"id": 7, "name": "No known copyright restrictions", "url": "https://www.flickr.com/commons/usage/"},
            {"id": 8, "name": "United States Government Work", "url": "http://www.usa.gov/copyright.shtml"},
        ]
    }
    for row in selected:
        licence = licences[int(row["license"])]
        row["license_name"] = licence["name"]
        row["license_url"] = licence["url"]

    image_dir = args.output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda row: download_one(row, image_dir), selected))
    rows.sort(key=lambda row: rank(args.seed, int(row["coco_image_id"])))

    manifest = args.output / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    inventory = hashlib.sha256(
        "".join(f'{row["coco_image_id"]}:{row["sha256"]}\n' for row in rows).encode()
    ).hexdigest()
    report = {
        "source": "COCO train2017 only",
        "annotations_archive": {
            "path": str(args.annotations),
            "bytes": args.annotations.stat().st_size,
            "sha256": sha256_file(args.annotations),
            "official_url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        },
        "seed": args.seed,
        "selected_rows": len(rows),
        "allowed_license_ids": sorted(ALLOWED_LICENSE_IDS),
        "selected_license_counts": dict(sorted(Counter(row["source_license_id"] for row in rows).items())),
        "selection": selection_report,
        "manifest_sha256": sha256_file(manifest),
        "content_inventory_sha256": inventory,
        "organizer_demo_val2017_rows": 0,
        "noncommercial_or_noderivatives_rows": 0,
        "training_allowed": True,
    }
    (args.output / "acquisition.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
