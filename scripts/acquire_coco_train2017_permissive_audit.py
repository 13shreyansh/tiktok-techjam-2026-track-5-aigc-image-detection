#!/usr/bin/env python3
"""Acquire fresh, permissively licensed COCO train2017 rows for audit only.

This wrapper reuses the validated COCO downloader but excludes every image ID
listed in prior manifests.  It never reads the organizer-prohibited val2017
images and marks every output row as unavailable for training or tuning.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from collections import Counter
from pathlib import Path

try:
    import acquire_coco_train2017_permissive as coco
except ModuleNotFoundError:  # package-style import used by local tests
    from scripts import acquire_coco_train2017_permissive as coco


LICENSES = {
    4: ("Attribution License", "https://creativecommons.org/licenses/by/2.0/"),
    5: (
        "Attribution-ShareAlike License",
        "https://creativecommons.org/licenses/by-sa/2.0/",
    ),
    7: (
        "No known copyright restrictions",
        "https://www.flickr.com/commons/usage/",
    ),
    8: ("United States Government Work", "http://www.usa.gov/copyright.shtml"),
}


def read_excluded_ids(manifests: list[Path]) -> set[int]:
    excluded: set[int] = set()
    for manifest in manifests:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            if row.get("coco_image_id") is not None:
                excluded.add(int(row["coco_image_id"]))
    return excluded


def filter_fresh_rows(rows: list[dict], excluded_ids: set[int], count: int) -> list[dict]:
    fresh = [row for row in rows if int(row["id"]) not in excluded_ids]
    if len(fresh) < count:
        raise RuntimeError(f"only {len(fresh)} fresh rows remain; requested {count}")
    return fresh[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    excluded_ids = read_excluded_ids(args.exclude_manifest)
    ranked_rows, selection = coco.select_rows(
        args.annotations, args.count + len(excluded_ids), args.seed
    )
    selected = filter_fresh_rows(ranked_rows, excluded_ids, args.count)
    for row in selected:
        name, url = LICENSES[int(row["license"])]
        row["license_name"] = name
        row["license_url"] = url

    image_dir = args.output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        downloaded = list(pool.map(lambda row: coco.download_one(row, image_dir), selected))
    for row in downloaded:
        row["training_allowed"] = False
        row["workflow_purpose"] = "semantic-matching-audit"
        row["license_commercial_use_allowed"] = True
    downloaded.sort(key=lambda row: coco.rank(args.seed, int(row["coco_image_id"])))

    manifest = args.output / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in downloaded),
        encoding="utf-8",
    )
    inventory = coco.hashlib.sha256(
        "".join(
            f'{row["coco_image_id"]}:{row["sha256"]}\n' for row in downloaded
        ).encode()
    ).hexdigest()
    report = {
        "source": "COCO train2017 only",
        "source_split": "train2017",
        "purpose": "semantic-matching-audit",
        "audit_only": True,
        "training_allowed": False,
        "annotations_archive": {
            "path": str(args.annotations),
            "bytes": args.annotations.stat().st_size,
            "sha256": coco.sha256_file(args.annotations),
            "official_url": (
                "http://images.cocodataset.org/annotations/"
                "annotations_trainval2017.zip"
            ),
        },
        "seed": args.seed,
        "selected_rows": len(downloaded),
        "excluded_prior_coco_ids": len(excluded_ids),
        "allowed_license_ids": sorted(coco.ALLOWED_LICENSE_IDS),
        "selected_license_counts": dict(
            sorted(Counter(row["source_license_id"] for row in downloaded).items())
        ),
        "base_selection": selection,
        "manifest_sha256": coco.sha256_file(manifest),
        "content_inventory_sha256": inventory,
        "organizer_demo_val2017_rows": 0,
        "noncommercial_or_noderivatives_rows": 0,
    }
    (args.output / "acquisition.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
