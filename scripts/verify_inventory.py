#!/usr/bin/env python3
"""Verify the acquired preparation inventory without extracting datasets."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def image_count(archive: Path) -> int:
    with ZipFile(archive) as zipped:
        return sum(
            name.lower().endswith((".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"))
            for name in zipped.namelist()
        )


def main() -> int:
    json.loads((ROOT / "resources/resource_manifest.json").read_text(encoding="utf-8"))
    for resource_id in (
        "cifake_archive",
        "coco_val2017_archive",
        "coco_trainval2017_annotations",
        "wildfake_dalle3_index",
    ):
        run(sys.executable, "scripts/acquire_resources.py", "verify", resource_id)

    cifake = ROOT / "datasets/source_archives/cifake-2023-03-28.zip"
    coco = ROOT / "datasets/source_archives/val2017.zip"
    annotations = ROOT / "datasets/source_archives/annotations_trainval2017.zip"
    dalle_index = ROOT / "datasets/metadata/wildfake/dalle3.csv"
    if image_count(cifake) != 120_000:
        raise SystemExit("CIFAKE image count mismatch")
    if image_count(coco) != 5_000:
        raise SystemExit("COCO source image count mismatch")

    with ZipFile(annotations) as zipped:
        val = json.loads(zipped.read("annotations/instances_val2017.json"))
    if len(val["images"]) != 5_000 or not all("license" in item for item in val["images"]):
        raise SystemExit("COCO validation licence metadata mismatch")

    with dalle_index.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        "Generator": "Diffusion_based",
        "Architecture": "DALLE",
        "Category": "DALLE",
        "IsAdvanced": "1",
        "IsFake": "1",
    }
    if len(rows) != 8_843 or any(
        row.get(key) != value for row in rows for key, value in expected.items()
    ):
        raise SystemExit("DALL-E Advanced index mismatch")

    print("CIFAKE images=120000")
    print("COCO source images=5000; licence records present for all source images")
    print("DALL-E Advanced index rows=8843; all required filters match")
    print("preparation inventory verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
