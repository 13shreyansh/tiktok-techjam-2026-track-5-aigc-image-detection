#!/usr/bin/env python3
"""Acquire only the synthetic half of the pinned Apache-2.0 DiTFake release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "Jouesmak/DiTFake"
REVISION = "ca9ea06c8f926c3a11ca4b657074cc7cbb99e5c7"
GENERATORS = (
    "FLUX.1-schnell",
    "PixArt-Sigma-XL-2-1024-MS",
    "stable-diffusion-3-medium-diffusers",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("datasets/ditfake"))
    args = parser.parse_args()

    allow_patterns = [f"DiTFake/test/{name}/1_fake/*" for name in GENERATORS]
    allow_patterns.append("README.md")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=allow_patterns,
        local_dir=args.output_root,
    )

    inventory = []
    counts = {}
    for generator in GENERATORS:
        root = args.output_root / "DiTFake" / "test" / generator / "1_fake"
        files = sorted(path for path in root.iterdir() if path.is_file())
        if len(files) != 5000:
            raise SystemExit(f"{generator}: expected 5000 files, found {len(files)}")
        counts[generator] = len(files)
        inventory.extend(
            {
                "generator": generator,
                "path": str(path.relative_to(args.output_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        )

    canonical = "".join(
        f'{row["path"]}\0{row["bytes"]}\0{row["sha256"]}\n' for row in inventory
    )
    report = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "license": "Apache-2.0",
        "selection": "synthetic 1_fake directories only; all COCO 0_real files excluded",
        "counts": counts,
        "total_files": len(inventory),
        "total_bytes": sum(row["bytes"] for row in inventory),
        "canonical_inventory_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }
    (args.output_root / "acquisition-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
