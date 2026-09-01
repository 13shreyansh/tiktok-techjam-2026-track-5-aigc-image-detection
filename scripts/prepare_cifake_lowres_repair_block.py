#!/usr/bin/env python3
"""Freeze a matched, training-only CIFAKE block for the low-resolution repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def resolve_row_path(manifest: Path, row: dict) -> Path:
    value = Path(row["path"])
    return (value if value.is_absolute() else manifest.parent / value).resolve()


def read_exclusions(manifests: list[Path]) -> tuple[set[Path], set[str]]:
    excluded_paths: set[Path] = set()
    excluded_hashes: set[str] = set()
    for manifest in manifests:
        for row in read_rows(manifest):
            path = resolve_row_path(manifest, row)
            if not path.is_file():
                raise RuntimeError(f"missing excluded source image: {path}")
            excluded_paths.add(path)
            observed_hash = sha256_file(path)
            declared_hash = row.get("image_sha256")
            if declared_hash and str(declared_hash) != observed_hash:
                raise RuntimeError(
                    f"excluded source hash mismatch: {path}: "
                    f"declared={declared_hash}, observed={observed_hash}"
                )
            excluded_hashes.add(observed_hash)
    return excluded_paths, excluded_hashes


def select_candidates(
    paths: list[Path], excluded_paths: set[Path], excluded_hashes: set[str], count: int
) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    seen_hashes: set[str] = set()
    for path in sorted(paths):
        resolved = path.resolve()
        if resolved in excluded_paths:
            continue
        digest = sha256_file(resolved)
        if digest in excluded_hashes or digest in seen_hashes:
            continue
        with Image.open(resolved) as image:
            image.load()
            if image.size != (32, 32) or image.mode != "RGB":
                raise RuntimeError(f"unexpected CIFAKE image contract: {resolved}")
        seen_hashes.add(digest)
        candidates.append((digest, resolved))
    candidates.sort()
    if len(candidates) < count:
        raise RuntimeError(f"only {len(candidates)} eligible images for requested {count}")
    return candidates[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cifake-root", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=5000)
    args = parser.parse_args()
    if args.per_label < 1:
        raise SystemExit("--per-label must be positive")

    excluded_paths, excluded_hashes = read_exclusions(args.exclude_manifest)

    rows: list[dict] = []
    inventories = {}
    contracts = (
        (0, "REAL", "CIFAKE-CIFAR10-v10-supplement", None, "authentic-low-resolution"),
        (1, "FAKE", None, "CIFAKE-Stable-Diffusion-v10-supplement", "latent-diffusion"),
    )
    output_dir = args.output.parent.resolve()
    for label, directory, real_source, generator, family in contracts:
        source_paths = list((args.cifake_root / "train" / directory).glob("*.jpg"))
        selected = select_candidates(
            source_paths, excluded_paths, excluded_hashes, args.per_label
        )
        inventories[directory] = hashlib.sha256(
            "\n".join(digest for digest, _ in selected).encode()
        ).hexdigest()
        for digest, path in selected:
            rows.append(
                {
                    "path": os.path.relpath(path, output_dir),
                    "label": label,
                    "real_source": real_source,
                    "generator": generator,
                    "family": family,
                    "image_sha256": digest,
                    "low_resolution_repair_block": True,
                    "workflow_purpose": "train-candidate",
                    "original_format": "JPEG",
                    "original_mode": "RGB",
                    "original_width": 32,
                    "original_height": 32,
                }
            )

    rows.sort(key=lambda row: (int(row["label"]), str(row["image_sha256"])))
    if len({row["image_sha256"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate hash across repair labels")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    report = {
        "source": "CIFAKE public MIT-declared training split",
        "selection": "lowest SHA-256 per label after exclusions",
        "per_label": args.per_label,
        "rows": len(rows),
        "inventory_sha256_by_label_directory": inventories,
        "manifest_sha256": sha256_file(args.output),
        "excluded_manifests": [manifest.name for manifest in args.exclude_manifest],
        "excluded_paths": len(excluded_paths),
        "excluded_content_hashes": len(excluded_hashes),
        "demo_only_rows": 0,
        "training_allowed": True,
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
