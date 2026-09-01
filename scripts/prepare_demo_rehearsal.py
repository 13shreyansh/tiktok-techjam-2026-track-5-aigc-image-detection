#!/usr/bin/env python3
"""Freeze a small, non-cherry-picked demo rehearsal input set.

The source rows are already-consumed public-data evaluation rows.  They are not
used to select, tune or calibrate a model, and the generated demo result is not
evaluation evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


SEED = 20260831


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def rank(role: str, image_sha256: str) -> str:
    return hashlib.sha256(f"{SEED}:demo-rehearsal:{role}:{image_sha256}".encode()).hexdigest()


def resolve_rows(manifest: Path, predicate) -> list[tuple[dict, Path, str]]:
    resolved = []
    for row in read_rows(manifest):
        if not predicate(row):
            continue
        source = (manifest.parent / row["path"]).resolve()
        if not source.is_file() or "demo_only" in str(source).lower():
            continue
        observed = sha256_file(source)
        expected = row.get("image_sha256")
        if expected and expected != observed:
            raise RuntimeError(f"checksum mismatch: {source}")
        resolved.append((row, source, observed))
    return resolved


def choose(candidates: list[tuple[dict, Path, str]], role: str, count: int) -> list:
    if len(candidates) < count:
        raise RuntimeError(f"not enough candidates for {role}: {len(candidates)}")
    return sorted(candidates, key=lambda item: rank(role, item[2]))[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--broad-manifest",
        type=Path,
        default=Path("datasets/family_mixture_v6/eval_selection.jsonl"),
    )
    parser.add_argument(
        "--qwen-manifest",
        type=Path,
        default=Path("datasets/qwen_image_bench_holdout/combined_gate.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/demo-rehearsal-input")
    )
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to reuse existing demo directory: {args.output}")

    real = resolve_rows(
        args.broad_manifest,
        lambda row: int(row["label"]) == 0 and row.get("real_source") == "CIFAKE-CIFAR10",
    )
    cifake_fake = resolve_rows(
        args.broad_manifest,
        lambda row: int(row["label"]) == 1 and row.get("generator") == "CIFAKE-Stable-Diffusion",
    )
    qwen_fake = resolve_rows(
        args.qwen_manifest,
        lambda row: int(row["label"]) == 1 and row.get("generator") == "FLUX.2-pro",
    )
    selected = [
        *(('cifake-real', item) for item in choose(real, "cifake-real", 2)),
        ('cifake-stable-diffusion', choose(cifake_fake, "cifake-fake", 1)[0]),
        ('qwen-flux2-pro', choose(qwen_fake, "qwen-fake", 1)[0]),
    ]
    args.output.mkdir(parents=True)
    records = []
    for index, (role, (row, source, digest)) in enumerate(selected):
        target = args.output / f"{index:02d}-{role}{source.suffix.lower()}"
        shutil.copy2(source, target)
        records.append(
            {
                "role": role,
                "label": int(row["label"]),
                "generator": row.get("generator"),
                "real_source": row.get("real_source"),
                "image_sha256": digest,
                "output_name": target.name,
                "selection_rank": rank(role, digest),
                "workflow_purpose": "demo-rehearsal-only-not-evaluation",
            }
        )
    manifest = args.output / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "seed": SEED,
                "selection": "lowest role-specific SHA-256 rank before inference",
                "rows": records,
                "organizer_demo_rows": 0,
                "model_selection_or_calibration": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"images": len(records), "manifest_sha256": sha256_file(manifest)}))


if __name__ == "__main__":
    main()
