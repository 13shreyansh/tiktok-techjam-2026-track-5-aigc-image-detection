#!/usr/bin/env python3
"""Freeze a fresh, source-matched CIFAKE gate for v12 falsification.

Only CIFAKE's official test split is eligible.  Rows already named by any
local JSONL evidence manifest are excluded by source-image hash.  The same
label-blind v12 canonicalization is applied to both classes before the gate is
packaged; the result is evaluation-only and can never enter training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from materialize_canonical_v12 import POLICY, canonical_bytes, file_sha256


PER_LABEL = 1_000
SOURCE_LICENSE = "MIT"
SOURCE_LICENSE_URL = (
    "https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-"
    "Synthetic-Images/blob/e112a942abaecd02b6b1f6f646c807d56be8fb62/README.md#license"
)


def read_jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []


def prior_cifake_hashes(dataset_root: Path) -> tuple[set[str], list[str]]:
    excluded: set[str] = set()
    used_manifests = []
    for manifest in sorted(dataset_root.rglob("*.jsonl")):
        rows = read_jsonl(manifest)
        relevant = False
        for row in rows:
            identity = " ".join(
                str(row.get(key, ""))
                for key in ("dataset", "real_source", "fake_source", "path")
            ).lower()
            if "cifake" not in identity:
                continue
            relevant = True
            for key in ("source_image_sha256", "image_sha256", "sha256"):
                value = row.get(key)
                if isinstance(value, str) and len(value) == 64:
                    excluded.add(value)
        if relevant:
            used_manifests.append(str(manifest.resolve()))
    return excluded, used_manifests


def candidates(source_root: Path, excluded: set[str], label: int) -> list[tuple[str, Path]]:
    directory = source_root / ("FAKE" if label else "REAL")
    by_digest = {}
    for path in sorted(directory.glob("*.jpg")):
        digest = file_sha256(path)
        if digest not in excluded:
            by_digest.setdefault(digest, path.resolve())
    return sorted(by_digest.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    excluded, exclusion_manifests = prior_cifake_hashes(args.dataset_root.resolve())
    selected = {
        label: candidates(args.source.resolve(), excluded, label)[:PER_LABEL]
        for label in (0, 1)
    }
    if any(len(rows) != PER_LABEL for rows in selected.values()):
        raise RuntimeError({label: len(rows) for label, rows in selected.items()})
    source_overlap = {digest for digest, _ in selected[0]} & {
        digest for digest, _ in selected[1]
    }
    if source_overlap:
        raise RuntimeError(f"cross-label source collision: {len(source_overlap)}")

    image_root = args.output / "images"
    image_root.mkdir(parents=True)
    rows = []
    derivative_labels = {}
    source_bytes = 0
    derivative_bytes = 0
    for label in (0, 1):
        for index, (source_digest, source) in enumerate(selected[label], start=1):
            data, original_size = canonical_bytes(source)
            digest = hashlib.sha256(data).hexdigest()
            previous = derivative_labels.get(digest)
            if previous is not None and previous != label:
                raise RuntimeError(f"cross-label canonical collision: {digest}")
            derivative_labels[digest] = label
            destination = image_root / digest[:2] / f"{digest}.jpg"
            destination.parent.mkdir(exist_ok=True)
            if not destination.exists():
                destination.write_bytes(data)
                derivative_bytes += len(data)
            source_bytes += source.stat().st_size
            rows.append(
                {
                    "canonical_format": "JPEG",
                    "canonical_height": 336,
                    "canonical_width": 336,
                    "canonicalization": POLICY,
                    "dataset": "CIFAKE-matched-source-v12-gate",
                    "evaluation_only": True,
                    "family": "latent-diffusion" if label else "low-resolution-photo",
                    "generator": "CIFAKE-Stable-Diffusion" if label else None,
                    "label": label,
                    "license_commercial_use_allowed": True,
                    "organizer_demo_row": False,
                    "original_height": original_size[1],
                    "original_width": original_size[0],
                    "path": str(destination.resolve()),
                    "real_source": "CIFAKE-CIFAR10-test" if not label else None,
                    "sha256": digest,
                    "source_image_sha256": source_digest,
                    "source_license": SOURCE_LICENSE,
                    "source_license_url": SOURCE_LICENSE_URL,
                    "source_path_role": "CIFAKE/test",
                    "training_allowed": False,
                    "workflow_purpose": "v12-source-matched-falsification-gate",
                }
            )
            processed = label * PER_LABEL + index
            if processed % 500 == 0:
                print(f"canonicalized {processed}/{2 * PER_LABEL}", flush=True)

    rows.sort(key=lambda row: (row["label"], row["sha256"]))
    if len({row["source_image_sha256"] for row in rows}) != 2 * PER_LABEL:
        raise RuntimeError("source deduplication invariant failed")
    if len({row["sha256"] for row in rows}) != 2 * PER_LABEL:
        raise RuntimeError("canonical deduplication invariant failed")
    manifest = args.output / "eval_matched.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    inventory = hashlib.sha256(
        "".join(sorted(row["sha256"] for row in rows)).encode()
    ).hexdigest()
    report = {
        "source": "CIFAKE official test split",
        "source_license": SOURCE_LICENSE,
        "source_license_url": SOURCE_LICENSE_URL,
        "selection": "lowest source SHA-256 after all local CIFAKE JSONL exclusions",
        "rows": len(rows),
        "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "unique_source_images": len({row["source_image_sha256"] for row in rows}),
        "unique_derivative_images": len({row["sha256"] for row in rows}),
        "excluded_source_hashes": len(excluded),
        "exclusion_manifest_count": len(exclusion_manifests),
        "source_bytes": source_bytes,
        "derivative_bytes": derivative_bytes,
        "manifest_sha256": file_sha256(manifest),
        "derivative_inventory_sha256": inventory,
        "organizer_demo_rows": 0,
        "training_allowed_rows": 0,
        "interpretation": (
            "This tests same-source real/fake separation after identical canonicalization. "
            "It is one low-resolution source, not proof of broad hidden-set transfer."
        ),
    }
    (args.output / "selection.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
