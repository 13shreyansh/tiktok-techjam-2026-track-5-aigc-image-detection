#!/usr/bin/env python3
"""Freeze a fresh high-resolution, same-source SID_Set gate for v12.

The source is immutable validation shard 00001, which was not present in any
earlier local experiment.  Historical SID identities and every v12 identity
are excluded before deterministic selection.  Both labels receive the same
v12 canonicalization.  The resulting pixels are evaluation-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from materialize_canonical_v12 import POLICY, canonical_bytes, file_sha256


PER_LABEL = 284
SOURCE_MANIFEST_SHA256 = "5f0815ac6ffac25bfd7724747a53d7536ea464d600dab9a3071480e724313c7f"
SOURCE_SHARD = "validation-00001-of-00034.parquet"
SOURCE_SHARD_BYTES = 505_844_042
SOURCE_SHARD_SHA256 = "1447bbd98adf7eda68fca5615560c6b1de34c8e30157ff6b34ebd1e015a18042"
SOURCE_REVISION = "dc03ead57929879319ce30a82bfcfb8d317b10bd"
SOURCE_LICENSE = "CC-BY-4.0"
SOURCE_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"


def read_jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []


def prior_sid_hashes(dataset_root: Path, source_manifest: Path) -> tuple[set[str], list[str]]:
    excluded: set[str] = set()
    used_manifests = []
    source_manifest = source_manifest.resolve()
    for manifest in sorted(dataset_root.rglob("*.jsonl")):
        if manifest.resolve() == source_manifest:
            continue
        rows = read_jsonl(manifest)
        relevant = False
        for row in rows:
            identity = " ".join(
                str(row.get(key, ""))
                for key in ("dataset", "real_source", "fake_source", "generator", "path")
            ).lower()
            if "sid_set" not in identity and "sid-set" not in identity and "sid_binary" not in identity:
                continue
            relevant = True
            for key in ("source_image_sha256", "image_sha256", "sha256"):
                value = row.get(key)
                if isinstance(value, str) and len(value) == 64:
                    excluded.add(value)
        if relevant:
            used_manifests.append(str(manifest.resolve()))
    return excluded, used_manifests


def source_rows(path: Path, excluded: set[str]) -> dict[int, list[dict]]:
    if file_sha256(path) != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("fresh SID source manifest checksum mismatch")
    by_label: dict[int, dict[str, dict]] = {0: {}, 1: {}}
    for row in read_jsonl(path):
        label = int(row["label"])
        if label not in by_label:
            continue
        digest = str(row["image_sha256"])
        if digest in excluded:
            continue
        source = Path(row["path"])
        if not source.is_absolute():
            source = (path.parent / source).resolve()
        if file_sha256(source) != digest:
            raise RuntimeError(f"source image checksum mismatch: {source}")
        by_label[label].setdefault(digest, {**row, "_source": source})
    return {
        label: [by_label[label][digest] for digest in sorted(by_label[label])[:PER_LABEL]]
        for label in (0, 1)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-shard", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.source_shard.name != SOURCE_SHARD:
        raise RuntimeError("fresh SID shard name mismatch")
    if args.source_shard.stat().st_size != SOURCE_SHARD_BYTES or file_sha256(
        args.source_shard
    ) != SOURCE_SHARD_SHA256:
        raise RuntimeError("fresh SID shard identity mismatch")

    excluded, exclusion_manifests = prior_sid_hashes(
        args.dataset_root.resolve(), args.source_manifest.resolve()
    )
    selected = source_rows(args.source_manifest.resolve(), excluded)
    if any(len(rows) != PER_LABEL for rows in selected.values()):
        raise RuntimeError({label: len(rows) for label, rows in selected.items()})
    if {row["image_sha256"] for row in selected[0]} & {
        row["image_sha256"] for row in selected[1]
    }:
        raise RuntimeError("cross-label source collision")

    image_root = args.output / "images"
    image_root.mkdir(parents=True)
    rows = []
    derivative_labels = {}
    source_bytes = 0
    derivative_bytes = 0
    for label in (0, 1):
        for index, source_row in enumerate(selected[label], start=1):
            source = source_row["_source"]
            source_digest = source_row["image_sha256"]
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
                    "dataset": "SID_Set-fresh-validation-v12-gate",
                    "evaluation_only": True,
                    "family": "fully-synthetic-unspecified" if label else "diverse-real-photography",
                    "generator": "SID_Set-unspecified" if label else None,
                    "label": label,
                    "license_commercial_use_allowed": True,
                    "organizer_demo_row": False,
                    "original_height": original_size[1],
                    "original_width": original_size[0],
                    "path": str(destination.resolve()),
                    "real_source": "SID_Set-validation-00001" if not label else None,
                    "sha256": digest,
                    "source_image_sha256": source_digest,
                    "source_license": SOURCE_LICENSE,
                    "source_license_url": SOURCE_LICENSE_URL,
                    "source_path_role": "SID_Set/validation-00001-of-00034",
                    "source_revision": SOURCE_REVISION,
                    "training_allowed": False,
                    "workflow_purpose": "v12-fresh-highres-source-matched-falsification-gate",
                }
            )
            processed = label * PER_LABEL + index
            if processed % 100 == 0 or processed == 2 * PER_LABEL:
                print(f"canonicalized {processed}/{2 * PER_LABEL}", flush=True)

    rows.sort(key=lambda row: (row["label"], row["sha256"]))
    if len({row["source_image_sha256"] for row in rows}) != 2 * PER_LABEL:
        raise RuntimeError("source deduplication invariant failed")
    if len({row["sha256"] for row in rows}) != 2 * PER_LABEL:
        raise RuntimeError("canonical deduplication invariant failed")
    manifest = args.output / "eval_matched.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    inventory = hashlib.sha256(
        "".join(sorted(row["sha256"] for row in rows)).encode()
    ).hexdigest()
    report = {
        "source": "SID_Set fresh validation shard 00001-of-00034",
        "source_revision": SOURCE_REVISION,
        "source_shard_sha256": SOURCE_SHARD_SHA256,
        "source_license": SOURCE_LICENSE,
        "source_license_url": SOURCE_LICENSE_URL,
        "selection": "lowest source SHA-256 per label after all prior local SID JSONL exclusions",
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
            "This tests fresh high-resolution same-source real/fake separation after "
            "identical canonicalization. SID_Set does not expose generator identity, "
            "so it is not an unseen-generator-family claim."
        ),
    }
    (args.output / "selection.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
