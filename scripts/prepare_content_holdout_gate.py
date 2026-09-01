#!/usr/bin/env python3
"""Build a disjoint real-content gate from immutable WildFake shards.

The gate keeps the new real domains completely outside training. It combines
them with fake generators that were also outside the checkpoint's training
manifest, then checks both resolved-path and byte-content overlap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


def stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def load(path: Path, label: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            if label is not None and int(row["label"]) != label:
                continue
            source = Path(row["path"])
            if not source.is_absolute():
                source = path.parent / source
            source = source.resolve()
            if not source.is_file():
                raise SystemExit(f"missing source: {source}")
            rows.append({**row, "_source": source})
    return rows


def transplant(row: dict, destination: Path) -> dict:
    return {
        **{key: value for key, value in row.items() if key != "_source"},
        "path": os.path.relpath(row["_source"], destination.parent.resolve()),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def limit_groups(rows: list[dict], key: str, maximum: int | None, seed: int) -> list[dict]:
    if maximum is None:
        return rows
    selected = []
    groups = sorted({str(row.get(key, "unknown")) for row in rows})
    for group in groups:
        pool = [row for row in rows if str(row.get(key, "unknown")) == group]
        random.Random(stable_seed(seed, f"{key}-{group}")).shuffle(pool)
        selected.extend(pool[:maximum])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("datasets/family_mixture_v2_source_repair/train.jsonl"),
    )
    parser.add_argument(
        "--external-fakes",
        type=Path,
        default=Path("datasets/external_gates_v2/all_external_sources.jsonl"),
    )
    parser.add_argument(
        "--modern-fakes",
        type=Path,
        default=Path("datasets/family_mixture_v3/eval_modern_pretrain_diagnostic.jsonl"),
    )
    parser.add_argument(
        "--extra-fake-manifest",
        action="append",
        type=Path,
        default=None,
        help="Repeat for additional fully held-out fake sources.",
    )
    parser.add_argument(
        "--only-extra-fakes",
        action="store_true",
        help="Omit the default external and modern fake manifests.",
    )
    parser.add_argument(
        "--real-manifest",
        action="append",
        type=Path,
        default=None,
        help="Repeat for each held-out real source.",
    )
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--max-per-group",
        type=int,
        help="Deterministically cap every fake generator and real source.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("datasets/content_holdout_gate_v1")
    )
    args = parser.parse_args()
    real_manifests = args.real_manifest or [
        Path("datasets/wildfake_real_celebahq_subset/manifest.jsonl"),
        Path("datasets/wildfake_real_church_subset/manifest.jsonl"),
        Path("datasets/wildfake_real_laion5b_subset/manifest.jsonl"),
    ]
    if args.output_root.exists():
        raise SystemExit(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)
    destination = args.output_root / "all.jsonl"

    train_rows = load(args.train)
    external_fakes = [] if args.only_extra_fakes else load(args.external_fakes, label=1)
    modern_fakes = [] if args.only_extra_fakes else load(args.modern_fakes, label=1)
    extra_fakes = [
        row
        for manifest in (args.extra_fake_manifest or [])
        for row in load(manifest, label=1)
    ]
    real_rows = [row for manifest in real_manifests for row in load(manifest, label=0)]
    fake_rows = limit_groups(
        external_fakes + modern_fakes + extra_fakes,
        "generator",
        args.max_per_group,
        args.seed,
    )
    real_rows = limit_groups(
        real_rows, "real_source", args.max_per_group, args.seed
    )
    gate_rows = fake_rows + real_rows

    train_paths = {row["_source"] for row in train_rows}
    gate_paths = {row["_source"] for row in gate_rows}
    path_overlap = train_paths & gate_paths
    if path_overlap:
        raise SystemExit(f"train/gate path overlap: {len(path_overlap)}")
    if len(gate_paths) != len(gate_rows):
        raise SystemExit("duplicate source paths inside content gate")

    train_hashes = {file_sha256(path) for path in train_paths}
    gate_hashes = {file_sha256(path) for path in gate_paths}
    content_overlap = train_hashes & gate_hashes
    if content_overlap:
        raise SystemExit(f"train/gate SHA-256 overlap: {len(content_overlap)}")

    output_rows = [transplant(row, destination) for row in gate_rows]
    random.Random(stable_seed(args.seed, "content-holdout-v1")).shuffle(output_rows)
    destination.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    report = {
        "seed": args.seed,
        "train_rows_checked": len(train_rows),
        "gate_rows": len(gate_rows),
        "fake_rows": len(fake_rows),
        "real_rows": len(real_rows),
        "real_sources": {
            source: sum(row.get("real_source") == source for row in real_rows)
            for source in sorted({row.get("real_source", "unknown") for row in real_rows})
        },
        "fake_generators": {
            generator: sum(row.get("generator") == generator for row in fake_rows)
            for generator in sorted(
                {row.get("generator", "unknown") for row in fake_rows}
            )
        },
        "resolved_path_overlap_with_train": 0,
        "sha256_content_overlap_with_train": 0,
        "forbidden_demo_data_present": False,
        "interpretation": (
            "Diagnostic content/source gate only. Strong performance reduces, but cannot prove "
            "the absence of subject or dataset shortcuts."
        ),
    }
    (args.output_root / "selection.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
