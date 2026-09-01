#!/usr/bin/env python3
"""Create diagnostic, non-training gates from approved independent sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
FORBIDDEN_PATH_PARTS = {"demo_only", "demo_only_DO_NOT_TRAIN"}


def stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def choose(rows: list, count: int, seed: int, name: str) -> list:
    if len(rows) < count:
        raise SystemExit(f"{name} has {len(rows)} rows, fewer than requested {count}")
    return random.Random(stable_seed(seed, name)).sample(rows, count)


def resolve_row(row: dict, manifest: Path) -> dict:
    source = Path(row["path"])
    if not source.is_absolute():
        source = manifest.parent / source
    source = source.resolve()
    if not source.is_file():
        raise SystemExit(f"missing image referenced by {manifest}: {source}")
    if FORBIDDEN_PATH_PARTS.intersection(source.parts):
        raise SystemExit(f"refusing forbidden demo-only source: {source}")
    return {**row, "_source": source}


def load_manifest(path: Path, label: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if label is None or int(row["label"]) == label:
                rows.append(resolve_row(row, path))
    return rows


def images(root: Path) -> list[Path]:
    return sorted(path.resolve() for path in root.rglob("*") if path.suffix.lower() in SUFFIXES)


def output_row(row: dict, manifest: Path, **metadata) -> dict:
    return {
        **{key: value for key, value in row.items() if key not in {"path", "_source"}},
        "path": os.path.relpath(row["_source"], manifest.parent.resolve()),
        **metadata,
    }


def path_row(path: Path, label: int, manifest: Path, **metadata) -> dict:
    if FORBIDDEN_PATH_PARTS.intersection(path.resolve().parts):
        raise SystemExit(f"refusing forbidden demo-only source: {path}")
    return {"path": os.path.relpath(path.resolve(), manifest.parent.resolve()), "label": label, **metadata}


def write_manifest(path: Path, rows: list[dict], seed: int, name: str) -> None:
    shuffled = list(rows)
    random.Random(stable_seed(seed, name)).shuffle(shuffled)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in shuffled),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--heldout-generator-manifest",
        type=Path,
        default=Path("datasets/family_mixture_v2/eval_heldout_generators_known_reals.jsonl"),
    )
    parser.add_argument(
        "--ffhq-manifest",
        type=Path,
        default=Path("datasets/family_mixture_v2/eval_heldout_generators_ffhq_reals.jsonl"),
    )
    parser.add_argument("--cifake-root", type=Path, default=Path("datasets/cifake/test"))
    parser.add_argument(
        "--sid-manifest",
        type=Path,
        default=Path("datasets/sid_binary/manifest-test-validation-00000-of-00034.jsonl"),
    )
    parser.add_argument("--per-class", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output-root", type=Path, default=Path("datasets/external_gates_v1"))
    args = parser.parse_args()

    if args.output_root.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_root}")
    args.output_root.mkdir(parents=True)

    heldout_fakes = load_manifest(args.heldout_generator_manifest, label=1)
    ffhq_reals = load_manifest(args.ffhq_manifest, label=0)
    cifake_reals = choose(images(args.cifake_root / "REAL"), args.per_class, args.seed, "cifake-reals")
    cifake_fakes = choose(images(args.cifake_root / "FAKE"), args.per_class, args.seed, "cifake-fakes")

    cifake_real_gate = args.output_root / "heldout_generators_vs_cifake_reals.jsonl"
    rows = [output_row(row, cifake_real_gate) for row in heldout_fakes]
    rows += [
        path_row(path, 0, cifake_real_gate, real_source="CIFAKE-CIFAR10", family="authentic-low-resolution")
        for path in cifake_reals
    ]
    write_manifest(cifake_real_gate, rows, args.seed, "heldout-vs-cifake-real")

    cifake_fake_gate = args.output_root / "cifake_fakes_vs_ffhq_reals.jsonl"
    selected_ffhq = choose(ffhq_reals, args.per_class, args.seed, "ffhq-reals-for-cifake")
    rows = [
        path_row(path, 1, cifake_fake_gate, fake_source="CIFAKE", generator="CIFAKE-Stable-Diffusion", family="latent-diffusion")
        for path in cifake_fakes
    ]
    rows += [output_row(row, cifake_fake_gate) for row in selected_ffhq]
    write_manifest(cifake_fake_gate, rows, args.seed, "cifake-fake-vs-ffhq-real")

    sid_source_rows = load_manifest(args.sid_manifest)
    sid_gate = args.output_root / "sid_set_unseen_source.jsonl"
    sid_rows = []
    for row in sid_source_rows:
        if int(row["label"]) == 1:
            sid_rows.append(
                output_row(
                    row,
                    sid_gate,
                    fake_source="SID_Set",
                    generator="SID_Set-unspecified",
                    family="synthetic-unspecified",
                )
            )
        else:
            sid_rows.append(
                output_row(row, sid_gate, real_source="SID_Set", family="authentic-photograph")
            )
    write_manifest(sid_gate, sid_rows, args.seed, "sid-source-gate")

    all_sources_gate = args.output_root / "all_external_sources.jsonl"
    all_rows = [output_row(row, all_sources_gate) for row in heldout_fakes]
    all_rows += [output_row(row, all_sources_gate) for row in selected_ffhq]
    all_rows += [
        path_row(
            path,
            0,
            all_sources_gate,
            real_source="CIFAKE-CIFAR10",
            family="authentic-low-resolution",
        )
        for path in cifake_reals
    ]
    all_rows += [
        path_row(
            path,
            1,
            all_sources_gate,
            fake_source="CIFAKE",
            generator="CIFAKE-Stable-Diffusion",
            family="latent-diffusion",
        )
        for path in cifake_fakes
    ]
    for row in sid_source_rows:
        if int(row["label"]) == 1:
            all_rows.append(
                output_row(
                    row,
                    all_sources_gate,
                    fake_source="SID_Set",
                    generator="SID_Set-unspecified",
                    family="synthetic-unspecified",
                )
            )
        else:
            all_rows.append(
                output_row(
                    row,
                    all_sources_gate,
                    real_source="SID_Set",
                    family="authentic-photograph",
                )
            )
    write_manifest(all_sources_gate, all_rows, args.seed, "all-external-sources")

    report = {
        "purpose": "diagnostic-only source and generator transfer gates; excluded from training",
        "seed": args.seed,
        "forbidden_sources": ["COCO demo-only", "DALL-E Advanced demo-only"],
        "gates": {
            cifake_real_gate.name: {"fake": len(heldout_fakes), "real": len(cifake_reals)},
            cifake_fake_gate.name: {"fake": len(cifake_fakes), "real": len(selected_ffhq)},
            sid_gate.name: {
                "fake": sum(int(row["label"]) == 1 for row in sid_rows),
                "real": sum(int(row["label"]) == 0 for row in sid_rows),
            },
            all_sources_gate.name: {
                "fake": sum(int(row["label"]) == 1 for row in all_rows),
                "real": sum(int(row["label"]) == 0 for row in all_rows),
            },
        },
    }
    (args.output_root / "selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
