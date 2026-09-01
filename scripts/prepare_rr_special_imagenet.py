#!/usr/bin/env python3
"""Create manifests using only RR special-scenario fakes and ImageNet reals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
RR_ARCHIVE_SHA256 = "b7f72dabe654877354300c7cd1181f493ccc8299bcea0a76dacf64fea88e0936"


def images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in SUFFIXES)


def special_fakes(root: Path) -> list[Path]:
    return [path for path in images(root) if not path.stem.startswith("normal_")]


def relative_row(path: Path, manifest: Path, label: int, **metadata: str) -> dict:
    return {
        "path": os.path.relpath(path.resolve(), manifest.parent.resolve()),
        "label": label,
        **metadata,
    }


def write_manifest(path: Path, reals: list[Path], fakes: list[Path]) -> None:
    rows = [
        relative_row(
            item,
            path,
            0,
            real_source="WildFake-ImageNet",
            family="authentic-photograph",
        )
        for item in reals
    ]
    rows.extend(
        relative_row(
            item,
            path,
            1,
            fake_source="RRDataset-special-scenarios",
            generator="SD3.5-Large-or-Flux.1-unresolved",
            family="modern-diffusion-transformer",
        )
        for item in fakes
    )
    random.Random(20260829).shuffle(rows)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rr-archive",
        type=Path,
        default=Path("datasets/rrdataset/RRDataset_original_train_val.tar.gz"),
    )
    parser.add_argument(
        "--rr-root",
        type=Path,
        default=Path("datasets/rrdataset/extracted/RRDataset_original_train_val"),
    )
    parser.add_argument(
        "--real-train-root",
        type=Path,
        default=Path("datasets/wildfake_ddpm2k_train_ddim_test/train/REAL"),
    )
    parser.add_argument(
        "--real-eval-root",
        type=Path,
        default=Path("datasets/wildfake_ddpm2k_train_ddim_test/test/REAL"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("datasets/rr_special_imagenet")
    )
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_root}")
    if digest(args.rr_archive) != RR_ARCHIVE_SHA256:
        raise SystemExit("RRDataset archive SHA-256 mismatch")

    train_fakes = special_fakes(args.rr_root / "train" / "ai")
    eval_fakes = special_fakes(args.rr_root / "val" / "ai")
    if (len(train_fakes), len(eval_fakes)) != (771, 157):
        raise SystemExit(
            f"unexpected RR special-scenario counts: {len(train_fakes)}, {len(eval_fakes)}"
        )
    train_reals = images(args.real_train_root)[: len(train_fakes)]
    eval_reals = images(args.real_eval_root)[: len(eval_fakes)]
    if len(train_reals) != len(train_fakes) or len(eval_reals) != len(eval_fakes):
        raise SystemExit("insufficient ImageNet real images")
    if set(train_reals) & set(eval_reals):
        raise SystemExit("real-image train/evaluation overlap detected")

    args.output_root.mkdir(parents=True)
    train_manifest = args.output_root / "train.jsonl"
    eval_manifest = args.output_root / "eval.jsonl"
    write_manifest(train_manifest, train_reals, train_fakes)
    write_manifest(eval_manifest, eval_reals, eval_fakes)
    report = {
        "design": "RR special-scenario fakes against disjoint WildFake ImageNet reals",
        "forbidden_rr_prefix_excluded": "normal_",
        "train": {"real": len(train_reals), "fake": len(train_fakes)},
        "eval": {"real": len(eval_reals), "fake": len(eval_fakes)},
        "rr_archive_sha256": RR_ARCHIVE_SHA256,
    }
    (args.output_root / "selection.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
