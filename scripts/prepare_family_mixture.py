#!/usr/bin/env python3
"""Build balanced multi-family train and held-out evaluation manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def images(root: Path) -> list[Path]:
    return sorted(path.resolve() for path in root.rglob("*") if path.suffix.lower() in SUFFIXES)


def stable_seed(seed: int, name: str) -> int:
    value = hashlib.sha256(f"{seed}\0{name}".encode()).digest()
    return int.from_bytes(value[:8], "big")


def choose(pool: list, count: int, seed: int, name: str) -> list:
    if len(pool) < count:
        raise SystemExit(f"{name} has {len(pool)} rows, fewer than requested {count}")
    return random.Random(stable_seed(seed, name)).sample(pool, count)


def load_manifest(path: Path, label: int = 1) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["label"]) != label:
                continue
            source = Path(row["path"])
            if not source.is_absolute():
                source = path.parent / source
            source = source.resolve()
            if not source.is_file():
                raise SystemExit(f"manifest source does not exist: {source}")
            rows.append({**row, "_resolved_path": source})
    return rows


def row_for(path: Path, manifest: Path, label: int, **metadata) -> dict:
    return {
        "path": os.path.relpath(path.resolve(), manifest.parent.resolve()),
        "label": label,
        **metadata,
    }


def transplant(row: dict, manifest: Path) -> dict:
    return {
        **{key: value for key, value in row.items() if key != "_resolved_path"},
        "path": os.path.relpath(row["_resolved_path"], manifest.parent.resolve()),
    }


def write_manifest(path: Path, rows: list[dict], seed: int, name: str) -> None:
    if len({Path(row["path"]) for row in rows}) != len(rows):
        raise SystemExit(f"duplicate paths in {name}")
    shuffled = list(rows)
    random.Random(stable_seed(seed, name)).shuffle(shuffled)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in shuffled),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddpm-root", type=Path, default=Path("datasets/wildfake_ddpm2k_train_ddim_test/train/FAKE"))
    parser.add_argument("--ddim-root", type=Path, default=Path("datasets/wildfake_ddpm2k_train_ddim_test/test/FAKE"))
    parser.add_argument("--imagenet-train-root", type=Path, default=Path("datasets/wildfake_ddpm2k_train_ddim_test/train/REAL"))
    parser.add_argument("--imagenet-eval-root", type=Path, default=Path("datasets/wildfake_ddpm2k_train_ddim_test/test/REAL"))
    parser.add_argument("--gan-train-manifest", type=Path, default=Path("datasets/wildfake_gan_train_subset/manifest.jsonl"))
    parser.add_argument("--gan-eval-manifest", type=Path, default=Path("datasets/wildfake_gan_eval_subset_v2/manifest.jsonl"))
    parser.add_argument("--sd15-manifest", type=Path, default=Path("datasets/wildfake_sdv15_train_subset/manifest.jsonl"))
    parser.add_argument("--rr-train-manifest", type=Path, default=Path("datasets/rr_special_imagenet/train.jsonl"))
    parser.add_argument("--rr-eval-manifest", type=Path, default=Path("datasets/rr_special_imagenet/eval.jsonl"))
    parser.add_argument("--afhq-train-manifest", type=Path, default=Path("datasets/official_afhqv2_train_subset/manifest.jsonl"))
    parser.add_argument("--afhq-eval-manifest", type=Path, default=Path("datasets/official_afhqv2_test_subset/manifest.jsonl"))
    parser.add_argument("--ffhq-root", type=Path, default=Path("datasets/wildfake/extracted/ffhq/ffhq/images"))
    parser.add_argument("--train-per-fake-group", type=int, default=680)
    parser.add_argument("--ffhq-holdout", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output-root", type=Path, default=Path("datasets/family_mixture_v2"))
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_root}")
    args.output_root.mkdir(parents=True)

    train_path = args.output_root / "train.jsonl"
    known_eval_path = args.output_root / "eval_heldout_generators_known_reals.jsonl"
    ffhq_eval_path = args.output_root / "eval_heldout_generators_ffhq_reals.jsonl"
    modern_eval_path = args.output_root / "eval_modern_rr_special.jsonl"

    fake_groups: dict[str, list[dict]] = {
        "DDPM": [
            {"_resolved_path": path, "label": 1, "fake_source": "WildFake", "generator": "DDPM", "family": "classical-diffusion"}
            for path in images(args.ddpm_root)
        ],
        "SDv1.5-DPMSolver": load_manifest(args.sd15_manifest),
        "RR-SD3.5-or-Flux": load_manifest(args.rr_train_manifest),
    }
    for row in load_manifest(args.gan_train_manifest):
        fake_groups.setdefault(str(row["generator"]), []).append(row)
    expected_train_groups = {"DDPM", "SDv1.5-DPMSolver", "RR-SD3.5-or-Flux", "styleGAN", "BigGAN", "starGAN"}
    if set(fake_groups) != expected_train_groups:
        raise SystemExit(f"unexpected training fake groups: {sorted(fake_groups)}")

    train_fakes = []
    for generator in sorted(fake_groups):
        selected = choose(fake_groups[generator], args.train_per_fake_group, args.seed, f"train-fake-{generator}")
        train_fakes.extend(transplant(row, train_path) for row in selected)
    base_reals, remainder_reals = divmod(len(train_fakes), 3)
    imagenet_count = base_reals + (1 if remainder_reals > 0 else 0)
    afhq_count = base_reals + (1 if remainder_reals > 1 else 0)
    ffhq_count = len(train_fakes) - imagenet_count - afhq_count
    imagenet_train = choose(images(args.imagenet_train_root), imagenet_count, args.seed, "train-real-imagenet")
    afhq_train = choose(load_manifest(args.afhq_train_manifest, label=0), afhq_count, args.seed, "train-real-afhq-v2")
    ffhq_all = images(args.ffhq_root)
    ffhq_train_pool = [path for path in ffhq_all if int(path.stem.removeprefix("img")) < 60000]
    ffhq_eval_pool = [path for path in ffhq_all if int(path.stem.removeprefix("img")) >= 60000]
    ffhq_train = choose(ffhq_train_pool, ffhq_count, args.seed, "train-real-ffhq-official-split")
    train_reals = [row_for(path, train_path, 0, real_source="WildFake-ImageNet", family="authentic-photograph") for path in imagenet_train]
    train_reals += [transplant(row, train_path) for row in afhq_train]
    train_reals += [row_for(path, train_path, 0, real_source="FFHQ-train", family="authentic-photograph") for path in ffhq_train]
    write_manifest(train_path, train_reals + train_fakes, args.seed, "train")

    eval_fake_groups: dict[str, list[dict]] = {
        "DDIM": [
            {"_resolved_path": path, "label": 1, "fake_source": "WildFake", "generator": "DDIM", "family": "classical-diffusion"}
            for path in images(args.ddim_root)
        ]
    }
    for row in load_manifest(args.gan_eval_manifest):
        eval_fake_groups.setdefault(str(row["generator"]), []).append(row)
    expected_eval_groups = {"DDIM", "GigaGAN", "GALIP", "DF-GAN"}
    if set(eval_fake_groups) != expected_eval_groups:
        raise SystemExit(f"unexpected evaluation fake groups: {sorted(eval_fake_groups)}")
    eval_fakes_raw = [row for generator in sorted(eval_fake_groups) for row in eval_fake_groups[generator]]

    known_fakes = [transplant(row, known_eval_path) for row in eval_fakes_raw]
    known_reals = [row_for(path, known_eval_path, 0, real_source="WildFake-ImageNet", family="authentic-photograph") for path in images(args.imagenet_eval_root)]
    afhq_eval = load_manifest(args.afhq_eval_manifest, label=0)
    known_reals += [transplant(row, known_eval_path) for row in afhq_eval]
    write_manifest(known_eval_path, known_reals + known_fakes, args.seed, "known-eval")

    ffhq_fakes = [transplant(row, ffhq_eval_path) for row in eval_fakes_raw]
    ffhq_reals_selected = choose(ffhq_eval_pool, args.ffhq_holdout, args.seed, "holdout-real-ffhq-official-split")
    ffhq_reals = [row_for(path, ffhq_eval_path, 0, real_source="FFHQ-validation", family="authentic-photograph") for path in ffhq_reals_selected]
    write_manifest(ffhq_eval_path, ffhq_reals + ffhq_fakes, args.seed, "ffhq-eval")

    modern_fakes_raw = load_manifest(args.rr_eval_manifest)
    modern_reals_raw = choose(images(args.imagenet_eval_root), len(modern_fakes_raw), args.seed, "modern-real-imagenet")
    modern_rows = [transplant(row, modern_eval_path) for row in modern_fakes_raw]
    modern_rows += [row_for(path, modern_eval_path, 0, real_source="WildFake-ImageNet", family="authentic-photograph") for path in modern_reals_raw]
    write_manifest(modern_eval_path, modern_rows, args.seed, "modern-eval")

    report = {
        "seed": args.seed,
        "forbidden_sources": ["COCO demo-only", "DALL-E Advanced demo-only", "RR normal fakes", "all RR reals"],
        "ffhq_split_policy": "official convention: numeric IDs 00000-59999 eligible for training; 60000-69999 eligible for validation",
        "train": {
            "fake_groups": {name: args.train_per_fake_group for name in sorted(fake_groups)},
            "real_groups": {"WildFake-ImageNet": len(imagenet_train), "AFHQ-v2": len(afhq_train), "FFHQ-train": len(ffhq_train)},
            "rows": len(train_reals) + len(train_fakes),
        },
        "eval_heldout_generators_known_reals": {
            "fake_groups": {name: len(rows) for name, rows in sorted(eval_fake_groups.items())},
            "real_groups": {"WildFake-ImageNet": len(images(args.imagenet_eval_root)), "AFHQ-v2": len(afhq_eval)},
            "rows": len(known_reals) + len(known_fakes),
        },
        "eval_heldout_generators_ffhq_reals": {"FFHQ-validation": len(ffhq_reals), "fake": len(ffhq_fakes)},
        "eval_modern_rr_special": {"real": len(modern_reals_raw), "fake": len(modern_fakes_raw)},
    }
    (args.output_root / "selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
