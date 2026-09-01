#!/usr/bin/env python3
"""Add paired low-resolution/source controls and named DiT families to v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
FORBIDDEN_PATH_PARTS = {"demo_only", "demo_only_DO_NOT_TRAIN"}
MODERN_GENERATORS = {
    "FLUX.1-schnell": "FLUX.1-schnell",
    "stable-diffusion-3-medium-diffusers": "Stable-Diffusion-3-Medium",
    "PixArt-Sigma-XL-2-1024-MS": "PixArt-Sigma",
}


def stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def choose(rows: list, count: int, seed: int, name: str) -> list:
    if len(rows) < count:
        raise SystemExit(f"{name}: found {len(rows)}, need {count}")
    return random.Random(stable_seed(seed, name)).sample(rows, count)


def image_paths(root: Path) -> list[Path]:
    return sorted(path.resolve() for path in root.rglob("*") if path.suffix.lower() in SUFFIXES)


def load_manifest(path: Path, label: int | None = None) -> list[dict]:
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
            if FORBIDDEN_PATH_PARTS.intersection(source.parts):
                raise SystemExit(f"forbidden demo-only source: {source}")
            rows.append({**row, "_source": source})
    return rows


def row_for(source: Path, destination: Path, label: int, **metadata) -> dict:
    source = source.resolve()
    if FORBIDDEN_PATH_PARTS.intersection(source.parts):
        raise SystemExit(f"forbidden demo-only source: {source}")
    return {
        "path": os.path.relpath(source, destination.parent.resolve()),
        "label": label,
        **metadata,
    }


def transplant(row: dict, destination: Path, **metadata) -> dict:
    return {
        **{key: value for key, value in row.items() if key not in {"path", "_source"}},
        "path": os.path.relpath(row["_source"], destination.parent.resolve()),
        **metadata,
    }


def write_manifest(path: Path, rows: list[dict], seed: int, name: str) -> None:
    resolved = [(path.parent / row["path"]).resolve() for row in rows]
    if len(set(resolved)) != len(resolved):
        raise SystemExit(f"duplicate paths in {name}")
    shuffled = list(rows)
    random.Random(stable_seed(seed, name)).shuffle(shuffled)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in shuffled),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-train", type=Path, default=Path("datasets/family_mixture_v2/train.jsonl"))
    parser.add_argument("--external-eval", type=Path, default=Path("datasets/external_gates_v2/all_external_sources.jsonl"))
    parser.add_argument("--cifake-train", type=Path, default=Path("datasets/cifake/train"))
    parser.add_argument("--sid-train", type=Path, default=Path("datasets/sid_binary/manifest-train-train-00000-of-00249.jsonl"))
    parser.add_argument("--ditfake-root", type=Path, default=Path("datasets/ditfake/DiTFake/test"))
    parser.add_argument("--imagenet-root", type=Path, default=Path("datasets/wildfake_ddpm2k_train_ddim_test/train/REAL"))
    parser.add_argument("--afhq-manifest", type=Path, default=Path("datasets/official_afhqv2_train_subset/manifest.jsonl"))
    parser.add_argument("--ffhq-root", type=Path, default=Path("datasets/wildfake/extracted/ffhq/ffhq/images"))
    parser.add_argument("--per-paired-group", type=int, default=680)
    parser.add_argument("--per-modern-train-group", type=int, default=680)
    parser.add_argument("--modern-diagnostic-per-group", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output-root", type=Path, default=Path("datasets/family_mixture_v3"))
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)

    train_path = args.output_root / "train.jsonl"
    modern_diagnostic_path = args.output_root / "eval_modern_pretrain_diagnostic.jsonl"
    selection_eval_path = args.output_root / "eval_selection.jsonl"
    base_train = load_manifest(args.v2_train)
    used = {row["_source"] for row in base_train}

    cifake_fake = choose(image_paths(args.cifake_train / "FAKE"), args.per_paired_group, args.seed, "cifake-fake")
    cifake_real = choose(image_paths(args.cifake_train / "REAL"), args.per_paired_group, args.seed, "cifake-real")
    sid_fake_all = load_manifest(args.sid_train, label=1)
    sid_real_all = load_manifest(args.sid_train, label=0)
    sid_count = min(len(sid_fake_all), len(sid_real_all))
    sid_fake = choose(sid_fake_all, sid_count, args.seed, "sid-fake")
    sid_real = choose(sid_real_all, sid_count, args.seed, "sid-real")

    modern_pools = {
        directory: image_paths(args.ditfake_root / directory / "1_fake")
        for directory in MODERN_GENERATORS
    }
    modern_train_directories = ("FLUX.1-schnell", "stable-diffusion-3-medium-diffusers")
    modern_holdout_directory = "PixArt-Sigma-XL-2-1024-MS"
    modern_train = {
        directory: choose(
            modern_pools[directory], args.per_modern_train_group, args.seed, f"modern-train-{directory}"
        )
        for directory in modern_train_directories
    }

    added_fakes = len(cifake_fake) + len(sid_fake) + sum(map(len, modern_train.values()))
    paired_reals = len(cifake_real) + len(sid_real)
    general_real_needed = added_fakes - paired_reals
    base_count, remainder = divmod(general_real_needed, 3)
    general_counts = (base_count + (remainder > 0), base_count + (remainder > 1), base_count)
    imagenet_pool = [path for path in image_paths(args.imagenet_root) if path not in used]
    afhq_pool = [row for row in load_manifest(args.afhq_manifest, label=0) if row["_source"] not in used]
    ffhq_pool = [
        path for path in image_paths(args.ffhq_root)
        if int(path.stem.removeprefix("img")) < 60000 and path not in used
    ]
    extra_imagenet = choose(imagenet_pool, general_counts[0], args.seed, "extra-imagenet")
    extra_afhq = choose(afhq_pool, general_counts[1], args.seed, "extra-afhq")
    extra_ffhq = choose(ffhq_pool, general_counts[2], args.seed, "extra-ffhq")

    train_rows = [transplant(row, train_path) for row in base_train]
    train_rows += [
        row_for(path, train_path, 1, fake_source="CIFAKE", generator="CIFAKE-Stable-Diffusion", family="latent-diffusion")
        for path in cifake_fake
    ]
    train_rows += [
        row_for(path, train_path, 0, real_source="CIFAKE-CIFAR10", family="authentic-low-resolution")
        for path in cifake_real
    ]
    train_rows += [transplant(row, train_path, fake_source="SID_Set", generator="SID_Set-unspecified", family="synthetic-unspecified") for row in sid_fake]
    train_rows += [transplant(row, train_path, real_source="SID_Set", family="authentic-photograph") for row in sid_real]
    for directory, paths in modern_train.items():
        train_rows += [
            row_for(path, train_path, 1, fake_source="DiTFake", generator=MODERN_GENERATORS[directory], family="diffusion-transformer")
            for path in paths
        ]
    train_rows += [row_for(path, train_path, 0, real_source="WildFake-ImageNet", family="authentic-photograph") for path in extra_imagenet]
    train_rows += [transplant(row, train_path) for row in extra_afhq]
    train_rows += [row_for(path, train_path, 0, real_source="FFHQ-train", family="authentic-photograph") for path in extra_ffhq]
    write_manifest(train_path, train_rows, args.seed, "train-v3")

    external_rows = load_manifest(args.external_eval)
    diagnostic_reals = [row for row in external_rows if int(row["label"]) == 0]
    diagnostic_rows = [transplant(row, modern_diagnostic_path) for row in diagnostic_reals]
    for directory, pool in modern_pools.items():
        selected = choose(pool, args.modern_diagnostic_per_group, args.seed, f"modern-diagnostic-{directory}")
        diagnostic_rows += [
            row_for(path, modern_diagnostic_path, 1, fake_source="DiTFake", generator=MODERN_GENERATORS[directory], family="diffusion-transformer")
            for path in selected
        ]
    write_manifest(modern_diagnostic_path, diagnostic_rows, args.seed, "modern-pretrain-diagnostic")

    selection_rows = [transplant(row, selection_eval_path) for row in external_rows]
    pixart_holdout = choose(
        modern_pools[modern_holdout_directory],
        args.modern_diagnostic_per_group,
        args.seed,
        "modern-selection-pixart",
    )
    selection_rows += [
        row_for(path, selection_eval_path, 1, fake_source="DiTFake", generator=MODERN_GENERATORS[modern_holdout_directory], family="diffusion-transformer")
        for path in pixart_holdout
    ]
    write_manifest(selection_eval_path, selection_rows, args.seed, "selection-v3")

    label_counts = {str(label): sum(int(row["label"]) == label for row in train_rows) for label in (0, 1)}
    if label_counts["0"] != label_counts["1"]:
        raise SystemExit(f"unbalanced training manifest: {label_counts}")
    report = {
        "seed": args.seed,
        "forbidden_sources": ["organizer COCO demo-only", "organizer DALL-E Advanced demo-only", "all DiTFake COCO 0_real"],
        "train_label_counts": label_counts,
        "paired_controls": {"CIFAKE_each_label": len(cifake_real), "SID_Set_each_label": len(sid_real)},
        "modern_train": {MODERN_GENERATORS[key]: len(value) for key, value in modern_train.items()},
        "modern_holdout": {MODERN_GENERATORS[modern_holdout_directory]: len(pixart_holdout)},
        "extra_real_sources": {
            "WildFake-ImageNet": len(extra_imagenet),
            "AFHQ-v2": len(extra_afhq),
            "FFHQ-train": len(extra_ffhq),
        },
        "selection_rows": len(selection_rows),
    }
    (args.output_root / "selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
