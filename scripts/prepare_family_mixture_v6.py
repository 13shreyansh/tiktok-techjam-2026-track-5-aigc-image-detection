#!/usr/bin/env python3
"""Build v6 with paired disjoint controls for both LAION and Church.

V6 adds Church shard A to the v5 training mixture and balances it with unused
FLUX and Stable Diffusion 3 images.  Byte-disjoint shard B from both LAION and
Church remains evaluation-only.  Imagen and PixArt remain complete generator
holdouts.  Organizer demo-only data is rejected by the shared loader.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from prepare_family_mixture_v5 import (
    group_counts,
    load,
    stable_seed,
    validate_disjoint,
    write,
)


SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MODERN_GENERATORS = {
    "FLUX.1-schnell": "FLUX.1-schnell",
    "stable-diffusion-3-medium-diffusers": "Stable-Diffusion-3-Medium",
}


def choose_unused(
    root: Path, used: set[Path], count: int, seed: int, name: str
) -> list[dict]:
    candidates = sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.suffix.lower() in SUFFIXES and path.resolve() not in used
    )
    if len(candidates) < count:
        raise SystemExit(f"{name}: found {len(candidates)} unused images, need {count}")
    selected = random.Random(stable_seed(seed, name)).sample(candidates, count)
    return [
        {
            "path": str(path),
            "label": 1,
            "fake_source": "DiTFake",
            "generator": MODERN_GENERATORS[name],
            "family": "diffusion-transformer",
            "_source": path,
        }
        for path in selected
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v5-train", type=Path, default=Path("datasets/family_mixture_v5/train.jsonl")
    )
    parser.add_argument(
        "--v3-selection",
        type=Path,
        default=Path("datasets/family_mixture_v3/eval_selection.jsonl"),
    )
    parser.add_argument(
        "--ditfake-root", type=Path, default=Path("datasets/ditfake/DiTFake/test")
    )
    parser.add_argument(
        "--imagen", type=Path, default=Path("datasets/wildfake_imagen_subset/manifest.jsonl")
    )
    parser.add_argument(
        "--church-train",
        type=Path,
        default=Path("datasets/wildfake_real_church_subset/manifest.jsonl"),
    )
    parser.add_argument(
        "--church-eval",
        type=Path,
        default=Path("datasets/wildfake_real_church_subset_b/manifest.jsonl"),
    )
    parser.add_argument(
        "--laion-eval",
        type=Path,
        default=Path("datasets/wildfake_real_laion5b_subset_b/manifest.jsonl"),
    )
    parser.add_argument("--holdout-per-real-source", type=int, default=1024)
    parser.add_argument("--extra-per-modern-generator", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--output-root", type=Path, default=Path("datasets/family_mixture_v6")
    )
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)

    train_rows = load(args.v5_train)
    used = {row["_source"] for row in train_rows}
    church_train = load(args.church_train, label=0)
    for row in church_train:
        if row["_source"] in used:
            raise SystemExit(f"Church shard A already in v5 training: {row['_source']}")
    train_rows += church_train

    extra_fakes: list[dict] = []
    for directory in MODERN_GENERATORS:
        selected = choose_unused(
            args.ditfake_root / directory / "1_fake",
            used,
            args.extra_per_modern_generator,
            args.seed,
            directory,
        )
        extra_fakes += selected
        used.update(row["_source"] for row in selected)
    train_rows += extra_fakes

    label_counts = Counter(int(row["label"]) for row in train_rows)
    if label_counts[0] != label_counts[1]:
        raise SystemExit(f"unbalanced train labels: {dict(label_counts)}")

    v3_selection = load(args.v3_selection)
    imagen_holdout = load(args.imagen, label=1)
    church_holdout = load(
        args.church_eval, label=0, limit=args.holdout_per_real_source
    )
    laion_holdout = load(
        args.laion_eval, label=0, limit=args.holdout_per_real_source
    )
    selection_rows = v3_selection + imagen_holdout + church_holdout + laion_holdout
    disjoint = validate_disjoint(train_rows, selection_rows)
    content_rows = [row for row in v3_selection if int(row["label"]) == 1]
    content_rows += imagen_holdout + church_holdout + laion_holdout

    train_path = args.output_root / "train.jsonl"
    selection_path = args.output_root / "eval_selection.jsonl"
    content_path = args.output_root / "eval_content_holdout.jsonl"
    write(train_path, train_rows, args.seed, "v6-train")
    write(selection_path, selection_rows, args.seed, "v6-selection")
    write(content_path, content_rows, args.seed, "v6-content-holdout")
    report = {
        "seed": args.seed,
        "design": (
            "LAION and Church shard A in training; byte-disjoint shard B of "
            "both sources evaluation-only; Imagen and PixArt complete holdouts"
        ),
        "forbidden_sources": [
            "organizer COCO demo-only",
            "organizer DALL-E Advanced demo-only",
            "all DiTFake COCO 0_real",
        ],
        "train_label_counts": {
            str(key): value for key, value in sorted(label_counts.items())
        },
        "train_fake_generators": group_counts(train_rows, 1, "generator"),
        "train_real_sources": group_counts(train_rows, 0, "real_source"),
        "new_modern_fakes": dict(
            sorted(Counter(row["generator"] for row in extra_fakes).items())
        ),
        "complete_fake_holdouts": ["Imagen", "PixArt-Sigma"],
        "disjoint_same_source_real_holdouts": {
            "WildFake-LAION5B": len(laion_holdout),
            "WildFake-LSUN-Church": len(church_holdout),
        },
        "selection_rows": len(selection_rows),
        "content_holdout_rows": len(content_rows),
        **disjoint,
    }
    (args.output_root / "selection.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
