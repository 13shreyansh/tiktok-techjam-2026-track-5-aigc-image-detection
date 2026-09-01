#!/usr/bin/env python3
"""Expand v3 on both labels while preserving complete source holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path


FORBIDDEN_PATH_PARTS = {"demo_only", "demo_only_DO_NOT_TRAIN"}


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
            if FORBIDDEN_PATH_PARTS.intersection(source.parts):
                raise SystemExit(f"forbidden demo-only source: {source}")
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


def validate_disjoint(train: list[dict], evaluation: list[dict]) -> dict:
    train_paths = {row["_source"] for row in train}
    eval_paths = {row["_source"] for row in evaluation}
    if len(train_paths) != len(train) or len(eval_paths) != len(evaluation):
        raise SystemExit("duplicate path inside train or evaluation manifest")
    if train_paths & eval_paths:
        raise SystemExit(f"train/evaluation path overlap: {len(train_paths & eval_paths)}")
    train_hashes = {file_sha256(path) for path in train_paths}
    eval_hashes = {file_sha256(path) for path in eval_paths}
    if train_hashes & eval_hashes:
        raise SystemExit(
            f"train/evaluation SHA-256 overlap: {len(train_hashes & eval_hashes)}"
        )
    return {
        "resolved_path_overlap": 0,
        "sha256_content_overlap": 0,
        "train_unique_paths": len(train_paths),
        "evaluation_unique_paths": len(eval_paths),
    }


def write(path: Path, rows: list[dict], seed: int, name: str) -> None:
    output = [transplant(row, path) for row in rows]
    random.Random(stable_seed(seed, name)).shuffle(output)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output),
        encoding="utf-8",
    )


def group_counts(rows: list[dict], label: int, key: str) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(row.get(key, "unknown"))
                for row in rows
                if int(row["label"]) == label
            ).items()
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-train", type=Path, default=Path("datasets/family_mixture_v3/train.jsonl"))
    parser.add_argument("--v3-selection", type=Path, default=Path("datasets/family_mixture_v3/eval_selection.jsonl"))
    parser.add_argument("--adm", type=Path, default=Path("datasets/wildfake_adm_subset/manifest.jsonl"))
    parser.add_argument("--vqdm", type=Path, default=Path("datasets/wildfake_vqdm_subset/manifest.jsonl"))
    parser.add_argument("--imagen", type=Path, default=Path("datasets/wildfake_imagen_subset/manifest.jsonl"))
    parser.add_argument("--celebahq", type=Path, default=Path("datasets/wildfake_real_celebahq_subset/manifest.jsonl"))
    parser.add_argument("--church", type=Path, default=Path("datasets/wildfake_real_church_subset/manifest.jsonl"))
    parser.add_argument("--laion", type=Path, default=Path("datasets/wildfake_real_laion5b_subset/manifest.jsonl"))
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output-root", type=Path, default=Path("datasets/family_mixture_v4"))
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)

    v3_train = load(args.v3_train)
    v3_selection = load(args.v3_selection)
    new_train_fakes = load(args.adm, label=1) + load(args.vqdm, label=1)
    new_train_reals = load(args.celebahq, label=0) + load(args.church, label=0)
    train_rows = v3_train + new_train_fakes + new_train_reals
    label_counts = Counter(int(row["label"]) for row in train_rows)
    if label_counts[0] != label_counts[1]:
        raise SystemExit(f"unbalanced train labels: {dict(label_counts)}")

    imagen_holdout = load(args.imagen, label=1)
    laion_holdout = load(args.laion, label=0)
    selection_rows = v3_selection + imagen_holdout + laion_holdout
    disjoint = validate_disjoint(train_rows, selection_rows)
    content_rows = [row for row in v3_selection if int(row["label"]) == 1]
    content_rows += imagen_holdout + laion_holdout

    train_path = args.output_root / "train.jsonl"
    selection_path = args.output_root / "eval_selection.jsonl"
    content_path = args.output_root / "eval_content_holdout.jsonl"
    write(train_path, train_rows, args.seed, "v4-train")
    write(selection_path, selection_rows, args.seed, "v4-selection")
    write(content_path, content_rows, args.seed, "v4-content-holdout")
    report = {
        "seed": args.seed,
        "forbidden_sources": [
            "organizer COCO demo-only",
            "organizer DALL-E Advanced demo-only",
            "all DiTFake COCO 0_real",
        ],
        "train_label_counts": {str(key): value for key, value in sorted(label_counts.items())},
        "train_fake_generators": group_counts(train_rows, 1, "generator"),
        "train_real_sources": group_counts(train_rows, 0, "real_source"),
        "complete_fake_holdouts": ["Imagen", "PixArt-Sigma"],
        "complete_real_holdout": "WildFake-LAION5B",
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
