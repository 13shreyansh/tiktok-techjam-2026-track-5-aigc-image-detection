"""Add disjoint Qwen frontier generators to v6 without changing its gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def rewrite_rows(source_manifest: Path, output_dir: Path) -> list[dict]:
    rows = []
    for row in read_jsonl(source_manifest):
        source = (source_manifest.parent / row["path"]).resolve()
        if not source.is_file():
            raise RuntimeError(f"missing source image: {source}")
        observed_sha256 = row.get("image_sha256") or sha256(source)
        rows.append(
            {
                **row,
                "path": os.path.relpath(source, output_dir),
                "image_sha256": observed_sha256,
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v6-root", type=Path, default=Path("datasets/family_mixture_v6")
    )
    parser.add_argument(
        "--frontier-train",
        type=Path,
        default=Path("datasets/qwen_image_bench_train_candidate/manifest.jsonl"),
    )
    parser.add_argument(
        "--frontier-audit",
        type=Path,
        default=Path("datasets/qwen_image_bench_audit/manifest.jsonl"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("datasets/family_mixture_v8")
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_dir = args.output_root.resolve()

    v6_train = rewrite_rows(args.v6_root / "train.jsonl", output_dir)
    frontier_train = rewrite_rows(args.frontier_train, output_dir)
    frontier_audit = read_jsonl(args.frontier_audit)
    if any(int(row["label"]) != 1 for row in frontier_train):
        raise RuntimeError("frontier candidate must contain only fake images")
    if {row.get("workflow_purpose") for row in frontier_train} != {"train-candidate"}:
        raise RuntimeError("frontier candidate purpose is not train-candidate")

    train_prompts = {int(row["prompt_id"]) for row in frontier_train}
    audit_prompts = {int(row["prompt_id"]) for row in frontier_audit}
    prompt_overlap = train_prompts & audit_prompts
    if prompt_overlap:
        raise RuntimeError(f"frontier train/audit prompt overlap: {prompt_overlap}")

    train_rows = v6_train + frontier_train
    eval_rows = rewrite_rows(args.v6_root / "eval_selection.jsonl", output_dir)
    content_rows = rewrite_rows(
        args.v6_root / "eval_content_holdout.jsonl", output_dir
    )
    train_hashes = {row["image_sha256"] for row in train_rows}
    eval_hashes = {row["image_sha256"] for row in eval_rows}
    content_hashes = {row["image_sha256"] for row in content_rows}
    if len(train_hashes) != len(train_rows):
        raise RuntimeError("duplicate image content within v8 training rows")
    overlap = train_hashes & (eval_hashes | content_hashes)
    if overlap:
        raise RuntimeError(f"v8 train/frozen-gate overlap: {len(overlap)}")
    audit_hashes = {row["image_sha256"] for row in frontier_audit}
    frontier_train_hashes = {row["image_sha256"] for row in frontier_train}
    if audit_hashes & frontier_train_hashes:
        raise RuntimeError("frontier train/audit pixel overlap")

    write_jsonl(args.output_root / "train.jsonl", train_rows)
    write_jsonl(args.output_root / "eval_selection.jsonl", eval_rows)
    write_jsonl(args.output_root / "eval_content_holdout.jsonl", content_rows)
    report = {
        "policy": "v6 plus disjoint prompt-matched Qwen frontier generators",
        "single_changed_factor": "generator training breadth",
        "v6_train_rows": len(v6_train),
        "frontier_train_rows": len(frontier_train),
        "train_rows": len(train_rows),
        "eval_selection_rows": len(eval_rows),
        "eval_content_holdout_rows": len(content_rows),
        "train_labels": dict(sorted(Counter(int(row["label"]) for row in train_rows).items())),
        "frontier_generators": dict(
            sorted(Counter(str(row["generator_model"]) for row in frontier_train).items())
        ),
        "frontier_train_prompt_ids": sorted(train_prompts),
        "frontier_audit_prompt_ids": sorted(audit_prompts),
        "frontier_train_audit_prompt_overlap": 0,
        "frontier_train_audit_sha256_overlap": 0,
        "train_frozen_gate_sha256_overlap": 0,
        "train_manifest_sha256": sha256(args.output_root / "train.jsonl"),
        "eval_selection_manifest_sha256": sha256(
            args.output_root / "eval_selection.jsonl"
        ),
        "eval_content_holdout_manifest_sha256": sha256(
            args.output_root / "eval_content_holdout.jsonl"
        ),
    }
    (args.output_root / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
