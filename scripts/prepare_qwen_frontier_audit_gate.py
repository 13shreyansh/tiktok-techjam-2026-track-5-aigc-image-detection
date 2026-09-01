"""Balance the Qwen frontier fake audit against frozen external-gate reals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fake-manifest",
        type=Path,
        default=Path("datasets/qwen_image_bench_audit/manifest.jsonl"),
    )
    parser.add_argument(
        "--real-manifest",
        type=Path,
        default=Path("datasets/community_forensics_external_gate/manifest.jsonl"),
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=Path("datasets/family_mixture_v6/train.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/qwen_image_bench_audit/combined_gate.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    fake_rows = read_jsonl(args.fake_manifest)
    if not fake_rows or any(int(row["label"]) != 1 for row in fake_rows):
        raise RuntimeError("fake manifest must contain only generated images")
    fake_counts: dict[str, int] = defaultdict(int)
    for row in fake_rows:
        fake_counts[str(row["generator_model"])] += 1
    if len(set(fake_counts.values())) != 1:
        raise RuntimeError(f"fake generators are not balanced: {fake_counts}")

    real_rows = [
        row for row in read_jsonl(args.real_manifest) if int(row["label"]) == 0
    ]
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in real_rows:
        by_source[str(row["real_source"])].append(row)
    target_reals = len(fake_rows)
    sources = sorted(by_source)
    base, remainder = divmod(target_reals, len(sources))
    selected_reals = []
    rng = random.Random(args.seed)
    for index, source in enumerate(sources):
        count = base + int(index < remainder)
        candidates = sorted(by_source[source], key=lambda row: row["path"])
        selected_reals.extend(rng.sample(candidates, count))

    training_hashes = {
        str(row.get("image_sha256"))
        for row in read_jsonl(args.training_manifest)
        if row.get("image_sha256")
    }
    output_dir = args.output.parent.resolve()
    prepared = []
    selected_hashes = set()
    for source_manifest, rows in (
        (args.real_manifest, selected_reals),
        (args.fake_manifest, fake_rows),
    ):
        for row in rows:
            path = (source_manifest.parent / row["path"]).resolve()
            observed_sha256 = row.get("image_sha256") or sha256(path)
            if observed_sha256 in training_hashes:
                raise RuntimeError(f"audit/training SHA-256 overlap: {path}")
            if observed_sha256 in selected_hashes:
                raise RuntimeError(f"duplicate audit image SHA-256: {path}")
            selected_hashes.add(observed_sha256)
            prepared.append(
                {
                    **row,
                    "path": os.path.relpath(path, output_dir),
                    "image_sha256": observed_sha256,
                }
            )

    prepared.sort(
        key=lambda row: (
            int(row["label"]),
            str(row.get("real_source") or row.get("generator_model")),
            str(row["path"]),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prepared)
    )
    report = {
        "output": str(args.output),
        "manifest_sha256": sha256(args.output),
        "rows": len(prepared),
        "real": len(selected_reals),
        "fake": len(fake_rows),
        "real_sources": {
            source: sum(str(row["real_source"]) == source for row in selected_reals)
            for source in sources
        },
        "fake_generators": dict(sorted(fake_counts.items())),
        "training_sha256_overlap": 0,
        "within_gate_sha256_duplicates": 0,
        "seed": args.seed,
        "evaluation_policy": "force JPEG q96 then square stretch for both labels",
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
