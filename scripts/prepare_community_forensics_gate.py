#!/usr/bin/env python3
"""Build a balanced external gate from Community Forensics fakes and v6 reals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


SEED = 20260829
FORBIDDEN_PATH_TERMS = ("coco", "dall-e", "dalle")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def rank(row: dict) -> str:
    key = str(row.get("image_sha256") or row["path"])
    return hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fake-manifest",
        type=Path,
        default=Path("datasets/community_forensics_small_shard0_fakes/manifest.jsonl"),
    )
    parser.add_argument(
        "--real-manifest",
        type=Path,
        default=Path("datasets/family_mixture_v6/eval_robustness_balanced.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/community_forensics_external_gate/manifest.jsonl"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    fake_rows = read_rows(args.fake_manifest)
    if len(fake_rows) != 312 or any(int(row["label"]) != 1 for row in fake_rows):
        raise SystemExit("expected exactly 312 synthetic-only Community Forensics rows")
    if len({str(row["generator_model"]) for row in fake_rows}) != 78:
        raise SystemExit("expected exactly 78 Community Forensics model variants")

    grouped_reals: dict[str, list[dict]] = defaultdict(list)
    for row in read_rows(args.real_manifest):
        if int(row["label"]) == 0:
            grouped_reals[str(row["real_source"])].append(row)
    sources = sorted(grouped_reals)
    if len(sources) != 5:
        raise SystemExit(f"expected five real sources, observed {sources}")

    base, remainder = divmod(len(fake_rows), len(sources))
    selected_reals = []
    for index, source in enumerate(sources):
        limit = base + (1 if index < remainder else 0)
        candidates = sorted(grouped_reals[source], key=rank)
        if len(candidates) < limit:
            raise SystemExit(f"insufficient rows for {source}: {len(candidates)} < {limit}")
        selected_reals.extend(candidates[:limit])

    output_parent = args.output.parent.resolve()
    output_rows = []
    for row in fake_rows:
        source_path = (args.fake_manifest.parent / row["path"]).resolve()
        normalized = dict(row)
        normalized["path"] = os.path.relpath(source_path, output_parent)
        normalized["generator"] = "CommunityForensics-LatDiff-78-models"
        output_rows.append(normalized)
    for row in selected_reals:
        source_path = (args.real_manifest.parent / row["path"]).resolve()
        normalized = dict(row)
        normalized["path"] = os.path.relpath(source_path, output_parent)
        output_rows.append(normalized)

    hashes = [str(row.get("image_sha256", "")) for row in output_rows]
    nonempty_hashes = [value for value in hashes if value]
    if len(nonempty_hashes) != len(set(nonempty_hashes)):
        raise SystemExit("content hash overlap inside external gate")
    for row in output_rows:
        path = (output_parent / row["path"]).resolve()
        if not path.is_file():
            raise SystemExit(f"missing gate image: {path}")
        lowered = str(path).casefold()
        if any(term in lowered for term in FORBIDDEN_PATH_TERMS):
            raise SystemExit(f"forbidden path term in gate: {path}")

    output_rows.sort(
        key=lambda row: (
            int(row["label"]),
            str(row.get("real_source") or row.get("generator_model")),
            str(row["path"]),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows)
    )
    report = {
        "seed": SEED,
        "fake_manifest": str(args.fake_manifest),
        "real_manifest": str(args.real_manifest),
        "rows": len(output_rows),
        "labels": dict(sorted(Counter(str(row["label"]) for row in output_rows).items())),
        "real_sources": dict(
            sorted(
                Counter(
                    str(row["real_source"])
                    for row in output_rows
                    if int(row["label"]) == 0
                ).items()
            )
        ),
        "fake_models": len(
            {str(row["generator_model"]) for row in output_rows if int(row["label"]) == 1}
        ),
        "manifest_sha256": sha256(args.output),
        "training_allowed": False,
        "purpose": "external synthetic-model breadth audit only",
    }
    report_path = args.output.with_name("report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
