#!/usr/bin/env python3
"""Build a deterministic source-balanced subset for full transform evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank(seed: int, row: dict) -> str:
    identity = str(
        row.get("image_sha256")
        or row.get("archive_member")
        or row.get("path")
    )
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fake-per-generator", type=int, default=192)
    parser.add_argument("--real-per-source", type=int, default=307)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        label = int(row["label"])
        group = str(
            row.get("generator", "unknown")
            if label == 1
            else row.get("real_source", "unknown")
        )
        groups[(label, group)].append(row)

    selected: list[dict] = []
    for (label, group), candidates in sorted(groups.items()):
        limit = args.fake_per_generator if label == 1 else args.real_per_source
        if len(candidates) < limit:
            raise RuntimeError(
                f"insufficient rows for label={label}, group={group}: "
                f"{len(candidates)} < {limit}"
            )
        selected.extend(sorted(candidates, key=lambda row: rank(args.seed, row))[:limit])

    selected.sort(
        key=lambda row: (
            int(row["label"]),
            str(row.get("real_source") or row.get("generator")),
            str(row["path"]),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    group_counts = Counter(
        (
            int(row["label"]),
            str(row.get("generator") if int(row["label"]) == 1 else row.get("real_source")),
        )
        for row in selected
    )
    report = {
        "input": str(args.input),
        "input_sha256": file_sha256(args.input),
        "output": str(args.output),
        "output_sha256": file_sha256(args.output),
        "seed": args.seed,
        "selection": "lowest SHA-256 rank of seed plus immutable row identity",
        "fake_per_generator": args.fake_per_generator,
        "real_per_source": args.real_per_source,
        "input_rows": len(rows),
        "output_rows": len(selected),
        "group_counts": {
            f'{"fake" if label else "real"}:{group}': count
            for (label, group), count in sorted(group_counts.items())
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
