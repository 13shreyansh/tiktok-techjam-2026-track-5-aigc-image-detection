#!/usr/bin/env python3
"""Create an auditable manifest view with explicitly named rows excluded."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-archive-member", action="append", default=[])
    parser.add_argument("--expected-input-rows", type=int, required=True)
    parser.add_argument("--expected-output-rows", type=int, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != args.expected_input_rows:
        raise RuntimeError(
            f"input row mismatch: {len(rows)} != {args.expected_input_rows}"
        )
    excluded = set(args.exclude_archive_member)
    observed = {str(row.get("archive_member")) for row in rows} & excluded
    missing = excluded - observed
    if missing:
        raise RuntimeError(f"requested exclusions are absent: {sorted(missing)}")
    filtered = [row for row in rows if str(row.get("archive_member")) not in excluded]
    if len(filtered) != args.expected_output_rows:
        raise RuntimeError(
            f"output row mismatch: {len(filtered)} != {args.expected_output_rows}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in filtered),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "input_rows": len(rows),
                "output_rows": len(filtered),
                "excluded_archive_members": sorted(excluded),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
