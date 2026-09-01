#!/usr/bin/env python3
"""Remove every v12 source/canonical identity from the frozen external gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "2d770ff99f781320a10a9a15fa03de79d2cab40929b09fb1b4db7e759848398c"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def derive(gate_rows: list[dict], v12_rows: list[dict]) -> tuple[list[dict], dict]:
    v12_identities = {
        str(value)
        for row in v12_rows
        for value in (row.get("image_sha256"), row.get("source_image_sha256"))
        if value
    }
    retained = [
        row for row in gate_rows if str(row.get("image_sha256")) not in v12_identities
    ]
    excluded = [
        row for row in gate_rows if str(row.get("image_sha256")) in v12_identities
    ]
    retained_hashes = {str(row["image_sha256"]) for row in retained}
    if retained_hashes & v12_identities:
        raise RuntimeError("derived Community Forensics gate still overlaps v12")
    return retained, {
        "source_rows": len(gate_rows),
        "retained_rows": len(retained),
        "retained_labels": dict(sorted(Counter(int(row["label"]) for row in retained).items())),
        "retained_fake_models": len(
            {
                str(row["generator_model"])
                for row in retained
                if int(row["label"]) == 1
            }
        ),
        "excluded_rows": len(excluded),
        "excluded_labels": dict(sorted(Counter(int(row["label"]) for row in excluded).items())),
        "excluded_real_sources": dict(
            sorted(
                Counter(
                    str(row.get("real_source"))
                    for row in excluded
                    if int(row["label"]) == 0
                ).items()
            )
        ),
        "excluded_image_sha256": sorted(str(row["image_sha256"]) for row in excluded),
        "v12_identity_overlap_after_filter": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate",
        type=Path,
        default=Path("datasets/community_forensics_external_gate/manifest.jsonl"),
    )
    parser.add_argument(
        "--v12-train",
        type=Path,
        default=Path("datasets/permissive_mixture_v12_canonical/train.jsonl"),
    )
    parser.add_argument(
        "--v12-eval",
        type=Path,
        default=Path("datasets/permissive_mixture_v12_canonical/eval_frozen.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/community_forensics_v12_disjoint/manifest.jsonl"),
    )
    args = parser.parse_args()
    if sha256(args.gate) != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise RuntimeError("source Community Forensics manifest checksum mismatch")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    gate_rows = rows(args.gate)
    v12_train = rows(args.v12_train)
    v12_eval = rows(args.v12_eval)
    retained, report = derive(gate_rows, v12_train + v12_eval)
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in retained)
    )
    report.update(
        {
            "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
            "derived_manifest_sha256": sha256(args.output),
            "v12_train_rows_compared": len(v12_train),
            "v12_eval_rows_compared": len(v12_eval),
            "training_allowed": False,
            "organizer_demo_rows": 0,
            "purpose": "v12-disjoint external audit only",
        }
    )
    report_path = args.output.with_name("report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
