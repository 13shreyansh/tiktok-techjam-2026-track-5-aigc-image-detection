#!/usr/bin/env python3
"""Build an audit manifest exactly matched on image container and geometry."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def load_rows(manifest: Path) -> list[dict]:
    result = []
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        path = Path(str(row["path"]))
        if not path.is_absolute():
            path = (manifest.parent / path).resolve()
        with Image.open(path) as image:
            key = (
                str(image.format or "unknown"),
                str(image.mode),
                int(image.width),
                int(image.height),
            )
        result.append({**row, "path": str(path), "metadata_stratum": key})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    groups: dict[tuple, dict[int, list[dict]]] = defaultdict(
        lambda: {0: [], 1: []}
    )
    for row in load_rows(args.manifest):
        groups[tuple(row["metadata_stratum"])][int(row["label"])].append(row)

    rng = random.Random(args.seed)
    selected = []
    strata_report = []
    for stratum, by_label in sorted(groups.items()):
        count = min(len(by_label[0]), len(by_label[1]))
        if count == 0:
            continue
        for label in (0, 1):
            pool = list(by_label[label])
            rng.shuffle(pool)
            selected.extend(pool[:count])
        strata_report.append(
            {
                "format": stratum[0],
                "mode": stratum[1],
                "width": stratum[2],
                "height": stratum[3],
                "selected_per_label": count,
                "available": {
                    "real": len(by_label[0]),
                    "fake": len(by_label[1]),
                },
            }
        )
    rng.shuffle(selected)
    if not selected:
        raise SystemExit("no metadata stratum contains both labels")

    output_rows = []
    for row in selected:
        stratum = row.pop("metadata_stratum")
        output_rows.append(
            {
                **row,
                "audit_metadata_stratum": {
                    "format": stratum[0],
                    "mode": stratum[1],
                    "width": stratum[2],
                    "height": stratum[3],
                },
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    label_counts = Counter(int(row["label"]) for row in output_rows)
    report = {
        "status": "audit_only_not_a_training_manifest",
        "source_manifest": str(args.manifest.resolve()),
        "output_manifest": str(args.output.resolve()),
        "seed": args.seed,
        "exact_match_features": ["format", "mode", "width", "height"],
        "selected_rows": len(output_rows),
        "label_counts": {str(key): value for key, value in sorted(label_counts.items())},
        "strata": strata_report,
        "interpretation_boundary": (
            "This gate removes exact container/geometry differences inside each "
            "retained stratum. It does not remove dataset, semantic, or generator shortcuts."
        ),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
