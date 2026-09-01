#!/usr/bin/env python3
"""Summarize image-container cues that may leak labels or dataset sources."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "max": ordered[-1],
    }


def summarize(rows: list[dict]) -> dict:
    widths = [row["width"] for row in rows]
    heights = [row["height"] for row in rows]
    areas = [row["width"] * row["height"] for row in rows]
    ratios = [row["width"] / row["height"] for row in rows]
    sizes = [row["bytes"] for row in rows]
    dimensions = Counter(f'{row["width"]}x{row["height"]}' for row in rows)
    return {
        "count": len(rows),
        "suffixes": dict(sorted(Counter(row["suffix"] for row in rows).items())),
        "formats": dict(sorted(Counter(row["format"] for row in rows).items())),
        "modes": dict(sorted(Counter(row["mode"] for row in rows).items())),
        "width": quantiles(widths),
        "height": quantiles(heights),
        "pixel_area": quantiles(areas),
        "aspect_ratio": quantiles(ratios),
        "file_bytes": quantiles(sizes),
        "top_dimensions": dimensions.most_common(12),
    }


def audit(manifest: Path) -> dict:
    inspected: list[dict] = []
    with manifest.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            path = Path(row["path"])
            if not path.is_absolute():
                path = (manifest.parent / path).resolve()
            with Image.open(path) as image:
                width, height = image.size
                inspected.append(
                    {
                        **row,
                        "path": str(path),
                        "width": width,
                        "height": height,
                        "format": image.format or "unknown",
                        "mode": image.mode,
                        "suffix": path.suffix.lower(),
                        "bytes": path.stat().st_size,
                    }
                )

    by_label: dict[str, list[dict]] = defaultdict(list)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in inspected:
        label = int(row["label"])
        by_label[str(label)].append(row)
        source = row.get("generator") if label == 1 else row.get("real_source")
        by_source[f'{"fake" if label else "real"}:{source or "unknown"}'].append(row)

    return {
        "manifest": str(manifest.resolve()),
        "rows": len(inspected),
        "warning": (
            "Large differences between labels or sources are possible shortcuts. "
            "This report diagnoses container and size cues; it does not prove causation."
        ),
        "by_label": {key: summarize(rows) for key, rows in sorted(by_label.items())},
        "by_source": {key: summarize(rows) for key, rows in sorted(by_source.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": report["rows"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
