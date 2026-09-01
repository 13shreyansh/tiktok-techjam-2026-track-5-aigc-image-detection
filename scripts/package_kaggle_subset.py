#!/usr/bin/env python3
"""Package an exact manifest-selected dataset for an ephemeral GPU runtime.

The package contains only files referenced by the supplied manifests. Images
are content-addressed so rows shared by multiple evaluation manifests are
stored once. The command refuses known demo-only paths and writes a checksum
sidecar next to the ignored ZIP artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path


FORBIDDEN_PATH_PARTS = {"demo_only", "demo_only_DO_NOT_TRAIN"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            source = Path(row["path"])
            if not source.is_absolute():
                source = path.parent / source
            source = source.resolve()
            if not source.is_file():
                raise SystemExit(f"missing source at {path}:{line_number}: {source}")
            if FORBIDDEN_PATH_PARTS.intersection(source.parts):
                raise SystemExit(f"refusing forbidden demo-only source: {source}")
            rows.append({**row, "_source": source})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise SystemExit(f"refusing to overwrite package or partial file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")

    manifests = [(path.resolve(), manifest_rows(path.resolve())) for path in args.manifest]
    sources = sorted({row["_source"] for _, rows in manifests for row in rows})
    source_records = {}
    content_records = {}
    inventory = hashlib.sha256()
    total_source_bytes = 0

    try:
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED) as archive:
            for index, source in enumerate(sources, start=1):
                digest = sha256_file(source)
                size = source.stat().st_size
                suffix = source.suffix.lower()
                if digest not in content_records:
                    archive_path = f"images/{digest[:2]}/{digest}{suffix}"
                    record = {
                        "sha256": digest,
                        "bytes": size,
                        "archive_path": archive_path,
                    }
                    archive.write(source, archive_path)
                    content_records[digest] = record
                    total_source_bytes += size
                    inventory.update(f"{digest}\t{size}\t{archive_path}\n".encode())
                elif content_records[digest]["bytes"] != size:
                    raise RuntimeError(f"SHA-256 collision with unequal sizes: {source}")
                source_records[source] = content_records[digest]
                if index % 500 == 0 or index == len(sources):
                    print(
                        f"scanned {index}/{len(sources)} source paths; "
                        f"packaged {len(content_records)} unique images",
                        flush=True,
                    )

            manifest_reports = []
            used_names = set()
            for manifest, rows in manifests:
                name = manifest.name
                if name in used_names:
                    name = f"{manifest.parent.name}-{name}"
                used_names.add(name)
                rewritten = []
                for row in rows:
                    record = source_records[row["_source"]]
                    rewritten.append(
                        {
                            **{key: value for key, value in row.items() if key != "_source"},
                            "path": os.path.relpath(record["archive_path"], "manifests"),
                            "image_sha256": record["sha256"],
                        }
                    )
                data = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rewritten).encode()
                archive.writestr(f"manifests/{name}", data)
                manifest_reports.append(
                    {
                        # Packages may leave the local machine.  Preserve a
                        # stable human-readable identifier without exposing an
                        # absolute workstation path or username.
                        "source_manifest_name": manifest.name,
                        "packaged_manifest": f"manifests/{name}",
                        "rows": len(rows),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )

            embedded_report = {
                "format_version": 2,
                "compression": "ZIP_STORED",
                "source_paths": len(sources),
                "unique_images": len(content_records),
                "source_bytes": total_source_bytes,
                "inventory_sha256": inventory.hexdigest(),
                "forbidden_path_parts": sorted(FORBIDDEN_PATH_PARTS),
                "manifests": manifest_reports,
            }
            archive.writestr("package.json", json.dumps(embedded_report, indent=2) + "\n")

        partial.replace(output)
        report = {
            **embedded_report,
            "package": str(output),
            "package_bytes": output.stat().st_size,
            "package_sha256": sha256_file(output),
        }
        output.with_suffix(output.suffix + ".provenance.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
    except BaseException:
        # Keep a partial artifact visible for diagnosis; it cannot be mistaken
        # for the requested output because it has a distinct suffix.
        raise


if __name__ == "__main__":
    main()
