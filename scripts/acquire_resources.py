#!/usr/bin/env python3
"""Download explicitly selected public artifacts and verify locked metadata.

This utility does not choose data, train a model, or transform images. Large
downloads require an additional acknowledgement flag and always land under the
ignored datasets/ tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "resources" / "resource_manifest.json"
LARGE_THRESHOLD = 2_000_000_000


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_resources() -> dict[str, dict]:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {item["id"]: item for item in data["resources"]}


def verify(item: dict) -> tuple[bool, str]:
    destination = ROOT / item["local_path"]
    if not destination.is_file():
        return False, f"missing: {destination.relative_to(ROOT)}"
    expected_bytes = item.get("expected_bytes")
    if expected_bytes is not None and destination.stat().st_size != expected_bytes:
        return False, f"size mismatch: {destination.stat().st_size} != {expected_bytes}"
    for algorithm, field in (("sha256", "expected_sha256"), ("md5", "expected_md5")):
        expected = item.get(field)
        if expected and digest(destination, algorithm) != expected:
            return False, f"{algorithm} mismatch"
    return True, f"verified: {destination.relative_to(ROOT)}"


def download(item: dict, allow_large: bool) -> None:
    source = item.get("download_url")
    if not source or not item.get("local_path"):
        raise SystemExit(f"{item['id']} is manifest-only and has no direct artifact URL")
    expected_bytes = item.get("expected_bytes", 0)
    if expected_bytes >= LARGE_THRESHOLD and not allow_large:
        raise SystemExit(
            f"refusing {expected_bytes:,}-byte download without --allow-large-download"
        )
    destination = ROOT / item["local_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(source, headers={"User-Agent": "track5-preparation/1"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    partial.replace(destination)
    ok, message = verify(item)
    if not ok:
        raise SystemExit(message)
    print(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("list", "download", "verify"))
    parser.add_argument("resource", nargs="?")
    parser.add_argument("--allow-large-download", action="store_true")
    args = parser.parse_args()
    resources = load_resources()

    if args.action == "list":
        for resource_id, item in resources.items():
            print(
                f"{resource_id}\t{item['acquisition_state']}\t"
                f"{item.get('expected_bytes', 'n/a')} bytes"
            )
        return 0

    if not args.resource or args.resource not in resources:
        parser.error("download/verify requires a resource id from the list action")
    item = resources[args.resource]
    if args.action == "download":
        download(item, args.allow_large_download)
        return 0
    ok, message = verify(item)
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
