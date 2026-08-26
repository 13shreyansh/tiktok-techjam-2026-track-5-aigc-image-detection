#!/usr/bin/env python3
"""Verify pinned/canonical public repository inventories without downloading data."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "resources/resource_manifest.json"


def fetch(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "track5-preparation/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def canonical_digest(rows: list[dict]) -> str:
    payload = json.dumps(
        sorted(rows, key=lambda item: item["path"]),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    lock = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    resources = {item["id"]: item for item in lock["resources"]}

    sid = resources["sid_set_huggingface"]
    sid_url = (
        "https://huggingface.co/api/datasets/saberzl/SID_Set/revision/"
        f"{sid['revision']}?blobs=true"
    )
    sid_data = fetch(sid_url)
    sid_rows = [
        {
            "path": item["rfilename"],
            "size": item.get("size"),
            "oid": (item.get("lfs") or {}).get("sha256") or item.get("blobId"),
        }
        for item in sid_data["siblings"]
    ]
    if sid_data["sha"] != sid["revision"]:
        raise SystemExit("SID_Set revision mismatch")
    if sum((item["size"] or 0) for item in sid_rows) != sid["expected_bytes"]:
        raise SystemExit("SID_Set byte count mismatch")
    if canonical_digest(sid_rows) != sid["canonical_inventory_sha256"]:
        raise SystemExit("SID_Set canonical inventory mismatch")

    wild = resources["wildfake_repository"]
    wild_data = fetch(wild["metadata_url"])["Data"]["Files"]
    wild_rows = [
        {
            "path": item["Path"],
            "size": item["Size"],
            "sha256": item["Sha256"],
            "revision": item["Revision"],
        }
        for item in wild_data
        if item["Type"] != "tree"
    ]
    if sum(item["size"] for item in wild_rows) != wild["enumerated_file_bytes"]:
        raise SystemExit("WildFake byte count mismatch")
    if canonical_digest(wild_rows) != wild["canonical_inventory_sha256"]:
        raise SystemExit("WildFake canonical inventory mismatch")

    print(
        f"SID_Set remote inventory verified: {len(sid_rows)} files, "
        f"{sid['expected_bytes']} bytes"
    )
    print(
        f"WildFake remote inventory verified: {len(wild_rows)} files, "
        f"{wild['enumerated_file_bytes']} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
