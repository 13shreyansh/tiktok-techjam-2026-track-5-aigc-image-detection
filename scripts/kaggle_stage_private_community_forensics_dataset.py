#!/usr/bin/env python3
"""Stage the exact audit bundle for a private Kaggle preservation dataset."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


SOURCE = Path("/kaggle/working/community-forensics-v12-audit-preservation.zip")
EXPECTED_SHA256 = "7dc6b5e51f0fe5d4c13c9ce12e801be83cb06902fa1f2d615b9c0769871d89aa"
DESTINATION = Path("/kaggle/working/community-forensics-v12-private-dataset")
DATASET_ID = "shreyanshag13/track5-community-forensics-v12-one-shot-audit"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if DESTINATION.exists():
        raise RuntimeError(f"refusing to overwrite: {DESTINATION}")
    if not SOURCE.is_file() or sha256_file(SOURCE) != EXPECTED_SHA256:
        raise RuntimeError("audit preservation bundle is absent or changed")
    DESTINATION.mkdir()
    shutil.copy2(SOURCE, DESTINATION / SOURCE.name)
    metadata = {
        "title": "Track5 Community Forensics V12 One Shot Audit",
        "id": DATASET_ID,
        "licenses": [{"name": "other"}],
        "description": (
            "Private, audit-only preservation of exact predictions, logs and "
            "checksums. Not training data, a publication or a submission."
        ),
    }
    (DESTINATION / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "destination": str(DESTINATION),
                "dataset_id": DATASET_ID,
                "artifact_sha256": EXPECTED_SHA256,
                "public_requested": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
