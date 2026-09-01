#!/usr/bin/env python3
"""Preserve exact Community Forensics v12 audit outputs as a private bundle."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


WORKING = Path("/kaggle/working")
OUTPUT = WORKING / "community-forensics-v12-audit-preservation.zip"
STAGE = WORKING / "community-forensics-v12-audit-preservation"
PE_ROOT = (
    WORKING
    / "community-forensics-v12/pe_core/vit_pe_core_large_patch14_336"
    / "semantic-matched-modern-v6-gate"
)
DINO_ROOT = (
    WORKING
    / "community-forensics-v12/dinov2_control"
    / "vit_large_patch14_dinov2_lvd142m/semantic-matched-modern-v6-gate"
)
EXTRA = (
    WORKING / "v12_fixed_equal_blend_community_forensics.json",
    WORKING / "community_v12_pe_core.log",
    WORKING / "community_v12_dinov2_control.log",
    WORKING / "kaggle_evaluate_community_forensics_v12.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files() -> list[tuple[Path, Path]]:
    files: list[tuple[Path, Path]] = []
    for candidate, root in (("pe_core", PE_ROOT), ("dinov2_control", DINO_ROOT)):
        if not root.is_dir():
            raise RuntimeError(f"missing candidate output root: {root}")
        for source in sorted(root.glob("*.json*")):
            files.append((source, Path(candidate) / source.name))
    for source in EXTRA:
        if not source.is_file():
            raise RuntimeError(f"missing preservation source: {source}")
        files.append((source, Path("support") / source.name))
    return files


def main() -> None:
    if STAGE.exists() or OUTPUT.exists():
        raise RuntimeError("refusing to overwrite an existing preservation artifact")
    STAGE.mkdir()
    inventory = []
    for source, relative in source_files():
        destination = STAGE / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        inventory.append(
            {
                "path": relative.as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    inventory.sort(key=lambda row: row["path"])
    metadata = {
        "status": "private_audit_preservation_only",
        "training_tuning_calibration_allowed": False,
        "organizer_demo_rows": 0,
        "files": inventory,
    }
    (STAGE / "inventory.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(STAGE))
    report = {
        "artifact": str(OUTPUT),
        "bytes": OUTPUT.stat().st_size,
        "sha256": sha256_file(OUTPUT),
        "files": len(inventory) + 1,
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
