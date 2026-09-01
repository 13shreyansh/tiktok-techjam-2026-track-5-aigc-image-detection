#!/usr/bin/env python3
"""Run the frozen v12 candidates on the predeclared Open Images source rotation.

This deliberately reuses the same modern fake identities as the consumed COCO
gate and changes only the real collection. It is a diagnostic, never a fresh
promotion, calibration, weighting or model-selection gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import kaggle_evaluate_semantic_modern_v12_gate as base


PACKAGE_NAME = "semantic-matched-modern-openimages-v1.zip"
PACKAGE_SHA256 = "c9862550a1476e60b13e9c262e751d3d75e8c582a71d60496c064185b63e4906"
INVENTORY_SHA256 = "ed12e13391e84770aa3296eb1db1e13d97e74fba1d65fca480bd77cac2250382"
MANIFEST_SHA256 = "a696d1a781dacaa66183fb96e6a4078ebdf3dd429dedb657524ddf846b3667d6"
WORK_ROOT = Path("/kaggle/working/semantic-matched-modern-openimages-v1")
STAGING_ROOT = Path("/kaggle/working/semantic-openimages-gate-package")
AUDIT_MODEL_ROOT = Path("/kaggle/working/openimages-source-rotation-v12")
GATE_MOUNT_SEARCH_ROOT = Path("/kaggle/input/datasets")
GATE_MOUNT_SLUG = "track5-openimages-source-rotation-v12-audit"
ORIGINAL_VALIDATE_ROWS = base.validate_gate_rows


def validate_mounted_gate() -> tuple[Path, dict]:
    """Validate Kaggle's automatic extraction of the exact frozen package.

    Kaggle expands uploaded ZIPs when creating a dataset, so the container bytes
    are unavailable at runtime. Recompute the embedded file inventory from every
    content-addressed image and verify the packaged manifest instead.
    """

    candidates = []
    pattern = f"*/{GATE_MOUNT_SLUG}/**/package.json"
    for package_path in GATE_MOUNT_SEARCH_ROOT.glob(pattern):
        metadata = json.loads(package_path.read_text())
        if metadata.get("inventory_sha256") == INVENTORY_SHA256:
            candidates.append((package_path.parent, metadata))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one mounted Open Images gate, found {candidates}")
    root, metadata = candidates[0]
    manifest = root / "manifests/eval_semantic_matched.jsonl"
    if base.runner.file_sha256(manifest) != MANIFEST_SHA256:
        raise RuntimeError("mounted Open Images manifest checksum mismatch")
    images = sorted(path for path in (root / "images").rglob("*") if path.is_file())
    if len(images) != int(metadata.get("unique_images", -1)):
        raise RuntimeError(f"mounted Open Images image count mismatch: {len(images)}")
    inventory = hashlib.sha256()
    for path in images:
        digest = base.runner.file_sha256(path)
        if path.stem != digest:
            raise RuntimeError(f"mounted Open Images content-address mismatch: {path}")
        archive_path = path.relative_to(root).as_posix()
        inventory.update(f"{digest}\t{path.stat().st_size}\t{archive_path}\n".encode())
    observed = inventory.hexdigest()
    if observed != INVENTORY_SHA256:
        raise RuntimeError(f"mounted Open Images inventory mismatch: {observed}")
    return root, {**metadata, "container_sha256": PACKAGE_SHA256, "mount_validation": "file_inventory"}


def validate_openimages_rows(rows: list[dict]) -> dict:
    report = ORIGINAL_VALIDATE_ROWS(rows)
    for index, row in enumerate(rows):
        if int(row["label"]) == 0:
            if row.get("dataset") != "Open-Images-V7-validation-CVDF" or row.get(
                "real_source"
            ) != "Open-Images-V7-validation-CVDF":
                raise RuntimeError(f"row {index}: Open Images real-source mismatch")
        elif row.get("dataset") == "Open-Images-V7-validation-CVDF":
            raise RuntimeError(f"row {index}: fake row carries Open Images source")
    return {**report, "real_source": "Open-Images-V7-validation-CVDF"}


def materialize_candidate_views() -> None:
    """Create separate hard-linked roots so consumed COCO outputs stay immutable."""

    for candidate, specification in base.CANDIDATES.items():
        model_dir = specification["model"].replace(".", "_")
        source = specification["root"] / model_dir
        target_parent = AUDIT_MODEL_ROOT / candidate
        target = target_parent / model_dir
        target.mkdir(parents=True, exist_ok=True)
        for name in ("model.pt", "report.json"):
            source_file = source / name
            target_file = target / name
            if not source_file.is_file():
                raise RuntimeError(f"missing frozen source artifact: {source_file}")
            if not target_file.exists():
                try:
                    os.link(source_file, target_file)
                except OSError:
                    shutil.copy2(source_file, target_file)
        observed = base.runner.file_sha256(target / "model.pt")
        if observed != specification["checkpoint_sha256"]:
            raise RuntimeError(f"candidate view hash mismatch for {candidate}: {observed}")
        specification["root"] = target_parent


def configure() -> None:
    base.GATE_PACKAGE_NAME = PACKAGE_NAME
    base.GATE_PACKAGE_SHA256 = PACKAGE_SHA256
    base.GATE_INVENTORY_SHA256 = INVENTORY_SHA256
    base.GATE_MANIFEST_SHA256 = MANIFEST_SHA256
    base.GATE_WORK_ROOT = WORK_ROOT
    base.GATE_STAGING_ROOT = STAGING_ROOT
    base.validate_gate_package = validate_mounted_gate
    base.validate_gate_rows = validate_openimages_rows
    materialize_candidate_views()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(base.CANDIDATES), required=True)
    args = parser.parse_args()
    configure()
    result = base.evaluate_candidate(args.candidate)
    result["decision_boundary"] = (
        "Post-score source-rotation diagnostic only; reused fake identities and may not "
        "train, tune, calibrate, reweight, select or promote a candidate."
    )
    base.workshop.atomic_json(
        base.CANDIDATES[args.candidate]["root"]
        / base.CANDIDATES[args.candidate]["model"].replace(".", "_")
        / "semantic-matched-modern-v6-gate"
        / "progress.json",
        result,
    )


if __name__ == "__main__":
    main()
