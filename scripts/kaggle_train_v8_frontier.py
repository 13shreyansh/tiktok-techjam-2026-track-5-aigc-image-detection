#!/usr/bin/env python3
"""Train PE-Core-L on v6 plus 18 disjoint 2026 frontier generators.

The existing 9 GB v6 Kaggle input is reused.  A separate 613 MB frontier-only
package is checksum-verified and merged through an absolute-path manifest at
runtime.  The frozen v6 selection and content gates are not repackaged or
changed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import kaggle_train_v3 as runner
import kaggle_train_v6  # noqa: F401  # checksum-pinned v6 package constants


FRONTIER_PACKAGE_NAME = "qwen-frontier-train-v8.zip"
FRONTIER_PACKAGE_SHA256 = (
    "650f309ddb4fd8d0b7ac05d101981460319b9de2f39fd7795b1413982145fa93"
)
FRONTIER_INVENTORY_SHA256 = (
    "f7f2036aec27d64d855253fded6de4b3996838f085b18c84912c0d808b5725a0"
)
FRONTIER_MANIFEST_SHA256 = (
    "a5c5ad4d1a631be90b31fc918d310d6f1f163bffeb971bf7e0ec9067c9b35e5d"
)
FRONTIER_ROWS = 576
FRONTIER_MODELS = 18
EXPECTED_TRAIN_ROWS = 19_534
OUTPUT_ROOT = Path("/kaggle/working/track5-v8-frontier-candidate")
FRONTIER_WORK_ROOT = Path("/kaggle/working/qwen-frontier-train-v8")
EXCLUDED_EVAL_SHA256 = {
    "ff3e8968b04eaa4d95b3d9b8bd88e61a673bf0589a7be761cd71470559cd3ab1",
}


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root not in target.parents and target != root:
            raise RuntimeError(f"unsafe frontier archive member: {member.filename}")
    archive.extractall(destination)


def find_frontier_root() -> tuple[Path, bool]:
    archives = list(Path("/kaggle/input").rglob(FRONTIER_PACKAGE_NAME))
    if len(archives) == 1:
        if runner.file_sha256(archives[0]) != FRONTIER_PACKAGE_SHA256:
            raise RuntimeError("frontier package SHA-256 mismatch")
        if FRONTIER_WORK_ROOT.exists():
            shutil.rmtree(FRONTIER_WORK_ROOT)
        FRONTIER_WORK_ROOT.mkdir(parents=True)
        with zipfile.ZipFile(archives[0]) as archive:
            safe_extract(archive, FRONTIER_WORK_ROOT)
        return FRONTIER_WORK_ROOT, True
    if len(archives) > 1:
        raise RuntimeError(f"multiple frontier archives: {archives}")

    candidates = []
    for report_path in Path("/kaggle/input").rglob("package.json"):
        try:
            report = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if report.get("inventory_sha256") == FRONTIER_INVENTORY_SHA256:
            candidates.append(report_path.parent)
    if len(candidates) != 1:
        raise RuntimeError(f"expected one expanded frontier package: {candidates}")
    return candidates[0], False


def verified_frontier_rows() -> tuple[Path, list[dict], dict]:
    root, zip_transport_verified = find_frontier_root()
    package = json.loads((root / "package.json").read_text())
    if package.get("inventory_sha256") != FRONTIER_INVENTORY_SHA256:
        raise RuntimeError("frontier inventory mismatch")
    manifest = root / "manifests/manifest.jsonl"
    if runner.file_sha256(manifest) != FRONTIER_MANIFEST_SHA256:
        raise RuntimeError("frontier manifest mismatch")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    models = Counter(str(row["generator_model"]) for row in rows)
    if (
        len(rows) != FRONTIER_ROWS
        or len(models) != FRONTIER_MODELS
        or set(models.values()) != {32}
        or any(int(row["label"]) != 1 for row in rows)
    ):
        raise RuntimeError(f"unexpected frontier composition: {len(rows)}, {models}")
    for index, row in enumerate(rows, 1):
        image = (manifest.parent / row["path"]).resolve()
        if runner.file_sha256(image) != row["image_sha256"]:
            raise RuntimeError(f"frontier image mismatch: {image}")
        if index % 100 == 0 or index == len(rows):
            print(json.dumps({"verified_frontier_images": index}), flush=True)
    return manifest, rows, {
        "package_sha256": FRONTIER_PACKAGE_SHA256,
        "inventory_sha256": FRONTIER_INVENTORY_SHA256,
        "manifest_sha256": FRONTIER_MANIFEST_SHA256,
        "zip_transport_verified": zip_transport_verified,
        "models": dict(sorted(models.items())),
    }


def absolute_rows(manifest: Path, rows: list[dict]) -> list[dict]:
    return [
        {**row, "path": str((manifest.parent / row["path"]).resolve())}
        for row in rows
    ]


def combined_digest(*values: str) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def main() -> None:
    runner.OUTPUT_ROOT = OUTPUT_ROOT
    runner.MODEL_NAMES = ("vit_pe_core_large_patch14_336",)
    runner.EXCLUDED_EVAL_SHA256 = EXCLUDED_EVAL_SHA256
    runner.AUGMENTATION_DESCRIPTION = (
        "v6_at_most_one_workshop_transform_plus_18_frontier_generator_groups"
    )
    v6_package_sha256, v6_package = runner.validate_package()
    v6_train = runner.WORK_ROOT / "manifests/train.jsonl"
    eval_manifest = runner.WORK_ROOT / "manifests/eval_selection.jsonl"
    content_manifest = runner.WORK_ROOT / "manifests/eval_content_holdout.jsonl"
    v6_rows = [json.loads(line) for line in v6_train.read_text().splitlines() if line]
    frontier_manifest, frontier_rows, frontier_provenance = verified_frontier_rows()
    combined_rows = absolute_rows(v6_train, v6_rows) + absolute_rows(
        frontier_manifest, frontier_rows
    )
    if len(combined_rows) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(f"unexpected combined rows: {len(combined_rows)}")
    train_hashes = {row["image_sha256"] for row in combined_rows}
    if len(train_hashes) != len(combined_rows):
        raise RuntimeError("duplicate content in combined v8 training manifest")
    eval_rows = [json.loads(line) for line in eval_manifest.read_text().splitlines() if line]
    content_rows = [
        json.loads(line) for line in content_manifest.read_text().splitlines() if line
    ]
    frozen_hashes = {
        row["image_sha256"] for row in eval_rows + content_rows if row.get("image_sha256")
    }
    if train_hashes & frozen_hashes:
        raise RuntimeError("v8 training content overlaps a frozen v6 gate")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    combined_manifest = OUTPUT_ROOT / "train-v8-absolute.jsonl"
    combined_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in combined_rows)
    )
    provenance = {
        "policy": "v6 plus 18 Qwen Image Bench frontier generators",
        "single_changed_factor": "generator training breadth",
        "v6": {
            "package_sha256": v6_package_sha256,
            "inventory_sha256": v6_package["inventory_sha256"],
            "rows": len(v6_rows),
        },
        "frontier": frontier_provenance,
        "combined_rows": len(combined_rows),
        "combined_manifest_sha256": runner.file_sha256(combined_manifest),
        "train_frozen_gate_sha256_overlap": 0,
    }
    (OUTPUT_ROOT / "input-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance, indent=2), flush=True)

    combined_package_sha256 = combined_digest(
        v6_package_sha256, FRONTIER_PACKAGE_SHA256
    )
    combined_inventory_sha256 = combined_digest(
        v6_package["inventory_sha256"], FRONTIER_INVENTORY_SHA256
    )
    report = runner.train_candidate(
        runner.MODEL_NAMES[0],
        combined_manifest,
        eval_manifest,
        content_manifest,
        combined_package_sha256,
        combined_inventory_sha256,
    )
    summary = {
        "model": report["model"],
        "clean_auc": report["selection_clean"]["clean_auc"],
        "worst_fake_generator_auc": report["selection_clean"]["groups"][
            "worst_fake_generator_auc"
        ],
        "worst_real_source_auc": report["selection_clean"]["groups"][
            "worst_real_source_auc"
        ],
        "worst_pair_auc": report["selection_clean"]["groups"][
            "worst_generator_real_source_pair_auc"
        ],
        "content_holdout_auc": report["content_holdout_clean"]["clean_auc"],
        "checkpoint_sha256": runner.file_sha256(
            OUTPUT_ROOT / runner.MODEL_NAMES[0] / "model.pt"
        ),
    }
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
