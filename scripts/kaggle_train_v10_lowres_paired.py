#!/usr/bin/env python3
"""Train one matched low-resolution block ablation from the v9 data recipe.

V10 changes one intervention from v9: it adds a checksum-frozen, balanced
CIFAKE train-only supplement and reserves 25% of each label's sampling mass
for the combined low-resolution pair.  Model, frozen-backbone policy, seed,
optimizer, one-epoch budget, frontier cap and one-transform augmentation remain
unchanged.  Organizer demo resources are never read.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import WeightedRandomSampler

import kaggle_train_v3 as runner
import kaggle_train_v6  # noqa: F401  # checksum-pinned v6 package constants
import kaggle_train_v8_frontier as v8


REPAIR_PACKAGE_NAME = "cifake-lowres-repair-v10.zip"
REPAIR_PACKAGE_SHA256 = "3643b283fae9793e05ea0ef80cabc84d34dfd9128f794e73d39e46b803274f8d"
REPAIR_INVENTORY_SHA256 = "201b27679d10be78a25766863c5406822b12557ea269ec870ff9cbaca9c8d3ac"
REPAIR_MANIFEST_SHA256 = "9ef468f4b34e472d3e46fdebf58c2659a7b1cf0991720a1c6739f475bf529f6f"
REPAIR_ROWS = 10_000
REPAIR_ROWS_PER_LABEL = 5_000
EXPECTED_TRAIN_ROWS = 29_534
OUTPUT_ROOT = Path("/kaggle/working/track5-v10-lowres-paired-candidate")
REPAIR_WORK_ROOT = Path("/kaggle/working/cifake-lowres-repair-v10")
LOWRES_MASS_WITHIN_LABEL = 0.25
FRONTIER_MASS_WITHIN_FAKE = 0.15
OTHER_REAL_MASS_WITHIN_LABEL = 1.0 - LOWRES_MASS_WITHIN_LABEL
OTHER_LEGACY_FAKE_MASS_WITHIN_LABEL = (
    1.0 - LOWRES_MASS_WITHIN_LABEL - FRONTIER_MASS_WITHIN_FAKE
)
FRONTIER_FAMILY = "frontier-2026-image-generation"


def combined_digest(*values: str) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root not in target.parents and target != root:
            raise RuntimeError(f"unsafe repair archive member: {member.filename}")
    archive.extractall(destination)


def find_repair_root() -> tuple[Path, bool]:
    archives = list(Path("/kaggle/input").rglob(REPAIR_PACKAGE_NAME))
    if len(archives) == 1:
        if runner.file_sha256(archives[0]) != REPAIR_PACKAGE_SHA256:
            raise RuntimeError("repair package SHA-256 mismatch")
        if REPAIR_WORK_ROOT.exists():
            shutil.rmtree(REPAIR_WORK_ROOT)
        REPAIR_WORK_ROOT.mkdir(parents=True)
        with zipfile.ZipFile(archives[0]) as archive:
            safe_extract(archive, REPAIR_WORK_ROOT)
        return REPAIR_WORK_ROOT, True
    if len(archives) > 1:
        raise RuntimeError(f"multiple repair archives: {archives}")

    candidates = []
    for report_path in Path("/kaggle/input").rglob("package.json"):
        try:
            report = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if report.get("inventory_sha256") == REPAIR_INVENTORY_SHA256:
            candidates.append(report_path.parent)
    if len(candidates) != 1:
        raise RuntimeError(f"expected one expanded repair package: {candidates}")
    return candidates[0], False


def verified_repair_rows() -> tuple[Path, list[dict], dict]:
    root, zip_transport_verified = find_repair_root()
    package = json.loads((root / "package.json").read_text())
    if package.get("inventory_sha256") != REPAIR_INVENTORY_SHA256:
        raise RuntimeError("repair inventory mismatch")
    manifest = root / "manifests/manifest.jsonl"
    if runner.file_sha256(manifest) != REPAIR_MANIFEST_SHA256:
        raise RuntimeError("repair manifest mismatch")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    labels = Counter(int(row["label"]) for row in rows)
    if (
        len(rows) != REPAIR_ROWS
        or labels != Counter({0: REPAIR_ROWS_PER_LABEL, 1: REPAIR_ROWS_PER_LABEL})
        or any(not row.get("low_resolution_repair_block") for row in rows)
        or any(row.get("workflow_purpose") != "train-candidate" for row in rows)
    ):
        raise RuntimeError(f"unexpected repair composition: {len(rows)}, {labels}")
    for index, row in enumerate(rows, 1):
        image = (manifest.parent / row["path"]).resolve()
        if runner.file_sha256(image) != row["image_sha256"]:
            raise RuntimeError(f"repair image mismatch: {image}")
        if index % 1000 == 0 or index == len(rows):
            print(json.dumps({"verified_repair_images": index}), flush=True)
    return manifest, rows, {
        "package_sha256": REPAIR_PACKAGE_SHA256,
        "inventory_sha256": REPAIR_INVENTORY_SHA256,
        "manifest_sha256": REPAIR_MANIFEST_SHA256,
        "zip_transport_verified": zip_transport_verified,
        "rows_by_label": dict(sorted(labels.items())),
    }


def absolute_rows(manifest: Path, rows: list[dict]) -> list[dict]:
    return [
        {**row, "path": str((manifest.parent / row["path"]).resolve())}
        for row in rows
    ]


def is_lowres(row: dict) -> bool:
    return bool(row.get("low_resolution_repair_block")) or (
        int(row["label"]) == 0
        and str(row.get("real_source", "")).startswith("CIFAKE-CIFAR10")
    ) or (
        int(row["label"]) == 1
        and str(row.get("generator", "")).startswith("CIFAKE-Stable-Diffusion")
    )


def lowres_paired_sampler(rows: list[dict]) -> tuple[WeightedRandomSampler, dict]:
    real_lowres = Counter(
        str(row.get("real_source"))
        for row in rows
        if int(row["label"]) == 0 and is_lowres(row)
    )
    real_other = Counter(
        str(row.get("real_source"))
        for row in rows
        if int(row["label"]) == 0 and not is_lowres(row)
    )
    fake_lowres = Counter(
        str(row.get("generator"))
        for row in rows
        if int(row["label"]) == 1 and is_lowres(row)
    )
    fake_frontier = Counter(
        str(row.get("generator"))
        for row in rows
        if int(row["label"]) == 1 and row.get("family") == FRONTIER_FAMILY
    )
    fake_other = Counter(
        str(row.get("generator"))
        for row in rows
        if int(row["label"]) == 1
        and not is_lowres(row)
        and row.get("family") != FRONTIER_FAMILY
    )
    groups = (real_lowres, real_other, fake_lowres, fake_frontier, fake_other)
    if any(not group for group in groups) or len(fake_frontier) != 18:
        raise RuntimeError(f"unexpected v10 groups: {groups}")

    weights = []
    for row in rows:
        label = int(row["label"])
        if label == 0 and is_lowres(row):
            group = str(row["real_source"])
            weight = 0.5 * LOWRES_MASS_WITHIN_LABEL / (
                len(real_lowres) * real_lowres[group]
            )
        elif label == 0:
            group = str(row["real_source"])
            weight = 0.5 * OTHER_REAL_MASS_WITHIN_LABEL / (
                len(real_other) * real_other[group]
            )
        elif is_lowres(row):
            group = str(row["generator"])
            weight = 0.5 * LOWRES_MASS_WITHIN_LABEL / (
                len(fake_lowres) * fake_lowres[group]
            )
        elif row.get("family") == FRONTIER_FAMILY:
            group = str(row["generator"])
            weight = 0.5 * FRONTIER_MASS_WITHIN_FAKE / (
                len(fake_frontier) * fake_frontier[group]
            )
        else:
            group = str(row["generator"])
            weight = 0.5 * OTHER_LEGACY_FAKE_MASS_WITHIN_LABEL / (
                len(fake_other) * fake_other[group]
            )
        weights.append(weight)

    observed = {
        "real_lowres": sum(
            weight
            for row, weight in zip(rows, weights)
            if int(row["label"]) == 0 and is_lowres(row)
        ),
        "real_other": sum(
            weight
            for row, weight in zip(rows, weights)
            if int(row["label"]) == 0 and not is_lowres(row)
        ),
        "fake_lowres": sum(
            weight
            for row, weight in zip(rows, weights)
            if int(row["label"]) == 1 and is_lowres(row)
        ),
        "fake_frontier": sum(
            weight
            for row, weight in zip(rows, weights)
            if int(row["label"]) == 1 and row.get("family") == FRONTIER_FAMILY
        ),
        "fake_other": sum(
            weight
            for row, weight in zip(rows, weights)
            if int(row["label"]) == 1
            and not is_lowres(row)
            and row.get("family") != FRONTIER_FAMILY
        ),
    }
    expected = {
        "real_lowres": 0.5 * LOWRES_MASS_WITHIN_LABEL,
        "real_other": 0.5 * OTHER_REAL_MASS_WITHIN_LABEL,
        "fake_lowres": 0.5 * LOWRES_MASS_WITHIN_LABEL,
        "fake_frontier": 0.5 * FRONTIER_MASS_WITHIN_FAKE,
        "fake_other": 0.5 * OTHER_LEGACY_FAKE_MASS_WITHIN_LABEL,
    }
    if any(abs(observed[key] - expected[key]) > 1e-9 for key in expected):
        raise RuntimeError(f"sampler mass mismatch: {observed}")
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(rows),
        replacement=True,
        generator=torch.Generator().manual_seed(runner.SEED),
    )
    return sampler, {
        "policy": (
            "equal labels; 25% of each label reserved for matched low-resolution "
            "rows; 15% of fake-label mass remains frontier; equal named groups "
            "within each remaining block"
        ),
        "target_weight_mass": expected,
        "observed_weight_mass": observed,
        "group_counts": {
            "real_lowres": dict(real_lowres),
            "real_other": dict(real_other),
            "fake_lowres": dict(fake_lowres),
            "fake_frontier": dict(fake_frontier),
            "fake_other": dict(fake_other),
        },
    }


def main() -> None:
    runner.OUTPUT_ROOT = OUTPUT_ROOT
    runner.MODEL_NAMES = ("vit_pe_core_large_patch14_336",)
    runner.EXCLUDED_EVAL_SHA256 = v8.EXCLUDED_EVAL_SHA256
    runner.AUGMENTATION_DESCRIPTION = (
        "v9_one_workshop_transform_plus_matched_lowres_train_block"
    )
    v6_package_sha256, v6_package = runner.validate_package()
    v6_train = runner.WORK_ROOT / "manifests/train.jsonl"
    eval_manifest = runner.WORK_ROOT / "manifests/eval_selection.jsonl"
    content_manifest = runner.WORK_ROOT / "manifests/eval_content_holdout.jsonl"
    v6_rows = [json.loads(line) for line in v6_train.read_text().splitlines() if line]
    frontier_manifest, frontier_rows, frontier_provenance = v8.verified_frontier_rows()
    repair_manifest, repair_rows, repair_provenance = verified_repair_rows()
    combined_rows = (
        absolute_rows(v6_train, v6_rows)
        + absolute_rows(frontier_manifest, frontier_rows)
        + absolute_rows(repair_manifest, repair_rows)
    )
    if len(combined_rows) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(f"unexpected combined rows: {len(combined_rows)}")
    train_hashes = {row["image_sha256"] for row in combined_rows}
    if len(train_hashes) != len(combined_rows):
        raise RuntimeError("duplicate content in combined v10 training manifest")
    frozen_rows = [
        json.loads(line)
        for manifest in (eval_manifest, content_manifest)
        for line in manifest.read_text().splitlines()
        if line
    ]
    frozen_hashes = {row["image_sha256"] for row in frozen_rows if row.get("image_sha256")}
    if train_hashes & frozen_hashes:
        raise RuntimeError("v10 training content overlaps a frozen v6 gate")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    combined_manifest = OUTPUT_ROOT / "train-v10-absolute.jsonl"
    combined_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in combined_rows)
    )
    provenance = {
        "candidate": "v10-lowres-paired",
        "single_intervention": (
            "add disjoint matched CIFAKE low-resolution train block and reserve "
            "25% of each label sampling mass for the combined low-resolution pair"
        ),
        "v6": {
            "package_sha256": v6_package_sha256,
            "inventory_sha256": v6_package["inventory_sha256"],
            "rows": len(v6_rows),
        },
        "frontier": frontier_provenance,
        "repair": repair_provenance,
        "combined_rows": len(combined_rows),
        "combined_manifest_sha256": runner.file_sha256(combined_manifest),
        "train_frozen_v6_gate_sha256_overlap": 0,
        "organizer_demo_rows": 0,
    }
    (OUTPUT_ROOT / "input-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance, indent=2), flush=True)

    runner.source_balanced_sampler = lowres_paired_sampler
    report = runner.train_candidate(
        runner.MODEL_NAMES[0],
        combined_manifest,
        eval_manifest,
        content_manifest,
        combined_digest(v6_package_sha256, v8.FRONTIER_PACKAGE_SHA256, REPAIR_PACKAGE_SHA256),
        combined_digest(
            v6_package["inventory_sha256"],
            v8.FRONTIER_INVENTORY_SHA256,
            REPAIR_INVENTORY_SHA256,
        ),
    )
    summary = {
        "model": report["model"],
        "clean_auc": report["selection_clean"]["clean_auc"],
        "worst_pair_auc": report["selection_clean"]["groups"][
            "worst_generator_real_source_pair_auc"
        ],
        "content_clean_auc": report["content_holdout_clean"]["clean_auc"],
        "content_worst_pair_auc": report["content_holdout_clean"]["groups"][
            "worst_generator_real_source_pair_auc"
        ],
        "checkpoint_sha256": runner.file_sha256(
            OUTPUT_ROOT / runner.MODEL_NAMES[0] / "model.pt"
        ),
    }
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
