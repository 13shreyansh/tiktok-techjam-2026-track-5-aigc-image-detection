#!/usr/bin/env python3
"""Run the frozen one-shot NTIRE v12 candidate arbitration matrix."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from sklearn.metrics import roc_auc_score

import kaggle_evaluate_semantic_modern_v12_gate as base


GATE_SLUG = "track5-ntire-v11-quality-route-gate"
GATE_PACKAGE_SHA256 = "5c68565fbf6a02242af5481e4720dd435b0145ae09306ff4fa80e00c30eeb8c9"
GATE_INVENTORY_SHA256 = "7b9801315bfef184820bf6eef216fd6d11e8dcaf1361387bb6fb9a536db665b6"
GATE_MANIFEST_SHA256 = "cab0d1347bcd78ba2fa8dd09caa0025a181c929ad1ce20fd245af8fd27e8d329"
EXPECTED_ROWS = 1024
EXPECTED_PER_LABEL = 512
STAGE_ROOT = Path("/kaggle/working/ntire-v12-final-arbitration-gate")
AUDIT_ROOT = Path("/kaggle/working/ntire-v12-final-arbitration")
SOURCE_CANDIDATES = {
    name: dict(specification) for name, specification in base.CANDIDATES.items()
}


def mounted_root() -> Path:
    candidates = []
    for package_path in Path("/kaggle/input/datasets").glob(
        f"*/{GATE_SLUG}/**/package.json"
    ):
        metadata = json.loads(package_path.read_text())
        if metadata.get("inventory_sha256") == GATE_INVENTORY_SHA256:
            candidates.append(package_path.parent)
    if len(candidates) != 1:
        raise RuntimeError(f"expected one exact NTIRE gate, found {candidates}")
    return candidates[0]


def validate_gate_package() -> tuple[Path, dict]:
    source_root = mounted_root()
    source_manifest = source_root / "manifests/manifest.jsonl"
    if base.runner.file_sha256(source_manifest) != GATE_MANIFEST_SHA256:
        raise RuntimeError("NTIRE gate manifest checksum mismatch")
    rows = base.read_rows(source_manifest)
    for index, row in enumerate(rows):
        image = (source_manifest.parent / row["path"]).resolve()
        if not image.is_file():
            raise RuntimeError(f"row {index}: missing NTIRE image")
        if base.runner.file_sha256(image) != row.get("image_sha256"):
            raise RuntimeError(f"row {index}: NTIRE image checksum mismatch")

    manifests = STAGE_ROOT / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    staged = manifests / "eval_semantic_matched.jsonl"
    if not staged.exists():
        temporary = staged.with_name(f"{staged.name}.{os.getpid()}.tmp")
        shutil.copy2(source_manifest, temporary)
        os.replace(temporary, staged)
    if base.runner.file_sha256(staged) != GATE_MANIFEST_SHA256:
        raise RuntimeError("staged NTIRE manifest checksum mismatch")
    images_link = STAGE_ROOT / "images"
    if not images_link.exists():
        os.symlink((source_root / "images").resolve(), images_link, target_is_directory=True)
    metadata = json.loads((source_root / "package.json").read_text())
    return STAGE_ROOT, {
        **metadata,
        "container_sha256": GATE_PACKAGE_SHA256,
        "runtime_verification": "manifest plus every image content hash",
    }


def validate_gate_rows(rows: list[dict]) -> dict:
    labels = Counter(int(row["label"]) for row in rows)
    hashes = [str(row.get("image_sha256", "")) for row in rows]
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"unexpected NTIRE rows: {len(rows)}")
    if labels != Counter({0: EXPECTED_PER_LABEL, 1: EXPECTED_PER_LABEL}):
        raise RuntimeError(f"unexpected NTIRE label balance: {labels}")
    if not all(hashes) or len(set(hashes)) != EXPECTED_ROWS:
        raise RuntimeError("NTIRE identities are missing or duplicated")
    for index, row in enumerate(rows):
        if row.get("training_allowed") is True:
            raise RuntimeError(f"row {index}: training permission must not be true")
        lowered = str(row.get("path", "")).casefold()
        if "demo_only" in lowered or "val2017" in lowered or "dall-e" in lowered:
            raise RuntimeError(f"row {index}: organizer-demo source term")
    return {
        "rows": len(rows),
        "labels": dict(labels),
        "unique_images": len(set(hashes)),
        "training_allowed_rows": 0,
        "organizer_demo_rows": 0,
        "generator_identity_available": False,
    }


def validate_identity_separation(rows: list[dict], v12_root: Path) -> dict:
    gate = {str(row["image_sha256"]) for row in rows}
    compared = {}
    for name, role in (("train.jsonl", "train"), ("eval_frozen.jsonl", "eval")):
        current = base.read_rows(v12_root / "manifests" / name)
        identities = {
            str(value)
            for row in current
            for value in (row.get("image_sha256"), row.get("source_image_sha256"))
            if value
        }
        overlap = len(gate & identities)
        if overlap:
            raise RuntimeError(f"NTIRE overlaps v12 {role} identities: {overlap}")
        compared[role] = len(current)
    return {
        "v12_train_rows_compared": compared["train"],
        "v12_eval_rows_compared": compared["eval"],
        "train_identity_overlap": 0,
        "eval_identity_overlap": 0,
        "v12_inventory_sha256": base.V12_INVENTORY_SHA256,
    }


def semantic_metrics(rows: list[dict], predictions: list[dict]) -> dict:
    ordered = sorted(predictions, key=lambda row: int(row["index"]))
    if [int(row["index"]) for row in ordered] != list(range(len(rows))):
        raise RuntimeError("NTIRE prediction indices are incomplete")
    labels = [int(row["label"]) for row in ordered]
    scores = [float(row["score"]) for row in ordered]
    return {"overall_auc": float(roc_auc_score(labels, scores))}


def gate_decision(metrics: dict, official_score: float, worst_condition_auc: float) -> dict:
    floors = {
        "clean_auc": 0.75,
        "official_style_score": 0.70,
        "worst_individual_condition_auc": 0.60,
    }
    observed = {
        "clean_auc": metrics["overall_auc"],
        "official_style_score": official_score,
        "worst_individual_condition_auc": worst_condition_auc,
    }
    checks = {name: observed[name] >= floor for name, floor in floors.items()}
    return {"floors": floors, "observed": observed, "checks": checks}


def materialize_candidate_views() -> None:
    for candidate, source_specification in SOURCE_CANDIDATES.items():
        model_dir = source_specification["model"].replace(".", "_")
        source = source_specification["root"] / model_dir
        target_parent = AUDIT_ROOT / candidate
        target = target_parent / model_dir
        target.mkdir(parents=True, exist_ok=True)
        for name in ("model.pt", "report.json"):
            source_file = source / name
            target_file = target / name
            if not source_file.is_file():
                raise RuntimeError(f"missing frozen v12 source artifact: {source_file}")
            if not target_file.exists():
                try:
                    os.link(source_file, target_file)
                except OSError:
                    shutil.copy2(source_file, target_file)
        observed = base.runner.file_sha256(target / "model.pt")
        if observed != source_specification["checkpoint_sha256"]:
            raise RuntimeError(f"candidate checkpoint mismatch for {candidate}: {observed}")
        base.CANDIDATES[candidate]["root"] = target_parent


def configure() -> None:
    base.GATE_MANIFEST_SHA256 = GATE_MANIFEST_SHA256
    base.validate_gate_package = validate_gate_package
    base.validate_gate_rows = validate_gate_rows
    base.validate_identity_separation = validate_identity_separation
    base.semantic_metrics = semantic_metrics
    base.gate_decision = gate_decision
    materialize_candidate_views()


def evaluate_candidate(candidate: str) -> dict:
    configure()
    result = base.evaluate_candidate(candidate)
    result["decision_boundary"] = (
        "Candidate-unseen but previously consumed external pixels. One-shot final "
        "arbitration only; no training, calibration, preprocessing or weight search."
    )
    specification = base.CANDIDATES[candidate]
    progress = (
        specification["root"]
        / specification["model"].replace(".", "_")
        / "semantic-matched-modern-v6-gate/progress.json"
    )
    base.workshop.atomic_json(progress, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(base.CANDIDATES), required=True)
    args = parser.parse_args()
    result = evaluate_candidate(args.candidate)
    print("NTIRE_V12_ARBITRATION_SUMMARY " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
