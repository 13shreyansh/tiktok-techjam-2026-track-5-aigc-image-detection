#!/usr/bin/env python3
"""Audit the exact v12 candidates on frozen Community Forensics pixels.

This is a diagnostic-only comparison.  The source was previously opened by
the historical v6 lineage, contains only four fakes per named model, and may
not train, tune, calibrate or reweight the v12 candidates.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import roc_auc_score

import kaggle_evaluate_semantic_modern_v12_gate as base


GATE_DATASET_SLUG = "track5-community-forensics-external-audit"
GATE_PACKAGE_SHA256 = "123a0e4bb8ae484a804a6a39a9f20063e62116e260fef813efe44824cc11a084"
GATE_INVENTORY_SHA256 = "5b345de1d57badd4a9bbc6b33876a29f2660c9cd173cff54aa9a379038578943"
SOURCE_MANIFEST_SHA256 = "2d770ff99f781320a10a9a15fa03de79d2cab40929b09fb1b4db7e759848398c"
GATE_MANIFEST_SHA256 = "5496fd110f73873c1d01f520ae3cf44893d3b036c3ab66f65ce4ef3ef04881d5"
EXPECTED_ROWS = 593
EXPECTED_REAL = 281
EXPECTED_FAKE = 312
EXPECTED_MODELS = 78
STAGE_ROOT = Path("/kaggle/working/community-forensics-v12-gate")
AUDIT_ROOT = Path("/kaggle/working/community-forensics-v12")
SOURCE_CANDIDATES = {
    name: dict(specification) for name, specification in base.CANDIDATES.items()
}


def _mounted_gate_root() -> Path:
    candidates = []
    pattern = f"*/{GATE_DATASET_SLUG}/**/package.json"
    for package_path in Path("/kaggle/input/datasets").glob(pattern):
        metadata = json.loads(package_path.read_text())
        if metadata.get("inventory_sha256") == GATE_INVENTORY_SHA256:
            candidates.append(package_path.parent)
    if len(candidates) != 1:
        raise RuntimeError(f"expected one Community Forensics gate, found {candidates}")
    return candidates[0]


def validate_gate_package() -> tuple[Path, dict]:
    root = _mounted_gate_root()
    manifest = root / "manifests/manifest.jsonl"
    if base.runner.file_sha256(manifest) != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("Community Forensics manifest checksum mismatch")
    rows = base.read_rows(manifest)
    for index, row in enumerate(rows):
        path = (manifest.parent / row["path"]).resolve()
        if not path.is_file():
            raise RuntimeError(f"row {index}: missing gate image")
        if base.runner.file_sha256(path) != row.get("image_sha256"):
            raise RuntimeError(f"row {index}: gate image checksum mismatch")

    v12_root = base.locate_v12_root()
    v12_identities = {
        str(value)
        for name in ("train.jsonl", "eval_frozen.jsonl")
        for row in base.read_rows(v12_root / "manifests" / name)
        for value in (row.get("image_sha256"), row.get("source_image_sha256"))
        if value
    }
    retained = [
        row for row in rows if str(row.get("image_sha256")) not in v12_identities
    ]
    excluded = [
        row for row in rows if str(row.get("image_sha256")) in v12_identities
    ]
    if len(retained) != EXPECTED_ROWS or len(excluded) != 31:
        raise RuntimeError(
            f"unexpected Community Forensics overlap derivation: retained={len(retained)}, excluded={len(excluded)}"
        )

    manifests = STAGE_ROOT / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    staged_manifest = manifests / "eval_semantic_matched.jsonl"
    temporary = staged_manifest.with_name(staged_manifest.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in retained)
    )
    temporary.replace(staged_manifest)
    if base.runner.file_sha256(staged_manifest) != GATE_MANIFEST_SHA256:
        raise RuntimeError("derived Community Forensics manifest checksum mismatch")
    images_link = STAGE_ROOT / "images"
    if not images_link.exists():
        os.symlink((root / "images").resolve(), images_link, target_is_directory=True)

    metadata = json.loads((root / "package.json").read_text())
    return STAGE_ROOT, {
        **metadata,
        "container_sha256": GATE_PACKAGE_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "derived_manifest_sha256": GATE_MANIFEST_SHA256,
        "overlap_rows_excluded": 31,
        "runtime_verification": "manifest plus every image content hash",
    }


def validate_gate_rows(rows: list[dict]) -> dict:
    labels = Counter(int(row["label"]) for row in rows)
    models = {
        str(row["generator_model"])
        for row in rows
        if int(row["label"]) == 1
    }
    hashes = [str(row.get("image_sha256", "")) for row in rows]
    if len(rows) != EXPECTED_ROWS or labels != Counter({0: EXPECTED_REAL, 1: EXPECTED_FAKE}):
        raise RuntimeError(f"unexpected Community Forensics balance: {labels}")
    if len(models) != EXPECTED_MODELS:
        raise RuntimeError(f"unexpected Community Forensics model count: {len(models)}")
    if not all(hashes) or len(set(hashes)) != EXPECTED_ROWS:
        raise RuntimeError("Community Forensics identities are missing or duplicated")
    explicit_training_flags = 0
    for index, row in enumerate(rows):
        # This 2026-08-29 package predates row-level use flags.  Its signed
        # package/report contract marks the complete gate audit-only.  Keep the
        # original manifest byte-exact, accept an absent legacy field, and
        # still fail closed if any row explicitly permits training.
        if row.get("training_allowed") is True:
            raise RuntimeError(f"row {index}: training_allowed must not be true")
        explicit_training_flags += int("training_allowed" in row)
        lowered = str(row.get("path", "")).casefold()
        if "coco" in lowered or "dall-e" in lowered or "dalle" in lowered:
            raise RuntimeError(f"row {index}: organizer-demo source term")
    return {
        "rows": len(rows),
        "labels": dict(labels),
        "fake_model_variants": len(models),
        "fake_rows_per_model": 4,
        "unique_images": len(set(hashes)),
        "organizer_demo_rows": 0,
        "training_allowed_rows": 0,
        "row_level_training_flags_present": explicit_training_flags,
        "audit_only_authority": "signed package report",
    }


def validate_identity_separation(rows: list[dict], v12_root: Path) -> dict:
    gate = {str(row["image_sha256"]) for row in rows}
    train_overlap = 0
    eval_overlap = 0
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
        if role == "train":
            train_overlap = overlap
        else:
            eval_overlap = overlap
        compared[role] = len(current)
    if train_overlap or eval_overlap:
        raise RuntimeError(
            f"Community Forensics overlaps v12 identities: train={train_overlap}, eval={eval_overlap}"
        )
    return {
        "v12_train_rows_compared": compared["train"],
        "v12_eval_rows_compared": compared["eval"],
        "train_identity_overlap": 0,
        "eval_identity_overlap": 0,
        "v12_inventory_sha256": base.V12_INVENTORY_SHA256,
    }


def _aligned_scores(rows: list[dict], predictions: list[dict]) -> list[float]:
    ordered = sorted(predictions, key=lambda row: int(row["index"]))
    if [int(row["index"]) for row in ordered] != list(range(len(rows))):
        raise RuntimeError("Community Forensics prediction indices are incomplete")
    if any(int(pred["label"]) != int(row["label"]) for row, pred in zip(rows, ordered)):
        raise RuntimeError("Community Forensics prediction label mismatch")
    return [float(row["score"]) for row in ordered]


def per_model_auc(rows: list[dict], scores: list[float]) -> dict[str, float]:
    real_scores = [score for row, score in zip(rows, scores) if int(row["label"]) == 0]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, score in zip(rows, scores):
        if int(row["label"]) == 1:
            grouped[str(row["generator_model"])].append(score)
    return {
        model: float(
            roc_auc_score(
                [0] * len(real_scores) + [1] * len(fake_scores),
                real_scores + fake_scores,
            )
        )
        for model, fake_scores in sorted(grouped.items())
    }


def clean_metrics(rows: list[dict], predictions: list[dict]) -> dict:
    scores = _aligned_scores(rows, predictions)
    labels = [int(row["label"]) for row in rows]
    by_model = per_model_auc(rows, scores)
    return {
        "overall_auc": float(roc_auc_score(labels, scores)),
        "by_fake_model": by_model,
        "worst_fake_model_auc": min(by_model.values()),
        "fake_models": len(by_model),
        "small_sample_warning": "four fake images per named latent-diffusion variant",
    }


def pooled_metrics(rows: list[dict], predictions: list[dict]) -> dict:
    condition_count = len(base.workshop.conditions()) - 1
    if len(predictions) != len(rows) * condition_count:
        raise RuntimeError("Community Forensics pooled prediction count mismatch")
    expanded = rows * condition_count
    scores = [float(row["score"]) for row in predictions]
    labels = [int(row["label"]) for row in predictions]
    by_model = per_model_auc(expanded, scores)
    groups = base.runner.grouped_metrics(expanded, labels, scores)
    groups.update(
        {
            "fake_model_auc": by_model,
            "worst_fake_model_auc": min(by_model.values()),
        }
    )
    return {"auc": float(roc_auc_score(labels, scores)), "groups": groups}


def preliminary_decision(metrics: dict, official_score: float, worst_condition_auc: float) -> dict:
    floors = {
        "clean_auc": 0.85,
        "official_style_score": 0.80,
        "worst_individual_condition_auc": 0.70,
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
    base.semantic_metrics = clean_metrics
    base.gate_decision = preliminary_decision
    base.workshop.pooled_metrics = pooled_metrics
    materialize_candidate_views()


def evaluate_candidate(candidate: str) -> dict:
    configure()
    result = base.evaluate_candidate(candidate)
    pooled_worst = result["pooled_robust_groups"]["worst_fake_model_auc"]
    preliminary = result["frozen_gate_decision"]
    floors = {**preliminary["floors"], "pooled_worst_fake_model_auc": 0.50}
    observed = {
        **preliminary["observed"],
        "pooled_worst_fake_model_auc": pooled_worst,
    }
    checks = {name: observed[name] >= floor for name, floor in floors.items()}
    result["frozen_gate_decision"] = {
        "floors": floors,
        "observed": observed,
        "checks": checks,
        "passes_all_frozen_floors": all(checks.values()),
        "boundary": "Diagnostic-only pass or failure; no tuning, reweighting or hidden-score claim.",
    }
    result["decision_boundary"] = (
        "Previously opened external source, now byte-disjoint from v12 train/eval. "
        "Diagnostic only; may not train, tune, calibrate or search blend weights."
    )
    specification = base.CANDIDATES[candidate]
    progress = (
        specification["root"]
        / specification["model"].replace(".", "_")
        / "semantic-matched-modern-v6-gate"
        / "progress.json"
    )
    base.workshop.atomic_json(progress, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(base.CANDIDATES), required=True)
    args = parser.parse_args()
    result = evaluate_candidate(args.candidate)
    print("COMMUNITY_FORENSICS_V12_SUMMARY " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
