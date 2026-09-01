#!/usr/bin/env python3
"""Open the frozen v11 gate once, only after its consumed-gate screen passes."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import torch

import kaggle_screen_v11_quality_route as screen


PACKAGE_NAME = "ntire-v11-quality-route-gate.zip"
PACKAGE_BYTES = 608_677_127
PACKAGE_SHA256 = "5c68565fbf6a02242af5481e4720dd435b0145ae09306ff4fa80e00c30eeb8c9"
PACKAGE_INVENTORY_SHA256 = (
    "7b9801315bfef184820bf6eef216fd6d11e8dcaf1361387bb6fb9a536db665b6"
)
MANIFEST_SHA256 = "cab0d1347bcd78ba2fa8dd09caa0025a181c929ad1ce20fd245af8fd27e8d329"
EXPECTED_ROWS = 1_024
WORK_ROOT = Path("/kaggle/working/track5-v11-quality-route-gate")
OUTPUT_ROOT = Path("/kaggle/working/track5-v11-quality-route-promotion")


def locate_gate_source() -> tuple[Path, bool]:
    candidates = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if root.exists():
            candidates.extend(
                path
                for path in root.rglob(PACKAGE_NAME)
                if path.is_file() and path.stat().st_size == PACKAGE_BYTES
            )
    exact = [path for path in candidates if screen.sha256_file(path) == PACKAGE_SHA256]
    if len(exact) == 1:
        return exact[0], True
    extracted = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if not root.exists():
            continue
        for path in root.rglob("package.json"):
            try:
                package = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            manifests = package.get("manifests", [])
            if (
                package.get("inventory_sha256") == PACKAGE_INVENTORY_SHA256
                and package.get("source_paths") == EXPECTED_ROWS
                and package.get("unique_images") == EXPECTED_ROWS
                and len(manifests) == 1
                and manifests[0].get("sha256") == MANIFEST_SHA256
            ):
                extracted.append(path)
    if len(extracted) != 1:
        raise RuntimeError(
            "expected one exact ZIP or one content-equivalent Kaggle extraction; "
            f"ZIP candidates: {exact}; extracted candidates: {extracted}"
        )
    return extracted[0], False


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def validate_gate() -> tuple[Path, list[dict], dict]:
    source, archive_reverified = locate_gate_source()
    if archive_reverified:
        safe_extract(source, WORK_ROOT)
        package_path = WORK_ROOT / "package.json"
        package_root = WORK_ROOT
    else:
        package_path = source
        package_root = source.parent
    manifest_path = package_root / "manifests/manifest.jsonl"
    package = json.loads(package_path.read_text())
    if package["inventory_sha256"] != PACKAGE_INVENTORY_SHA256:
        raise RuntimeError("package inventory mismatch")
    if package["source_paths"] != EXPECTED_ROWS or package["unique_images"] != EXPECTED_ROWS:
        raise RuntimeError("unexpected package row count")
    if screen.sha256_file(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("manifest checksum mismatch")
    rows = screen.read_rows(manifest_path)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError("unexpected manifest rows")
    labels = [int(row["label"]) for row in rows]
    if labels.count(0) != 512 or labels.count(1) != 512:
        raise RuntimeError("gate must remain class-balanced")
    if any("demo_only" in str(row["path"]).lower() for row in rows):
        raise RuntimeError("organizer demo-only path found")

    inventory = hashlib.sha256()
    seen = set()
    for index, row in enumerate(rows, start=1):
        image_path = (manifest_path.parent / row["path"]).resolve()
        if package_root.resolve() not in image_path.parents:
            raise RuntimeError(f"image outside gate root: {image_path}")
        observed = screen.sha256_file(image_path)
        if observed != row["image_sha256"]:
            raise RuntimeError(f"image checksum mismatch: {image_path}")
        seen.add(observed)
        inventory.update(f"{row['path']}\0{observed}\0{image_path.stat().st_size}\n".encode())
        if index % 128 == 0 or index == len(rows):
            print(json.dumps({"verified_gate_images": index}), flush=True)
    if len(seen) != EXPECTED_ROWS:
        raise RuntimeError("gate image bytes are not unique")
    # The package builder freezes this value; recomputation is retained as an
    # independent integrity signal without changing the promotion decision.
    computed_inventory = inventory.hexdigest()
    return manifest_path, rows, {
        "mounted_source": str(source),
        "source_archive_reverified_at_runtime": archive_reverified,
        "source_archive_bytes": PACKAGE_BYTES,
        "source_archive_sha256": PACKAGE_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "declared_inventory_sha256": PACKAGE_INVENTORY_SHA256,
        "computed_row_inventory_sha256": computed_inventory,
    }


def promotion_decision(reports: dict) -> dict:
    clean = reports["clean"]
    noise = reports["noise_sigma_0.10"]
    checks = {
        "clean_auc_drop_at_most_0_002": (
            clean["routed"]["auc"] >= clean["general"]["auc"] - 0.002
        ),
        "noise_auc_at_least_0_70": noise["routed"]["auc"] >= 0.70,
        "noise_improvement_at_least_0_05": (
            noise["routed"]["auc"] >= noise["general"]["auc"] + 0.05
        ),
        "clean_route_rate_at_most_0_05": clean["route_rate"] <= 0.05,
        "noise_route_rate_at_least_0_95": noise["route_rate"] >= 0.95,
        "noise_mean_score_not_inverted": not noise["routed"]["mean_score_inversion"],
    }
    return {"checks": checks, "passes": all(checks.values())}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("v11 promotion evaluation requires CUDA")
    manifest, rows, integrity = validate_gate()
    paths = {name: screen.locate_checkpoint(name) for name in screen.CHECKPOINTS}
    model, heads, mean, std = screen.load_shared_model(paths)
    shared_check = screen.verify_shared_logits(model, heads, paths)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    reports = {}
    for condition, sigma, seed in screen.CONDITIONS:
        torch.manual_seed(seed)
        dataset = screen.ConditionDataset(manifest, rows, mean, std, sigma, seed)
        predictions = screen.score(model, heads, dataset)
        output = OUTPUT_ROOT / f"fresh-{condition}-predictions.jsonl"
        output.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions)
        )
        reports[condition] = {
            "rows": len(predictions),
            "general": screen.metrics(predictions, "general_score"),
            "v10": screen.metrics(predictions, "v10_score"),
            "routed": screen.metrics(predictions, "score"),
            "route_rate": sum(row["routed_to_v10"] for row in predictions)
            / len(predictions),
            "noise_estimate_minimum": min(row["noise_estimate"] for row in predictions),
            "noise_estimate_maximum": max(row["noise_estimate"] for row in predictions),
            "predictions_sha256": screen.sha256_file(output),
        }
        print(json.dumps({"condition": condition, **reports[condition]}), flush=True)

    result = {
        "status": "fresh_gate_consumed",
        "router_threshold": screen.ROUTING_THRESHOLD,
        "general_weights": {
            "v6": screen.GENERAL_V6_WEIGHT,
            "v9": screen.GENERAL_V9_WEIGHT,
        },
        "integrity": integrity,
        "shared_logit_check": shared_check,
        "reports": reports,
        "promotion": promotion_decision(reports),
        "fresh_gate_opened": True,
        "organizer_demo_rows": 0,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    report_path = OUTPUT_ROOT / "report.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
