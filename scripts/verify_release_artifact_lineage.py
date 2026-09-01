#!/usr/bin/env python3
"""Verify that release records refer to the correct checkpoint artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(
    manifest_path: Path,
    evidence_paths: list[Path],
    compact_audit_path: Path,
    v6_local_path: Path,
    v9_local_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    v6 = manifest["checkpoints"]["v6"]
    v9 = manifest["checkpoints"]["v9"]
    expected = {"v6": v6["sha256"], "v9": v9["sha256"]}
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, observed: Any, expected_value: Any) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "observed": observed,
                "expected": expected_value,
            }
        )

    check(
        "combined_checkpoint_bytes",
        manifest["combined_checkpoint_bytes"] == v6["bytes"] + v9["bytes"],
        manifest["combined_checkpoint_bytes"],
        v6["bytes"] + v9["bytes"],
    )
    check(
        "compact_ablation_not_attached_to_v6",
        "rejected_fp16_storage_ablation" not in v6,
        "rejected_fp16_storage_ablation" in v6,
        False,
    )

    for path in evidence_paths:
        record = json.loads(path.read_text())
        blend = record.get("blend", {})
        observed_v6 = record.get("v6_checkpoint_sha256", blend.get("v6_checkpoint_sha256"))
        observed_v9 = record.get("v9_checkpoint_sha256", blend.get("v9_checkpoint_sha256"))
        check(f"{path.name}:v6", observed_v6 == expected["v6"], observed_v6, expected["v6"])
        check(f"{path.name}:v9", observed_v9 == expected["v9"], observed_v9, expected["v9"])

    compact = json.loads(compact_audit_path.read_text())
    rejected = v9["rejected_fp16_storage_ablation"]
    check(
        "compact_source_is_exact_v9",
        compact["source_checkpoint_sha256"] == expected["v9"],
        compact["source_checkpoint_sha256"],
        expected["v9"],
    )
    check(
        "compact_hash_attached_to_v9",
        compact["compact_checkpoint_sha256"] == rejected["sha256"],
        compact["compact_checkpoint_sha256"],
        rejected["sha256"],
    )
    check(
        "compact_decision_rejected",
        compact["passes_exact_clean_qwen_screen"] is False
        and rejected["exact_clean_qwen_screen_passed"] is False,
        compact["passes_exact_clean_qwen_screen"],
        False,
    )

    local: dict[str, Any] = {}
    for name, path, record in (("v6", v6_local_path, v6), ("v9", v9_local_path, v9)):
        if not path.exists():
            local[name] = {"path": str(path), "present": False}
            continue
        observed_size = path.stat().st_size
        observed_hash = sha256_file(path)
        local[name] = {
            "path": str(path),
            "present": True,
            "bytes": observed_size,
            "sha256": observed_hash,
            "matches_manifest": observed_size == record["bytes"] and observed_hash == record["sha256"],
        }
        check(
            f"local_{name}_matches_manifest",
            local[name]["matches_manifest"],
            {"bytes": observed_size, "sha256": observed_hash},
            {"bytes": record["bytes"], "sha256": record["sha256"]},
        )

    blockers = []
    if not local["v9"]["present"]:
        blockers.append("exact v9 checkpoint is absent locally")
    for name, record in (("v6", v6), ("v9", v9)):
        if not record.get("distribution_url"):
            blockers.append(f"{name} has no public immutable distribution URL")

    return {
        "purpose": "Artifact-lineage consistency and distribution-readiness audit.",
        "checks_passed": all(item["passed"] for item in checks),
        "checks": checks,
        "local_artifacts": local,
        "distribution_ready": not blockers and all(item["passed"] for item in checks),
        "distribution_blockers": blockers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("ENSEMBLE_CHECKPOINT_MANIFEST.json"))
    parser.add_argument("--evidence", action="append", type=Path, required=True)
    parser.add_argument("--compact-audit", type=Path, default=Path("FRONTIER_V9_FP16_EXPORT_AUDIT_RESULT.json"))
    parser.add_argument("--v6", type=Path, default=Path("models/model.pt"))
    parser.add_argument("--v9", type=Path, default=Path("models/model_v9.pt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = audit(args.manifest, args.evidence, args.compact_audit, args.v6, args.v9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
