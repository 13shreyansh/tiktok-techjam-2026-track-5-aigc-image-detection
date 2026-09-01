#!/usr/bin/env python3
"""Apply the frozen PE-versus-equal-blend NTIRE default decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import score_fixed_v12_blend as fixed


FINAL_PLAN_SHA256 = "1097f5484d1bd1903bebdbbec83e53733707a35b224a442baced87a021148231"
LOCAL_PLAN_SHA256 = "095aaa774d47ea456ad6d42070e26657f13f647c476055a48fb935a1010854b1"
BUFFER_RECOVERY_SHA256 = "cf80d731478ade26719e81459bba4f442dd9f14526d0e11e33cf9c7fbf02e90a"
THRESHOLDS = {
    "pe_official_minus_blend_official_at_least": 0.02,
    "pe_clean_minus_blend_clean_at_least": -0.005,
    "pe_worst_condition_minus_blend_worst_condition_at_least": -0.03,
    "pe_clean_auc_at_least": 0.80,
    "pe_official_style_score_at_least": 0.75,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_plans(root: Path) -> dict:
    expected = {
        "NTIRE_V12_FINAL_ARBITRATION_PLAN.json": FINAL_PLAN_SHA256,
        "NTIRE_V12_LOCAL_RECOVERY_PLAN.json": LOCAL_PLAN_SHA256,
        "NTIRE_V12_MPS_BUFFER_RECOVERY.json": BUFFER_RECOVERY_SHA256,
    }
    observed = {name: sha256_file(root / name) for name in expected}
    if observed != expected:
        raise RuntimeError(f"frozen arbitration plan mismatch: {observed}")
    return observed


def decision(pe: dict, blend: dict) -> dict:
    observed = {
        "pe_official_minus_blend_official": (
            pe["official_style"]["score"] - blend["official_style_score"]
        ),
        "pe_clean_minus_blend_clean": (
            pe["official_style"]["clean_auc"] - blend["clean_auc"]
        ),
        "pe_worst_condition_minus_blend_worst_condition": (
            pe["worst_individual_condition_auc"]
            - blend["worst_individual_condition_auc"]
        ),
        "pe_clean_auc": pe["official_style"]["clean_auc"],
        "pe_official_style_score": pe["official_style"]["score"],
    }
    checks = {
        "pe_official_minus_blend_official_at_least": (
            observed["pe_official_minus_blend_official"]
            >= THRESHOLDS["pe_official_minus_blend_official_at_least"]
        ),
        "pe_clean_minus_blend_clean_at_least": (
            observed["pe_clean_minus_blend_clean"]
            >= THRESHOLDS["pe_clean_minus_blend_clean_at_least"]
        ),
        "pe_worst_condition_minus_blend_worst_condition_at_least": (
            observed["pe_worst_condition_minus_blend_worst_condition"]
            >= THRESHOLDS["pe_worst_condition_minus_blend_worst_condition_at_least"]
        ),
        "pe_clean_auc_at_least": (
            observed["pe_clean_auc"] >= THRESHOLDS["pe_clean_auc_at_least"]
        ),
        "pe_official_style_score_at_least": (
            observed["pe_official_style_score"]
            >= THRESHOLDS["pe_official_style_score_at_least"]
        ),
    }
    return {
        "thresholds": THRESHOLDS,
        "observed": observed,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "selected_default": "pe_core" if all(checks.values()) else "fixed_equal_blend",
        "fallback": "pe_core",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pe-root", type=Path, required=True)
    parser.add_argument("--dino-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    plans = validate_plans(args.repo_root)
    pe = json.loads((args.pe_root / "progress.json").read_text())
    dino = json.loads((args.dino_root / "progress.json").read_text())
    if not pe.get("completed") or not dino.get("completed"):
        raise RuntimeError("both local candidate matrices must be complete")
    blend = fixed.score(args.pe_root, args.dino_root)
    result = {
        "status": "completed_frozen_final_arbitration",
        "plan_sha256": plans,
        "pe_core": {
            "checkpoint_sha256": pe["signature"]["checkpoint_sha256"],
            "clean_auc": pe["official_style"]["clean_auc"],
            "pooled_robust_auc": pe["official_style"]["pooled_robust_auc"],
            "official_style_score": pe["official_style"]["score"],
            "worst_individual_condition_auc": pe["worst_individual_condition_auc"],
        },
        "dinov2_control": {
            "checkpoint_sha256": dino["signature"]["checkpoint_sha256"],
            "clean_auc": dino["official_style"]["clean_auc"],
            "pooled_robust_auc": dino["official_style"]["pooled_robust_auc"],
            "official_style_score": dino["official_style"]["score"],
            "worst_individual_condition_auc": dino["worst_individual_condition_auc"],
        },
        "fixed_equal_blend": blend,
        "frozen_default_decision": decision(pe, blend),
        "organizer_demo_rows": 0,
        "boundary": (
            "Final one-shot runtime-default arbitration on a previously consumed "
            "external gate; not a hidden-set estimate or permission for more search."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
