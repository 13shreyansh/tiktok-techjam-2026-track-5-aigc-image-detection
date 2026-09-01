import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import score_ntire_v12_final_arbitration as arbitration


def progress(clean: float, pooled: float, worst: float) -> dict:
    return {
        "official_style": {
            "clean_auc": clean,
            "pooled_robust_auc": pooled,
            "score": 0.5 * clean + 0.5 * pooled,
        },
        "worst_individual_condition_auc": worst,
    }


def blend(clean: float, pooled: float, worst: float) -> dict:
    return {
        "clean_auc": clean,
        "pooled_robust_auc": pooled,
        "official_style_score": 0.5 * clean + 0.5 * pooled,
        "worst_individual_condition_auc": worst,
    }


def test_every_frozen_check_must_pass_to_select_pe():
    result = arbitration.decision(
        progress(clean=0.91, pooled=0.91, worst=0.81),
        blend(clean=0.91, pooled=0.86, worst=0.83),
    )
    assert result["all_checks_pass"] is True
    assert result["selected_default"] == "pe_core"


def test_worst_condition_guard_retains_blend():
    result = arbitration.decision(
        progress(clean=0.91, pooled=0.91, worst=0.79),
        blend(clean=0.91, pooled=0.86, worst=0.83),
    )
    assert result["all_checks_pass"] is False
    assert result["selected_default"] == "fixed_equal_blend"


def test_plan_hashes_are_bound_to_committed_files():
    root = Path(__file__).parents[1]
    observed = arbitration.validate_plans(root)
    assert observed["NTIRE_V12_FINAL_ARBITRATION_PLAN.json"] == arbitration.FINAL_PLAN_SHA256
    assert observed["NTIRE_V12_LOCAL_RECOVERY_PLAN.json"] == arbitration.LOCAL_PLAN_SHA256
    assert observed["NTIRE_V12_MPS_BUFFER_RECOVERY.json"] == arbitration.BUFFER_RECOVERY_SHA256
