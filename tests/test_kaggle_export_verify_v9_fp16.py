import json
from pathlib import Path


def test_fp16_packaging_audit_preserves_promotion_boundary() -> None:
    source = Path("scripts/kaggle_export_verify_v9_fp16.py").read_text()
    assert "expected_source_sha256=promotion.V9_SHA256" in source
    assert "passes_exact_clean_qwen_screen" in source
    assert "packaging_ablation_not_promoted" in source
    assert "unchanged full" in source


def test_observed_fp16_packaging_result_is_rejected_without_relaxing_gate() -> None:
    result = json.loads(Path("FRONTIER_V9_FP16_EXPORT_AUDIT_RESULT.json").read_text())
    assert result["completed"] is True
    assert result["passes_exact_clean_qwen_screen"] is False
    assert result["source_checkpoint_sha256"] == (
        "dd6b26c7849489447c7e96823f5b5e87c31623ca8ae0d28a5b162bb2dcb65075"
    )
    comparison = result["clean_qwen_exact_comparison"][
        "saved_prediction_comparison"
    ]["blend"]
    assert comparison["max_absolute_score_drift"] == 0.00390625
    assert comparison["max_rank_displacement"] == 4
    assert comparison["auc_drift"] != 0
