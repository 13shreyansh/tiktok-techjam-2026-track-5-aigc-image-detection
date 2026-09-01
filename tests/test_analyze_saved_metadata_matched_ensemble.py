from pathlib import Path


SCRIPT = Path("scripts/analyze_saved_metadata_matched_ensemble.py")


def test_saved_metadata_audit_is_diagnosis_only_and_exactly_matched():
    text = SCRIPT.read_text()
    assert '"diagnosis_only": True' in text
    assert 'str(image.format or "unknown")' in text
    assert "int(image.width)" in text
    assert "int(image.height)" in text
    assert "expected-matched-rows" in text


def test_saved_metadata_audit_uses_frozen_scores_and_official_metric():
    text = SCRIPT.read_text()
    assert 'metric_block(conditions, "v6_score")' in text
    assert 'metric_block(conditions, "score")' in text
    assert '"official_score": 0.5 * (clean_auc + pooled_robust_auc)' in text
    assert 'per_condition["noise_sigma_0.10"]' in text
