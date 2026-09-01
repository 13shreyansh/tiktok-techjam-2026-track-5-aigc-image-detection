import pytest

from aigc_detector.metrics import official_metrics


def test_official_score_uses_equal_auc_weights():
    result = official_metrics(
        clean_labels=[0, 0, 1, 1],
        clean_scores=[0.1, 0.2, 0.8, 0.9],
        robust_labels=[0, 0, 1, 1],
        robust_scores=[0.1, 0.7, 0.6, 0.8],
    )
    assert result.clean_auc == 1.0
    assert result.robust_auc == pytest.approx(0.75)
    assert result.official_score == pytest.approx(0.875)


def test_ai_generated_is_positive_class():
    result = official_metrics(
        clean_labels=[0, 1],
        clean_scores=[0.0, 1.0],
        robust_labels=[0, 1],
        robust_scores=[0.0, 1.0],
    )
    assert result.official_score == 1.0
