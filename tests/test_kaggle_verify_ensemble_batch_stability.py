from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/kaggle_verify_ensemble_batch_stability.py"
)


def source() -> str:
    return SCRIPT.read_text()


def test_batch_audit_freezes_sizes_and_tolerances():
    text = source()
    assert "BATCH_SIZES = (64, 128)" in text
    assert "MAX_ABSOLUTE_SCORE_DRIFT = 1e-4" in text
    assert "MAX_AUC_DRIFT = 1e-6" in text
    assert "MAX_RANK_DISPLACEMENT = 1" in text


def test_batch_audit_hashes_identical_transformed_tensors():
    text = source()
    assert "input_digest.update(images.contiguous().numpy().tobytes())" in text
    assert '"input_tensors_identical"' in text


def test_batch_audit_uses_frozen_checkpoints_and_blend():
    text = source()
    assert "promotion.V6_SHA256" in text
    assert "promotion.V9_SHA256" in text
    assert "promotion.V6_WEIGHT * v6_cpu + promotion.V9_WEIGHT * v9_cpu" in text
    assert "Qwen clean holdout only" in text


def test_batch_audit_compares_against_saved_promotion_predictions():
    text = source()
    assert "clean_predictions.jsonl" in text
    assert '"batch_128_vs_saved_clean"' in text
