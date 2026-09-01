from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/kaggle_verify_ensemble_fixed_batch.py"


def source() -> str:
    return SCRIPT.read_text()


def test_fixed_batch_policy_and_logical_sizes_are_frozen():
    text = source()
    assert "PHYSICAL_BATCH_SIZE = 64" in text
    assert "LOGICAL_BATCH_SIZES = (1, 17, 64)" in text
    assert "ROWS_PER_LABEL = 32" in text


def test_final_batch_is_padded_without_changing_original_outputs():
    text = source()
    assert "PHYSICAL_BATCH_SIZE - original_count" in text
    assert "v6_scores[:original_count]" in text
    assert "v9_scores[:original_count]" in text


def test_fixed_batch_gate_does_not_relax_observed_tolerances():
    text = source()
    assert "MAX_ABSOLUTE_SCORE_DRIFT = 1e-6" in text
    assert "MAX_AUC_DRIFT = 0.0" in text
    assert "MAX_RANK_DISPLACEMENT = 0" in text


def test_fixed_batch_is_compared_to_saved_promotion_scores():
    text = source()
    assert "clean_predictions.jsonl" in text
    assert '"native_64_vs_saved_clean_subset"' in text
