from pathlib import Path


SCRIPT = Path("scripts/kaggle_verify_ensemble_fp16_contract.py")


def test_fp16_contract_is_frozen_to_original_batch_and_blend_arithmetic():
    text = SCRIPT.read_text()
    assert "PHYSICAL_BATCH_SIZE = 64" in text
    assert "LOGICAL_BATCH_SIZES = (1, 17, 64)" in text
    assert 'promotion.V6_WEIGHT * v6_scores' in text
    assert 'promotion.V9_WEIGHT * v9_scores.to("cuda:0")' in text
    assert 'with torch.autocast(device_type="cuda", dtype=torch.float16)' in text


def test_fp16_contract_requires_exact_saved_prediction_reproduction():
    text = SCRIPT.read_text()
    assert '"max_absolute_score_drift": 0.0' in text
    assert '"auc_drift": 0.0' in text
    assert '"max_rank_displacement": 0' in text
    assert "fixed.SAVED_CLEAN" in text
    assert "and all(exact(values) for values in saved_checks.values())" in text


def test_fp16_contract_uses_only_frozen_checkpoints_and_open_qwen_rows():
    text = SCRIPT.read_text()
    assert "promotion.V6_SHA256" in text
    assert "promotion.V9_SHA256" in text
    assert 'sealed.MANIFESTS["qwen_prompt_holdout"]' in text
    assert "ROWS_PER_LABEL = 32" in text
    assert "demo" not in text.lower()
