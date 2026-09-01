from pathlib import Path


SCRIPT = Path("scripts/kaggle_verify_ensemble_single_gpu_contract.py")


def test_single_gpu_contract_freezes_original_policy_and_full_open_gate():
    text = SCRIPT.read_text()
    assert 'PHYSICAL_BATCH_SIZE = 64' in text
    assert 'EXPECTED_ROWS = 576' in text
    assert 'DEVICE = "cuda:0"' in text
    assert 'promotion.V6_WEIGHT * v6_scores' in text
    assert 'promotion.V9_WEIGHT * v9_scores' in text
    assert 'with torch.autocast(device_type="cuda", dtype=torch.float16)' in text


def test_single_gpu_contract_requires_zero_drift_against_saved_rows():
    text = SCRIPT.read_text()
    assert 'fixed.SAVED_CLEAN' in text
    assert 'values["max_absolute_score_drift"] == 0.0' in text
    assert 'values["auc_drift"] == 0.0' in text
    assert 'values["max_rank_displacement"] == 0' in text


def test_single_gpu_contract_records_real_resource_evidence():
    text = SCRIPT.read_text()
    assert 'load_seconds' in text
    assert 'images_per_forward_second' in text
    assert 'cuda_peak_allocated_bytes' in text
    assert 'torch.cuda.get_device_name(0)' in text


def test_single_gpu_contract_uses_only_frozen_open_resources():
    text = SCRIPT.read_text()
    assert 'promotion.V6_SHA256' in text
    assert 'promotion.V9_SHA256' in text
    assert 'sealed.MANIFESTS["qwen_prompt_holdout"]' in text
    assert 'demo' not in text.lower()
