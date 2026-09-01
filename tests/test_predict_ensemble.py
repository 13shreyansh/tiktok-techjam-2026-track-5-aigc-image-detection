from pathlib import Path

import pytest

from aigc_detector.predict_ensemble import (
    PHYSICAL_BATCH_SIZE,
    V6_SHA256,
    V6_WEIGHT,
    V9_SHA256,
    V9_WEIGHT,
    shared_inference_config,
)


def test_promoted_constants_are_frozen():
    assert PHYSICAL_BATCH_SIZE == 64
    assert V6_WEIGHT == 0.75
    assert V9_WEIGHT == 0.25
    assert V6_SHA256 == "48ea50773fbd1b7247fff25fde6f985183e29f2eb517b5ac0f6319c1fe38b644"
    assert V9_SHA256 == "dd6b26c7849489447c7e96823f5b5e87c31623ca8ae0d28a5b162bb2dcb65075"


def test_checkpoint_configuration_must_match():
    base = {
        "model_name": "m",
        "image_size": 224,
        "head_mode": "linear",
        "normalization_mean": (0.1, 0.2, 0.3),
        "normalization_std": (0.4, 0.5, 0.6),
        "preprocess_mode": "short_side_crop",
        "codec_normalization": "none",
    }
    assert shared_inference_config(base, dict(base)) == base
    changed = dict(base, preprocess_mode="stretch")
    with pytest.raises(SystemExit, match="preprocess_mode"):
        shared_inference_config(base, changed)


def test_candidate_runner_preserves_fp16_blend_and_cuda_fail_closed():
    text = Path("src/aigc_detector/predict_ensemble.py").read_text()
    assert 'with torch.autocast(device_type="cuda", dtype=torch.float16)' in text
    assert "V6_WEIGHT * v6_scores + V9_WEIGHT * v9_scores" in text
    assert 'if device.type != "cuda"' in text
    assert "use run.sh" in text
    assert 'config.get("preprocess_mode") or "stretch"' in text
    assert 'config.get("codec_normalization") or "none"' in text


def test_candidate_runner_does_not_replace_selected_runner():
    selected = Path("run.sh").read_text()
    candidate = Path("run_ensemble.sh").read_text()
    assert "aigc_detector.predict_ensemble" not in selected
    assert "aigc_detector.predict_ensemble" in candidate
    assert "model_v9.pt" in candidate
