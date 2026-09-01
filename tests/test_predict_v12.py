from pathlib import Path

import pytest
import torch
from PIL import Image

from aigc_detector.predict_v12 import (
    DEFAULT_BATCH_SIZE,
    DINO,
    DINO_WEIGHT,
    PE_CORE,
    PE_WEIGHT,
    combine_scores,
    image_batch,
)


def test_v12_candidate_hashes_and_blend_are_frozen():
    assert PE_CORE.sha256 == "f37bd6b445b12257ff29a9e54946c5bf9a9184e86a45dc5490537b9ea325ddd2"
    assert DINO.sha256 == "db07f30cbc94e4972f4a8c72c95bbe5df0dcd40b2dbd494ad82dc324cc1e2b5b"
    assert PE_WEIGHT == DINO_WEIGHT == 0.5
    assert DEFAULT_BATCH_SIZE == 1


def test_float32_probability_blend_is_exact_and_checks_count():
    assert combine_scores([0.1, 0.9], [0.3, 0.5]) == pytest.approx([0.2, 0.7])
    with pytest.raises(RuntimeError, match="count mismatch"):
        combine_scores([0.1], [])


def test_v12_runner_is_separate_and_has_single_model_fallback():
    selected = Path("run.sh").read_text()
    v12 = Path("run_v12.sh").read_text()
    source = Path("src/aigc_detector/predict_v12.py").read_text()
    assert "aigc_detector.predict_v12" not in selected
    assert "aigc_detector.predict_v12" in v12
    assert "AIGC_V12_MODE" in v12
    assert 'AIGC_V12_MODE:-pe_core' in v12
    assert 'default="pe_core"' in source
    assert 'choices=("blend", "pe_core")' in source
    assert "sequential_models_to_reduce_peak_device_memory" in source
    assert '"jpeg_q96"' in source


def test_v12_input_applies_exif_orientation_before_preprocessing(tmp_path):
    path = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (3, 2), "white")
    exif = image.getexif()
    exif[274] = 6  # Rotate 90 degrees clockwise for display.
    image.save(path, exif=exif)

    batch = image_batch(
        [path],
        lambda oriented: torch.tensor(oriented.size, dtype=torch.int64),
    )

    assert batch.tolist() == [[2, 3]]
