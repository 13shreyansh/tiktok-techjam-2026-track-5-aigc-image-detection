from pathlib import Path
import sys

sys.path.insert(0, str(Path("scripts").resolve()))
import kaggle_train_v12_permissive as v12  # noqa: E402
import kaggle_train_v12_dino_control as dino  # noqa: E402


def test_v12_sampler_masses_are_balanced_and_block_based() -> None:
    assert sum(v12.REAL_DATASET_MASS.values()) == 1.0
    assert sum(v12.FAKE_DATASET_MASS.values()) == 1.0
    assert v12.FAKE_DATASET_MASS["Qwen-Image-Bench"] == 0.20


def test_v12_forces_label_independent_codec_normalization() -> None:
    assert v12.runner.CODEC_NORMALIZATION == "jpeg_q96"
    assert v12.runner.PREPROCESS_MODE == "short_side_crop"


def test_v12_package_identity_is_frozen() -> None:
    assert v12.runner.PACKAGE_NAME == "permissive-mixture-v12-canonical.zip"
    assert len(v12.runner.EXPECTED_ZIP_SHA256) == 64
    assert len(v12.runner.EXPECTED_INVENTORY_SHA256) == 64
    assert "pending" not in v12.runner.EXPECTED_ZIP_SHA256
    assert "pending" not in v12.runner.EXPECTED_INVENTORY_SHA256


def test_v12_dino_control_is_a_separate_frozen_output() -> None:
    assert dino.MODEL_NAME == "vit_large_patch14_dinov2.lvd142m"
    assert dino.OUTPUT_ROOT.name == "track5-v12-dino-control"
    assert dino.OUTPUT_ROOT != v12.runner.OUTPUT_ROOT


def test_v12_promotion_floors_are_frozen() -> None:
    assert v12.PROMOTION_FLOORS == {
        "clean_auc": 0.85,
        "worst_fake_generator_auc": 0.65,
        "worst_real_source_auc": 0.65,
        "worst_generator_real_source_pair_auc": 0.60,
    }


def test_v12_rejects_noncommercial_and_demo_rows() -> None:
    base = {
        "dataset": "CIFAKE",
        "license_commercial_use_allowed": True,
        "organizer_demo_row": False,
        "training_allowed": True,
        "source_license": "MIT",
        "path": "images/example.jpg",
        "label": 0,
        "image_sha256": "a" * 64,
    }
    bad = dict(base, source_license="CC-BY-NC-4.0")
    try:
        v12.validate_rows([bad] * 13574, [base] * 2000)
    except RuntimeError as error:
        assert "noncommercial" in str(error)
    else:
        raise AssertionError("noncommercial row was accepted")
