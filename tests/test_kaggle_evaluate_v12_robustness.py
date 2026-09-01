import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kaggle_evaluate_v12_robustness as evaluation  # noqa: E402


def test_workshop_matrix_contains_clean_plus_exact_19_conditions():
    names = [name for name, _ in evaluation.conditions()]
    assert len(names) == 20
    assert len(set(names)) == 20
    assert names == [
        "clean",
        "jpeg_q90",
        "jpeg_q70",
        "jpeg_q50",
        "jpeg_q30",
        "blur_sigma_0.5",
        "blur_sigma_1",
        "blur_sigma_2",
        "resize_0.5",
        "resize_0.25",
        "noise_sigma_0.02",
        "noise_sigma_0.05",
        "noise_sigma_0.10",
        "brightness_0.8",
        "brightness_1.2",
        "contrast_0.8",
        "contrast_1.2",
        "saturation_0.8",
        "saturation_1.2",
        "center_crop_80",
    ]


def test_condition_transform_always_finishes_with_label_independent_jpeg_q96():
    _, official_transform = evaluation.conditions()[1]
    pipeline = evaluation.condition_transform(
        224,
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
        official_transform,
    )
    assert pipeline.transforms[0] is official_transform
    assert isinstance(pipeline.transforms[1], evaluation.runner.JpegCompression)
    assert pipeline.transforms[1].quality == 96
