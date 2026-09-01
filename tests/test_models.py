import torch

from aigc_detector.models import StayPositiveModel, create_binary_model, parameter_summary


def test_stay_positive_head_is_zero_initialized_frozen_and_clamped():
    model = create_binary_model(
        "resnet18.a1_in1k",
        pretrained=False,
        freeze_backbone=True,
        image_size=32,
        head_mode="stay_positive",
    )
    assert isinstance(model, StayPositiveModel)
    assert parameter_summary(model)["trainable"] == 513
    assert torch.count_nonzero(model.head.weight) == 0
    assert tuple(model(torch.zeros(2, 3, 32, 32)).shape) == (2, 1)
    with torch.no_grad():
        model.head.weight.fill_(-1.0)
    model.clamp_classifier_weights()
    assert torch.count_nonzero(model.head.weight) == 0


def test_unknown_head_mode_is_rejected():
    try:
        create_binary_model("resnet18.a1_in1k", pretrained=False, head_mode="unknown")
    except ValueError as error:
        assert "unknown head mode" in str(error)
    else:
        raise AssertionError("unknown head mode was accepted")
