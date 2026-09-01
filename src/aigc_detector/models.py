from __future__ import annotations

import timm
import torch

PARAMETER_LIMIT = 2_000_000_000
HEAD_MODES = ("linear", "stay_positive")


class StayPositiveModel(torch.nn.Module):
    """Frozen public feature extractor with a non-negative fake-evidence head.

    The ReLU before the final layer and the non-negative final weights implement
    the structural assumptions of Stay-Positive without copying upstream code.
    """

    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = torch.nn.Linear(int(backbone.num_features), 1)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        self.pretrained_cfg = getattr(backbone, "pretrained_cfg", {})

    def get_classifier(self) -> torch.nn.Module:
        return self.head

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(inputs)
        embedding = self.backbone.forward_head(features, pre_logits=True)
        return self.head(torch.relu(embedding))

    @torch.no_grad()
    def clamp_classifier_weights(self) -> None:
        self.head.weight.clamp_(min=0.0)


def create_binary_model(
    name: str,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    image_size: int | None = None,
    head_mode: str = "linear",
) -> torch.nn.Module:
    if head_mode not in HEAD_MODES:
        raise ValueError(f"unknown head mode: {head_mode}")
    model_kwargs = {"num_classes": 0 if head_mode == "stay_positive" else 1}
    if image_size is not None and name.startswith("vit_"):
        model_kwargs["img_size"] = image_size
    backbone_or_model = timm.create_model(name, pretrained=pretrained, **model_kwargs)
    model = (
        StayPositiveModel(backbone_or_model)
        if head_mode == "stay_positive"
        else backbone_or_model
    )
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        classifier = model.get_classifier()
        for parameter in classifier.parameters():
            parameter.requires_grad = True
    count = sum(parameter.numel() for parameter in model.parameters())
    if count >= PARAMETER_LIMIT:
        raise ValueError(f"model has {count:,} parameters; organizer limit is fewer than {PARAMETER_LIMIT:,}")
    return model


def parameter_summary(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def normalization_config(model: torch.nn.Module) -> dict[str, tuple[float, float, float]]:
    """Return the normalization expected by the selected public backbone."""
    config = getattr(model, "pretrained_cfg", {})
    mean = tuple(float(value) for value in config.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(float(value) for value in config.get("std", (0.229, 0.224, 0.225)))
    if len(mean) != 3 or len(std) != 3:
        raise ValueError(f"expected three-channel normalization, got mean={mean}, std={std}")
    return {"mean": mean, "std": std}
