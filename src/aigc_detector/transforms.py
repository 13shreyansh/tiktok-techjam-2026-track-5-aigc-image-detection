from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Callable

import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PREPROCESS_MODES = ("stretch", "short_side_crop")
CODEC_NORMALIZATION_MODES = ("none", "jpeg_q96")
INFERENCE_POLICIES = ("reference", "reference_flip_mean")


class JpegCompression:
    def __init__(self, quality: int) -> None:
        self.quality = quality

    def __call__(self, image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")


class DownUpResize:
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        down = (max(1, round(width * self.scale)), max(1, round(height * self.scale)))
        return image.resize(down, Image.Resampling.BICUBIC).resize((width, height), Image.Resampling.BICUBIC)


class CenterCropFraction:
    def __init__(self, fraction: float) -> None:
        self.fraction = fraction

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        crop_w, crop_h = max(1, round(width * self.fraction)), max(1, round(height * self.fraction))
        left, top = (width - crop_w) // 2, (height - crop_h) // 2
        return image.crop((left, top, left + crop_w, top + crop_h))


class GaussianBlurPIL:
    """Pickle-safe PIL blur used by multi-worker data loaders."""

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(self.sigma))


class FixedEnhancement:
    def __init__(self, kind: str, factor: float) -> None:
        self.kind = kind
        self.factor = factor

    def __call__(self, image: Image.Image) -> Image.Image:
        enhancer = {
            "brightness": ImageEnhance.Brightness,
            "contrast": ImageEnhance.Contrast,
            "saturation": ImageEnhance.Color,
        }[self.kind]
        return enhancer(image).enhance(self.factor)


class GaussianNoise(torch.nn.Module):
    def __init__(self, sigma: float) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor + torch.randn_like(tensor) * self.sigma).clamp(0.0, 1.0)


class GaussianNoisePIL:
    """Apply Gaussian noise while returning PIL so it composes with image transforms."""

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma
        self.to_image = v2.ToImage()
        self.to_float = v2.ToDtype(torch.float32, scale=True)
        self.to_pil = v2.ToPILImage()

    def __call__(self, image: Image.Image) -> Image.Image:
        tensor = self.to_float(self.to_image(image))
        return self.to_pil((tensor + torch.randn_like(tensor) * self.sigma).clamp(0.0, 1.0))


class ReferenceFlipViews:
    """Return reference and horizontally flipped views as one tensor."""

    def __init__(self, transform: Callable[[Image.Image], torch.Tensor]) -> None:
        self.transform = transform

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return torch.stack(
            [self.transform(image), self.transform(ImageOps.mirror(image))]
        )


class RandomSingleOfficialTransform:
    """Apply at most one workshop-listed redistribution to a training image."""

    def __init__(self, probability: float = 0.8) -> None:
        self.probability = probability
        self.transforms = [
            JpegCompression(quality) for quality in (90, 70, 50, 30)
        ] + [GaussianBlurPIL(sigma) for sigma in (0.5, 1.0, 2.0)] + [
            DownUpResize(scale) for scale in (0.5, 0.25)
        ] + [
            GaussianNoisePIL(sigma) for sigma in (0.02, 0.05, 0.10)
        ] + [
            FixedEnhancement(kind, factor)
            for kind in ("brightness", "contrast", "saturation")
            for factor in (0.8, 1.2)
        ] + [
            CenterCropFraction(0.8),
        ]

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        return random.choice(self.transforms)(image)


@dataclass(frozen=True)
class Condition:
    name: str
    image_transform: Callable[[Image.Image], Image.Image] | None = None
    tensor_transform: Callable[[torch.Tensor], torch.Tensor] | None = None


def official_conditions() -> list[Condition]:
    conditions = [Condition("clean")]
    conditions += [Condition(f"jpeg_q{q}", JpegCompression(q)) for q in (90, 70, 50, 30)]
    conditions += [Condition(f"blur_sigma_{sigma:g}", GaussianBlurPIL(sigma)) for sigma in (0.5, 1.0, 2.0)]
    conditions += [Condition(f"resize_{scale:g}", DownUpResize(scale)) for scale in (0.5, 0.25)]
    conditions += [Condition(f"noise_sigma_{sigma:.2f}", tensor_transform=GaussianNoise(sigma)) for sigma in (0.02, 0.05, 0.10)]
    for kind in ("brightness", "contrast", "saturation"):
        conditions += [Condition(f"{kind}_{factor:g}", FixedEnhancement(kind, factor)) for factor in (0.8, 1.2)]
    conditions.append(Condition("center_crop_80", CenterCropFraction(0.8)))
    return conditions


def evaluation_transform(
    image_size: int,
    condition: Condition | None = None,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    preprocess_mode: str = "stretch",
    codec_normalization: str = "none",
):
    if preprocess_mode not in PREPROCESS_MODES:
        raise ValueError(f"unknown preprocessing mode: {preprocess_mode}")
    if codec_normalization not in CODEC_NORMALIZATION_MODES:
        raise ValueError(f"unknown codec normalization: {codec_normalization}")
    condition = condition or Condition("clean")
    operations = []
    if condition.image_transform is not None:
        operations.append(condition.image_transform)
    if codec_normalization == "jpeg_q96":
        operations.append(JpegCompression(96))
    if preprocess_mode == "stretch":
        operations.append(v2.Resize((image_size, image_size), antialias=True))
    else:
        operations.extend(
            [v2.Resize(image_size, antialias=True), v2.CenterCrop((image_size, image_size))]
        )
    operations.extend([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    if condition.tensor_transform is not None:
        operations.append(condition.tensor_transform)
    operations.append(v2.Normalize(mean, std))
    return v2.Compose(operations)


def evaluation_inference_transform(
    image_size: int,
    condition: Condition | None = None,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    preprocess_mode: str = "stretch",
    codec_normalization: str = "none",
    inference_policy: str = "reference",
):
    """Build the exact reference transform or a predeclared view ensemble."""
    if inference_policy not in INFERENCE_POLICIES:
        raise ValueError(f"unknown inference policy: {inference_policy}")
    reference = evaluation_transform(
        image_size,
        condition,
        mean=mean,
        std=std,
        preprocess_mode=preprocess_mode,
        codec_normalization=codec_normalization,
    )
    if inference_policy == "reference":
        return reference
    return ReferenceFlipViews(reference)


def training_transform(
    image_size: int,
    augmentation: str = "standard",
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    preprocess_mode: str = "stretch",
    codec_normalization: str = "none",
):
    if augmentation not in {"standard", "robust"}:
        raise ValueError(f"unknown augmentation profile: {augmentation}")
    if preprocess_mode not in PREPROCESS_MODES:
        raise ValueError(f"unknown preprocessing mode: {preprocess_mode}")
    if codec_normalization not in CODEC_NORMALIZATION_MODES:
        raise ValueError(f"unknown codec normalization: {codec_normalization}")
    operations = []
    if augmentation == "robust":
        operations.append(RandomSingleOfficialTransform())
    if codec_normalization == "jpeg_q96":
        operations.append(JpegCompression(96))
    if preprocess_mode == "stretch":
        operations.append(v2.Resize((image_size, image_size), antialias=True))
    else:
        operations.extend(
            [v2.Resize(image_size, antialias=True), v2.CenterCrop((image_size, image_size))]
        )
    operations.extend(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    operations.append(v2.Normalize(mean, std))
    return v2.Compose(operations)
