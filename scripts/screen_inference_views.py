#!/usr/bin/env python3
"""Screen a fixed set of reversible inference-view policies.

This is deliberately separate from the submission runner. It cannot silently
change the selected model; it writes an evidence report which must pass the
predeclared gates in DEADLINE_ADVERSARIAL_PLAN.md first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageOps

from aigc_detector.data import binary_dataset
from aigc_detector.device import select_device
from aigc_detector.metrics import auc
from aigc_detector.models import create_binary_model, parameter_summary
from aigc_detector.transforms import (
    Condition,
    GaussianBlurPIL,
    evaluation_transform,
    official_conditions,
)


POLICY_WEIGHTS = {
    "reference": {"reference": 1.0},
    "reference_flip_mean": {"reference": 0.5, "flip": 0.5},
    "reference_stretch_mean": {"reference": 0.5, "stretch": 0.5},
    "reference_blur_blend": {"reference": 0.75, "mild_blur": 0.25},
}

SCREEN_CONDITIONS = {
    "clean",
    "jpeg_q30",
    "blur_sigma_2",
    "resize_0.25",
    "noise_sigma_0.10",
    "brightness_0.8",
    "contrast_0.8",
    "saturation_0.8",
    "center_crop_80",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SequentialImageTransform:
    def __init__(self, *transforms) -> None:
        self.transforms = transforms

    def __call__(self, image: Image.Image) -> Image.Image:
        for transform in self.transforms:
            image = transform(image)
        return image


def mild_blur_condition(condition: Condition) -> Condition:
    blur = GaussianBlurPIL(0.5)
    image_transform = (
        blur
        if condition.image_transform is None
        else SequentialImageTransform(condition.image_transform, blur)
    )
    return Condition(
        name=f"{condition.name}_plus_blur_sigma_0.5",
        image_transform=image_transform,
        tensor_transform=condition.tensor_transform,
    )


@torch.inference_mode()
def predict_one(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
) -> float:
    probability = torch.sigmoid(model(tensor.unsqueeze(0).to(device)).flatten()[0])
    return float(probability.cpu())


def promotion_assessment(policy_results: dict[str, dict[str, float]]) -> dict:
    reference = policy_results["reference"]
    assessments = {}
    for policy, condition_aucs in policy_results.items():
        if policy == "reference":
            continue
        clean_delta = condition_aucs["clean"] - reference["clean"]
        stress_names = sorted(set(condition_aucs) - {"clean"})
        stress_deltas = {
            name: condition_aucs[name] - reference[name] for name in stress_names
        }
        selected_official = 0.5 * (
            condition_aucs["clean"]
            + sum(condition_aucs[name] for name in stress_names) / len(stress_names)
        )
        reference_official = 0.5 * (
            reference["clean"]
            + sum(reference[name] for name in stress_names) / len(stress_names)
        )
        checks = {
            "selected_official_improved": selected_official > reference_official,
            "clean_delta_at_least_minus_0.002": clean_delta >= -0.002,
            "all_stress_deltas_at_least_minus_0.01": min(stress_deltas.values()) >= -0.01,
            "noise_sigma_0.10_improved": stress_deltas["noise_sigma_0.10"] > 0.0,
        }
        assessments[policy] = {
            "reference_selected_official": reference_official,
            "selected_official": selected_official,
            "selected_official_delta": selected_official - reference_official,
            "clean_delta": clean_delta,
            "stress_deltas": stress_deltas,
            "checks": checks,
            "passes_screen": all(checks.values()),
        }
    return assessments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = create_binary_model(
        checkpoint["model_name"],
        pretrained=False,
        image_size=int(checkpoint["image_size"]),
        head_mode=checkpoint.get("head_mode", "linear"),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    image_size = int(checkpoint["image_size"])
    mean = tuple(checkpoint.get("normalization_mean", (0.485, 0.456, 0.406)))
    std = tuple(checkpoint.get("normalization_std", (0.229, 0.224, 0.225)))
    reference_preprocess = checkpoint.get("preprocess_mode", "stretch")
    codec_normalization = checkpoint.get("codec_normalization", "none")
    dataset = binary_dataset(
        args.manifest,
        transform=None,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )
    conditions = [
        condition
        for condition in official_conditions()
        if condition.name in SCREEN_CONDITIONS
    ]
    if {condition.name for condition in conditions} != SCREEN_CONDITIONS:
        raise RuntimeError("screen condition list no longer matches official conditions")

    labels: list[float] = [float(label) for _, label in dataset.samples]
    paths = [str(path) for path, _ in dataset.samples]
    scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    started = time.perf_counter()
    for condition_index, condition in enumerate(conditions):
        transforms = {
            "reference": evaluation_transform(
                image_size,
                condition,
                mean=mean,
                std=std,
                preprocess_mode=reference_preprocess,
                codec_normalization=codec_normalization,
            ),
            "flip": evaluation_transform(
                image_size,
                condition,
                mean=mean,
                std=std,
                preprocess_mode=reference_preprocess,
                codec_normalization=codec_normalization,
            ),
            "stretch": evaluation_transform(
                image_size,
                condition,
                mean=mean,
                std=std,
                preprocess_mode="stretch",
                codec_normalization=codec_normalization,
            ),
            "mild_blur": evaluation_transform(
                image_size,
                mild_blur_condition(condition),
                mean=mean,
                std=std,
                preprocess_mode=reference_preprocess,
                codec_normalization=codec_normalization,
            ),
        }
        for sample_index, (path, _) in enumerate(dataset.samples):
            with Image.open(path) as handle:
                image = handle.convert("RGB")
            view_probabilities = {}
            deterministic_seed = args.seed + condition_index * 100_000 + sample_index
            for view_name, transform in transforms.items():
                torch.manual_seed(deterministic_seed)
                view_image = ImageOps.mirror(image) if view_name == "flip" else image
                view_probabilities[view_name] = predict_one(
                    model, transform(view_image), device
                )
            for policy, weights in POLICY_WEIGHTS.items():
                scores[policy][condition.name].append(
                    sum(view_probabilities[view] * weight for view, weight in weights.items())
                )
        print(
            json.dumps(
                {
                    "phase": "condition_complete",
                    "condition": condition.name,
                    "samples": len(dataset.samples),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )

    policy_results = {
        policy: {
            condition.name: auc(labels, scores[policy][condition.name])
            for condition in conditions
        }
        for policy in POLICY_WEIGHTS
    }
    output = {
        "format_version": 1,
        "status": "screen_only_not_promoted",
        "predeclaration": "DEADLINE_ADVERSARIAL_PLAN.md experiment A",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "device": str(device),
        "parameters": parameter_summary(model),
        "seed": args.seed,
        "max_per_class": args.max_per_class,
        "sample_count": len(dataset.samples),
        "paths": paths,
        "elapsed_seconds": time.perf_counter() - started,
        "policy_weights": POLICY_WEIGHTS,
        "condition_aucs": policy_results,
        "promotion_assessment": promotion_assessment(policy_results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
