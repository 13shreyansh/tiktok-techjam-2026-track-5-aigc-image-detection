from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

from .data import SUPPORTED_SUFFIXES
from .device import select_device
from .models import PARAMETER_LIMIT, create_binary_model, parameter_summary
from .transforms import evaluation_transform


V6_WEIGHT = 0.75
V9_WEIGHT = 0.25
PHYSICAL_BATCH_SIZE = 64
V6_SHA256 = "48ea50773fbd1b7247fff25fde6f985183e29f2eb517b5ac0f6319c1fe38b644"
V9_SHA256 = "dd6b26c7849489447c7e96823f5b5e87c31623ca8ae0d28a5b162bb2dcb65075"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict AI-generated confidence with the promoted v6/v9 ensemble"
    )
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("--v6-checkpoint", type=Path, required=True)
    parser.add_argument("--v9-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(path: Path, expected_sha256: str) -> dict:
    if not path.is_file():
        raise SystemExit(f"checkpoint is missing: {path}")
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise SystemExit(
            f"checkpoint SHA-256 mismatch for {path}: expected "
            f"{expected_sha256}, observed {observed}"
        )
    return torch.load(path, map_location="cpu", weights_only=True)


def shared_inference_config(first: dict, second: dict) -> dict:
    keys = (
        "model_name",
        "image_size",
        "head_mode",
        "normalization_mean",
        "normalization_std",
        "preprocess_mode",
        "codec_normalization",
    )
    normalized = {}
    for key in keys:
        left = first.get(key)
        right = second.get(key)
        if left != right:
            raise SystemExit(f"ensemble checkpoint configuration mismatch: {key}")
        normalized[key] = left
    return normalized


def load_model(checkpoint: dict, device: torch.device) -> torch.nn.Module:
    model = create_binary_model(
        checkpoint["model_name"],
        pretrained=False,
        image_size=int(checkpoint["image_size"]),
        head_mode=checkpoint.get("head_mode", "linear"),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def paths_in(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def image_batch(paths: list[Path], transform) -> torch.Tensor:
    tensors = []
    for path in paths:
        with Image.open(path) as handle:
            tensors.append(transform(handle.convert("RGB")))
    images = torch.stack(tensors)
    if len(paths) < PHYSICAL_BATCH_SIZE:
        images = torch.cat(
            [
                images,
                images[-1:].repeat(PHYSICAL_BATCH_SIZE - len(paths), 1, 1, 1),
            ],
            dim=0,
        )
    if int(images.shape[0]) != PHYSICAL_BATCH_SIZE:
        raise RuntimeError(f"physical batch mismatch: {tuple(images.shape)}")
    return images


def main() -> None:
    args = parse_args()
    if not args.image_directory.is_dir():
        raise SystemExit(f"not a readable directory: {args.image_directory}")
    device = select_device(args.device)
    if device.type != "cuda":
        raise SystemExit(
            "the promoted ensemble requires CUDA to preserve its verified "
            "physical-batch-64 FP16 inference contract; use run.sh for the "
            "verified single-model CPU/MPS fallback"
        )
    v6_checkpoint = validate_checkpoint(args.v6_checkpoint, V6_SHA256)
    v9_checkpoint = validate_checkpoint(args.v9_checkpoint, V9_SHA256)
    config = shared_inference_config(v6_checkpoint, v9_checkpoint)
    v6_model = load_model(v6_checkpoint, device)
    v9_model = load_model(v9_checkpoint, device)
    total_parameters = (
        parameter_summary(v6_model)["total"]
        + parameter_summary(v9_model)["total"]
    )
    if total_parameters >= PARAMETER_LIMIT:
        raise SystemExit(
            f"ensemble has {total_parameters:,} parameters; organizer limit is "
            f"fewer than {PARAMETER_LIMIT:,}"
        )
    transform = evaluation_transform(
        int(config["image_size"]),
        mean=tuple(config["normalization_mean"]),
        std=tuple(config["normalization_std"]),
        preprocess_mode=config.get("preprocess_mode") or "stretch",
        codec_normalization=config.get("codec_normalization") or "none",
    )
    paths = paths_in(args.image_directory)
    predictions = []
    with torch.inference_mode():
        for offset in range(0, len(paths), PHYSICAL_BATCH_SIZE):
            logical_paths = paths[offset : offset + PHYSICAL_BATCH_SIZE]
            images = image_batch(logical_paths, transform).to(
                device, non_blocking=True
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                v6_scores = torch.sigmoid(v6_model(images).flatten())
                v9_scores = torch.sigmoid(v9_model(images).flatten())
                blend_scores = V6_WEIGHT * v6_scores + V9_WEIGHT * v9_scores
            for path, probability in zip(
                logical_paths, blend_scores[: len(logical_paths)].float().cpu()
            ):
                predictions.append(
                    {"image_path": str(path), "pred": float(probability)}
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(predictions, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "images": len(predictions),
                "total_parameters": total_parameters,
                "device": str(device),
                "physical_batch_size": PHYSICAL_BATCH_SIZE,
                "weights": {"v6": V6_WEIGHT, "v9": V9_WEIGHT},
                "arithmetic": "sigmoid_and_probability_blend_in_cuda_fp16",
            }
        )
    )


if __name__ == "__main__":
    main()
