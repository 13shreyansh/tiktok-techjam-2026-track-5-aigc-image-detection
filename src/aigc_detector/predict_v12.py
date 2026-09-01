from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps

from .data import SUPPORTED_SUFFIXES
from .device import select_device
from .models import PARAMETER_LIMIT, create_binary_model, parameter_summary
from .transforms import evaluation_transform


PE_WEIGHT = 0.5
DINO_WEIGHT = 0.5
DEFAULT_BATCH_SIZE = 1


@dataclass(frozen=True)
class Candidate:
    key: str
    model_name: str
    sha256: str


PE_CORE = Candidate(
    key="pe_core",
    model_name="vit_pe_core_large_patch14_336",
    sha256="f37bd6b445b12257ff29a9e54946c5bf9a9184e86a45dc5490537b9ea325ddd2",
)
DINO = Candidate(
    key="dinov2_control",
    model_name="vit_large_patch14_dinov2.lvd142m",
    sha256="db07f30cbc94e4972f4a8c72c95bbe5df0dcd40b2dbd494ad82dc324cc1e2b5b",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict AI-generated confidence with the compliant v12 detector"
    )
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("--pe-checkpoint", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=("blend", "pe_core"), default="pe_core")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(path: Path, candidate: Candidate) -> dict:
    if not path.is_file():
        raise SystemExit(f"{candidate.key} checkpoint is missing: {path}")
    observed = file_sha256(path)
    if observed != candidate.sha256:
        raise SystemExit(
            f"{candidate.key} checkpoint SHA-256 mismatch: expected "
            f"{candidate.sha256}, observed {observed}"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "model_name": candidate.model_name,
        "image_size": 224,
        "head_mode": "linear",
        "preprocess_mode": "short_side_crop",
        "codec_normalization": "jpeg_q96",
    }
    for key, value in expected.items():
        observed_value = checkpoint.get(key, "linear" if key == "head_mode" else None)
        if observed_value != value:
            raise SystemExit(
                f"{candidate.key} checkpoint contract mismatch for {key}: "
                f"expected {value!r}, observed {observed_value!r}"
            )
    for key in ("normalization_mean", "normalization_std", "state_dict"):
        if key not in checkpoint:
            raise SystemExit(f"{candidate.key} checkpoint is missing {key}")
    return checkpoint


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
            # Match the v12 canonical training contract for phone/camera files
            # that store orientation in EXIF instead of rotating the pixels.
            image = ImageOps.exif_transpose(handle).convert("RGB")
            tensors.append(transform(image))
    return torch.stack(tensors)


def release_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def predict_candidate(
    candidate: Candidate,
    checkpoint_path: Path,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
) -> tuple[list[float], int]:
    checkpoint = validate_checkpoint(checkpoint_path, candidate)
    model = create_binary_model(
        checkpoint["model_name"],
        pretrained=False,
        image_size=int(checkpoint["image_size"]),
        head_mode=checkpoint.get("head_mode", "linear"),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    parameters = parameter_summary(model)["total"]
    transform = evaluation_transform(
        int(checkpoint["image_size"]),
        mean=tuple(checkpoint["normalization_mean"]),
        std=tuple(checkpoint["normalization_std"]),
        preprocess_mode=checkpoint["preprocess_mode"],
        codec_normalization=checkpoint["codec_normalization"],
    )
    del checkpoint
    scores: list[float] = []
    with torch.inference_mode():
        for offset in range(0, len(paths), batch_size):
            batch_paths = paths[offset : offset + batch_size]
            images = image_batch(batch_paths, transform).to(device)
            logits = model(images).flatten()
            scores.extend(float(value) for value in torch.sigmoid(logits).cpu())
    del model
    release_device_cache(device)
    return scores, parameters


def combine_scores(pe_scores: list[float], dino_scores: list[float]) -> list[float]:
    if len(pe_scores) != len(dino_scores):
        raise RuntimeError("candidate score count mismatch")
    return [
        PE_WEIGHT * pe + DINO_WEIGHT * dino
        for pe, dino in zip(pe_scores, dino_scores)
    ]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if not args.image_directory.is_dir():
        raise SystemExit(f"not a readable directory: {args.image_directory}")
    if args.batch_size < 1:
        raise SystemExit("batch size must be positive")
    if args.mode == "blend" and args.dino_checkpoint is None:
        raise SystemExit("blend mode requires --dino-checkpoint")

    device = select_device(args.device)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    paths = paths_in(args.image_directory)
    pe_scores, pe_parameters = predict_candidate(
        PE_CORE, args.pe_checkpoint, paths, device, args.batch_size
    )
    total_parameters = pe_parameters
    if args.mode == "blend":
        assert args.dino_checkpoint is not None
        dino_scores, dino_parameters = predict_candidate(
            DINO, args.dino_checkpoint, paths, device, args.batch_size
        )
        scores = combine_scores(pe_scores, dino_scores)
        total_parameters += dino_parameters
    else:
        scores = pe_scores
    if total_parameters >= PARAMETER_LIMIT:
        raise SystemExit(
            f"selected detector has {total_parameters:,} parameters; organizer "
            f"limit is fewer than {PARAMETER_LIMIT:,}"
        )
    predictions = [
        {"image_path": str(path), "pred": score}
        for path, score in zip(paths, scores)
    ]
    atomic_json(args.output, predictions)
    print(
        json.dumps(
            {
                "images": len(predictions),
                "mode": args.mode,
                "device": str(device),
                "batch_size": args.batch_size,
                "total_parameters": total_parameters,
                "checkpoint_sha256": {
                    "pe_core": PE_CORE.sha256,
                    **({"dinov2_control": DINO.sha256} if args.mode == "blend" else {}),
                },
                "blend_weights": (
                    {"pe_core": PE_WEIGHT, "dinov2_control": DINO_WEIGHT}
                    if args.mode == "blend"
                    else None
                ),
                "arithmetic": "float32_sigmoid_then_probability_blend",
                "loading_policy": "sequential_models_to_reduce_peak_device_memory",
                "elapsed_seconds": time.perf_counter() - started,
                "cuda_peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated() if device.type == "cuda" else None
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
