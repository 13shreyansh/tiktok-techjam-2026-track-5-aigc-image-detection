from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .data import SUPPORTED_SUFFIXES
from .device import select_device
from .models import create_binary_model, parameter_summary
from .transforms import evaluation_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict AI-generated confidence for an image directory")
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image_directory.is_dir():
        raise SystemExit(f"not a readable directory: {args.image_directory}")
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
    transform = evaluation_transform(
        int(checkpoint["image_size"]),
        mean=tuple(checkpoint.get("normalization_mean", (0.485, 0.456, 0.406))),
        std=tuple(checkpoint.get("normalization_std", (0.229, 0.224, 0.225))),
        preprocess_mode=checkpoint.get("preprocess_mode", "stretch"),
        codec_normalization=checkpoint.get("codec_normalization", "none"),
    )
    paths = sorted(
        path for path in args.image_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    predictions = []
    with torch.inference_mode():
        for path in paths:
            with Image.open(path) as handle:
                tensor = transform(handle.convert("RGB")).unsqueeze(0).to(device)
            probability = float(torch.sigmoid(model(tensor).flatten()[0]).cpu())
            predictions.append({"image_path": str(path), "pred": probability})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "images": len(predictions),
                "total_parameters": parameter_summary(model)["total"],
                "device": str(device),
            }
        )
    )


if __name__ == "__main__":
    main()
