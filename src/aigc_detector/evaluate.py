from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .device import select_device
from .evaluation import evaluate_conditions
from .models import create_binary_model, parameter_summary
from .transforms import (
    CODEC_NORMALIZATION_MODES,
    INFERENCE_POLICIES,
    PREPROCESS_MODES,
    official_conditions,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with Track 5 conditions")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--dataset-root", type=Path, help="binary REAL/FAKE directory")
    inputs.add_argument("--manifest", type=Path, help="source-aware JSONL manifest")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-per-class", type=int)
    parser.add_argument("--robust", action="store_true")
    parser.add_argument("--preprocess-mode", choices=PREPROCESS_MODES)
    parser.add_argument("--codec-normalization", choices=CODEC_NORMALIZATION_MODES)
    parser.add_argument(
        "--inference-policy", choices=INFERENCE_POLICIES, default="reference"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = create_binary_model(
        checkpoint["model_name"],
        pretrained=False,
        image_size=int(checkpoint["image_size"]),
        head_mode=checkpoint.get("head_mode", "linear"),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    evaluation_source = args.manifest or args.dataset_root
    preprocess_mode = args.preprocess_mode or checkpoint.get("preprocess_mode", "stretch")
    codec_normalization = args.codec_normalization or checkpoint.get("codec_normalization", "none")
    progress_path = Path(str(args.output) + ".progress.json")
    run_signature = {
        "format_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "dataset_source": str(evaluation_source.resolve()),
        "dataset_source_sha256": (
            file_sha256(evaluation_source) if evaluation_source.is_file() else None
        ),
        "model": checkpoint["model_name"],
        "image_size": int(checkpoint["image_size"]),
        "robust": args.robust,
        "conditions": [
            condition.name
            for condition in (
                official_conditions() if args.robust else official_conditions()[:1]
            )
        ],
        "max_per_class": args.max_per_class,
        "seed": args.seed,
        "preprocess_mode": preprocess_mode,
        "codec_normalization": codec_normalization,
        "inference_policy": args.inference_policy,
    }
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("signature") != run_signature:
            raise RuntimeError(
                f"refusing incompatible evaluation resume file: {progress_path}"
            )
    else:
        progress = {
            "signature": run_signature,
            "completed": False,
            "predictions": {},
        }

    def save_condition(condition_name: str, predictions: dict[str, list]) -> None:
        progress["predictions"][condition_name] = predictions
        progress["completed"] = False
        atomic_json_write(progress_path, progress)
        print(
            json.dumps(
                {
                    "phase": "evaluation_condition_saved",
                    "condition": condition_name,
                    "count": len(predictions["labels"]),
                    "progress_path": str(progress_path),
                }
            ),
            flush=True,
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = evaluate_conditions(
        model=model,
        dataset_root=evaluation_source,
        device=device,
        image_size=int(checkpoint["image_size"]),
        batch_size=args.batch_size,
        workers=args.workers,
        max_per_class=args.max_per_class,
        seed=args.seed,
        robust=args.robust,
        mean=tuple(checkpoint.get("normalization_mean", (0.485, 0.456, 0.406))),
        std=tuple(checkpoint.get("normalization_std", (0.229, 0.224, 0.225))),
        preprocess_mode=preprocess_mode,
        codec_normalization=codec_normalization,
        completed_predictions=progress["predictions"],
        prediction_callback=save_condition,
        inference_policy=args.inference_policy,
    )
    elapsed_seconds = time.perf_counter() - started
    resource: dict[str, int | str] = {"device": str(device)}
    if device.type == "cuda":
        resource["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    elif device.type == "mps":
        resource["current_allocated_bytes"] = int(torch.mps.current_allocated_memory())
        resource["driver_allocated_bytes"] = int(torch.mps.driver_allocated_memory())
    payload = {
        "checkpoint": str(args.checkpoint),
        "dataset_source": str(evaluation_source),
        "model": checkpoint["model_name"],
        "parameters_at_training": checkpoint.get("parameters", parameter_summary(model)),
        "total_parameters_at_evaluation": parameter_summary(model)["total"],
        "device": str(device),
        "elapsed_seconds": elapsed_seconds,
        "resource": resource,
        "preprocess_mode": preprocess_mode,
        "codec_normalization": codec_normalization,
        "head_mode": checkpoint.get("head_mode", "linear"),
        "inference_policy": args.inference_policy,
        "evaluation": result,
    }
    atomic_json_write(args.output, payload)
    progress["completed"] = True
    progress["result_output"] = str(args.output.resolve())
    atomic_json_write(progress_path, progress)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
