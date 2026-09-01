from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .data import binary_dataset, source_balanced_weights
from .device import select_device
from .evaluation import evaluate_conditions
from .models import HEAD_MODES, create_binary_model, normalization_config, parameter_summary
from .transforms import CODEC_NORMALIZATION_MODES, PREPROCESS_MODES, training_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Track 5 binary detector experiment")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--dataset-root", type=Path, help="directory containing train/ and test/")
    inputs.add_argument("--train-manifest", type=Path, help="source-aware JSONL training manifest")
    parser.add_argument("--eval-manifest", type=Path, help="source-aware JSONL evaluation manifest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="resnet18.a1_in1k")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--head-mode", choices=HEAD_MODES, default="linear")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-per-class", type=int)
    parser.add_argument("--max-eval-per-class", type=int)
    parser.add_argument("--augmentation", choices=("standard", "robust"), default="standard")
    parser.add_argument(
        "--sampling",
        choices=("random", "source-balanced"),
        default="random",
    )
    parser.add_argument("--eval-preprocess", choices=PREPROCESS_MODES, default="stretch")
    parser.add_argument(
        "--codec-normalization",
        choices=CODEC_NORMALIZATION_MODES,
        default="none",
        help="optional label-independent codec normalization applied to every image",
    )
    parser.add_argument("--robust-eval", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resource_snapshot(device: torch.device) -> dict[str, int | str]:
    payload: dict[str, int | str] = {"device": str(device)}
    if device.type == "cuda":
        payload["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    elif device.type == "mps":
        payload["current_allocated_bytes"] = int(torch.mps.current_allocated_memory())
        payload["driver_allocated_bytes"] = int(torch.mps.driver_allocated_memory())
    return payload


def main() -> None:
    args = parse_args()
    if args.train_manifest is not None and args.eval_manifest is None:
        raise SystemExit("--train-manifest requires --eval-manifest")
    if args.dataset_root is not None and args.eval_manifest is not None:
        raise SystemExit("--eval-manifest is only valid with --train-manifest")
    set_seed(args.seed)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = create_binary_model(
        args.model,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        image_size=args.image_size,
        head_mode=args.head_mode,
    ).to(device)
    parameters = parameter_summary(model)
    normalization = normalization_config(model)
    train_source = args.train_manifest or (args.dataset_root / "train")
    eval_source = args.eval_manifest or (args.dataset_root / "test")
    train_dataset = binary_dataset(
        train_source,
        transform=training_transform(
            args.image_size,
            args.augmentation,
            mean=normalization["mean"],
            std=normalization["std"],
            preprocess_mode=args.eval_preprocess,
            codec_normalization=args.codec_normalization,
        ),
        max_per_class=args.max_train_per_class,
        seed=args.seed,
    )
    sampling_report = {"policy": "uniform random rows"}
    sampler = None
    if args.sampling == "source-balanced":
        if args.train_manifest is None:
            raise SystemExit("--sampling source-balanced requires --train-manifest")
        weights, sampling_report = source_balanced_weights(
            args.train_manifest, train_dataset.samples
        )
        generator = torch.Generator().manual_seed(args.seed)
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_function = torch.nn.BCEWithLogitsLoss()
    started = time.perf_counter()
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for images, labels, _ in progress:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images).flatten()
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            if args.head_mode == "stay_positive":
                model.clamp_classifier_weights()
            count = labels.numel()
            seen += count
            running_loss += float(loss.detach().cpu()) * count
            progress.set_postfix(loss=f"{running_loss / seen:.4f}")

        checkpoint = {
            "format_version": 1,
            "model_name": args.model,
            "image_size": args.image_size,
            "state_dict": model.state_dict(),
            "parameters": parameters,
            "positive_class": "AI-generated",
            "normalization_mean": normalization["mean"],
            "normalization_std": normalization["std"],
            "preprocess_mode": args.eval_preprocess,
            "codec_normalization": args.codec_normalization,
            "head_mode": args.head_mode,
        }
        checkpoint_path = args.output_dir / "model.pt"
        torch.save(checkpoint, checkpoint_path)
        print(
            json.dumps(
                {
                    "phase": "checkpoint_saved",
                    "epoch": epoch,
                    "path": str(checkpoint_path),
                    "train_loss": running_loss / seen,
                }
            ),
            flush=True,
        )

        evaluation = evaluate_conditions(
            model=model,
            dataset_root=eval_source,
            device=device,
            image_size=args.image_size,
            batch_size=args.batch_size,
            workers=args.workers,
            max_per_class=args.max_eval_per_class,
            seed=args.seed,
            robust=args.robust_eval,
            mean=normalization["mean"],
            std=normalization["std"],
            preprocess_mode=args.eval_preprocess,
            codec_normalization=args.codec_normalization,
        )
        history.append({"epoch": epoch, "train_loss": running_loss / seen, "evaluation": evaluation})

    elapsed = time.perf_counter() - started
    report = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "parameters": parameters,
        "normalization": normalization,
        "sampling": sampling_report,
        "train_samples": len(train_dataset),
        "elapsed_seconds": elapsed,
        "resource": resource_snapshot(device),
        "history": history,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
