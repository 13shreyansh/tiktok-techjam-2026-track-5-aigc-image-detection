from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import binary_dataset, discover_manifest_records
from .device import select_device
from .evaluation import predict_loader
from .models import create_binary_model, parameter_summary
from .transforms import CODEC_NORMALIZATION_MODES, PREPROCESS_MODES, evaluation_transform


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_clean_progress_predictions(
    progress_path: Path, manifest_path: Path
) -> tuple[list[float], list[float], list[str], dict]:
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    signature = progress.get("signature", {})
    if signature.get("dataset_source") != str(manifest_path.resolve()):
        raise ValueError("evaluation progress belongs to a different manifest path")
    if signature.get("dataset_source_sha256") != file_sha256(manifest_path):
        raise ValueError("evaluation progress manifest checksum mismatch")
    clean = progress.get("predictions", {}).get("clean")
    if not clean:
        raise ValueError("evaluation progress has no completed clean condition")
    labels = [float(value) for value in clean.get("labels", [])]
    scores = [float(value) for value in clean.get("scores", [])]
    paths = [str(value) for value in clean.get("paths", [])]
    if not labels or not (len(labels) == len(scores) == len(paths)):
        raise ValueError("evaluation progress has invalid clean predictions")
    return labels, scores, paths, signature


def rank_errors(
    labels: list[float],
    scores: list[float],
    paths: list[str],
    metadata: dict[str, dict],
    limit: int,
) -> dict[str, list[dict]]:
    rows = []
    for label, score, path in zip(labels, scores, paths):
        resolved = str(Path(path).resolve())
        row = metadata[resolved]
        rows.append(
            {
                "path": resolved,
                "label": int(label),
                "ai_probability": float(score),
                **{
                    key: row[key]
                    for key in ("generator", "real_source", "family", "archive_member")
                    if key in row
                },
            }
        )
    false_positive_candidates = sorted(
        (row for row in rows if row["label"] == 0),
        key=lambda row: row["ai_probability"],
        reverse=True,
    )
    false_negative_candidates = sorted(
        (row for row in rows if row["label"] == 1),
        key=lambda row: row["ai_probability"],
    )
    return {
        "highest_scoring_reals": false_positive_candidates[:limit],
        "lowest_scoring_fakes": false_negative_candidates[:limit],
        "false_positives_at_0.5": [
            row for row in false_positive_candidates if row["ai_probability"] >= 0.5
        ][:limit],
        "false_negatives_at_0.5": [
            row for row in false_negative_candidates if row["ai_probability"] < 0.5
        ][:limit],
    }


def summarize_errors(
    labels: list[float],
    scores: list[float],
    paths: list[str],
    metadata: dict[str, dict],
    threshold: float = 0.5,
) -> dict:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    if not labels or not (len(labels) == len(scores) == len(paths)):
        raise ValueError("labels, scores and paths must be non-empty and aligned")

    rows = []
    for label, score, path in zip(labels, scores, paths):
        resolved = str(Path(path).resolve())
        record = metadata[resolved]
        rows.append(
            {
                "label": int(label),
                "score": float(score),
                "real_source": record.get("real_source", "unknown"),
                "generator": record.get("generator", "unknown"),
            }
        )

    reals = [row for row in rows if row["label"] == 0]
    fakes = [row for row in rows if row["label"] == 1]
    false_positives = [row for row in reals if row["score"] >= threshold]
    false_negatives = [row for row in fakes if row["score"] < threshold]

    def grouped(group_rows: list[dict], key: str, error_test) -> dict[str, dict]:
        names = sorted({str(row[key]) for row in group_rows})
        result = {}
        for name in names:
            selected = [row for row in group_rows if str(row[key]) == name]
            errors = sum(bool(error_test(row)) for row in selected)
            result[name] = {
                "count": len(selected),
                "errors": errors,
                "error_rate": errors / len(selected),
                "mean_ai_probability": float(
                    np.mean([row["score"] for row in selected])
                ),
            }
        return result

    return {
        "threshold": threshold,
        "real_count": len(reals),
        "fake_count": len(fakes),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "false_positive_rate": len(false_positives) / len(reals),
        "false_negative_rate": len(false_negatives) / len(fakes),
        "balanced_accuracy": 1.0
        - 0.5
        * (
            len(false_positives) / len(reals)
            + len(false_negatives) / len(fakes)
        ),
        "by_real_source": grouped(
            reals,
            "real_source",
            lambda row: row["score"] >= threshold,
        ),
        "by_fake_generator": grouped(
            fakes,
            "generator",
            lambda row: row["score"] < threshold,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Record source-aware FP/FN examples")
    parser.add_argument("--manifest", type=Path, required=True)
    prediction_source = parser.add_mutually_exclusive_group(required=True)
    prediction_source.add_argument("--checkpoint", type=Path)
    prediction_source.add_argument(
        "--evaluation-progress",
        type=Path,
        help="reuse the signed clean predictions written by aigc_detector.evaluate",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--preprocess-mode", choices=PREPROCESS_MODES)
    parser.add_argument("--codec-normalization", choices=CODEC_NORMALIZATION_MODES)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    metadata = {
        str(record["path"]): record for record in discover_manifest_records(args.manifest)
    }
    started = time.perf_counter()
    if args.evaluation_progress:
        labels, scores, paths, signature = load_clean_progress_predictions(
            args.evaluation_progress, args.manifest
        )
        payload = {
            "evaluation_progress": str(args.evaluation_progress),
            "manifest": str(args.manifest),
            "checkpoint": signature.get("checkpoint"),
            "checkpoint_sha256": signature.get("checkpoint_sha256"),
            "model": signature.get("model"),
            "preprocess_mode": signature.get("preprocess_mode"),
            "codec_normalization": signature.get("codec_normalization"),
            "reused_clean_predictions": True,
            "elapsed_seconds": time.perf_counter() - started,
            "summary_at_0.5": summarize_errors(
                labels, scores, paths, metadata, threshold=0.5
            ),
            "examples": rank_errors(labels, scores, paths, metadata, args.limit),
        }
    else:
        device = select_device(args.device)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        image_size = int(checkpoint["image_size"])
        model = create_binary_model(
            checkpoint["model_name"],
            pretrained=False,
            image_size=image_size,
            head_mode=checkpoint.get("head_mode", "linear"),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        preprocess_mode = args.preprocess_mode or checkpoint.get(
            "preprocess_mode", "stretch"
        )
        codec_normalization = args.codec_normalization or checkpoint.get(
            "codec_normalization", "none"
        )
        transform = evaluation_transform(
            image_size,
            mean=tuple(
                checkpoint.get("normalization_mean", (0.485, 0.456, 0.406))
            ),
            std=tuple(
                checkpoint.get("normalization_std", (0.229, 0.224, 0.225))
            ),
            preprocess_mode=preprocess_mode,
            codec_normalization=codec_normalization,
        )
        dataset = binary_dataset(args.manifest, transform=transform, seed=args.seed)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        labels, scores, paths = predict_loader(model, loader, device)
        payload = {
            "checkpoint": str(args.checkpoint),
            "manifest": str(args.manifest),
            "model": checkpoint["model_name"],
            "parameters": parameter_summary(model),
            "device": str(device),
            "preprocess_mode": preprocess_mode,
            "codec_normalization": codec_normalization,
            "head_mode": checkpoint.get("head_mode", "linear"),
            "reused_clean_predictions": False,
            "elapsed_seconds": time.perf_counter() - started,
            "summary_at_0.5": summarize_errors(
                labels, scores, paths, metadata, threshold=0.5
            ),
            "examples": rank_errors(labels, scores, paths, metadata, args.limit),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
