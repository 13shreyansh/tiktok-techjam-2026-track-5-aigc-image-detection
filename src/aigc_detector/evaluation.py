from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping

import torch
from torch.utils.data import DataLoader

from .data import binary_dataset, discover_manifest_records
from .metrics import auc, official_metrics
from .transforms import (
    Condition,
    evaluation_inference_transform,
    official_conditions,
)


@torch.inference_mode()
def predict_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    labels: list[float] = []
    scores: list[float] = []
    paths: list[str] = []
    model.eval()
    for images, batch_labels, batch_paths in loader:
        if images.ndim == 5:
            batch, views, channels, height, width = images.shape
            if device.type == "mps":
                # PE-Core-L is stable at batch one on the available Apple MPS
                # runtime but its batch-two path aborts in MPSNDArray. Keep the
                # view dimension sequential on MPS; CUDA can batch the views.
                logits = torch.stack(
                    [model(images[:, index].to(device)).flatten() for index in range(views)],
                    dim=1,
                )
            else:
                logits = model(
                    images.reshape(batch * views, channels, height, width).to(device)
                ).reshape(batch, views)
            probabilities = torch.sigmoid(logits).mean(dim=1).cpu().tolist()
        else:
            logits = model(images.to(device)).flatten()
            probabilities = torch.sigmoid(logits).cpu().tolist()
        labels.extend(batch_labels.tolist())
        scores.extend(probabilities)
        paths.extend(batch_paths)
    return labels, scores, paths


def grouped_metrics(
    labels: list[float],
    scores: list[float],
    paths: list[str],
    metadata: dict[str, dict],
) -> dict:
    """Report generator and real-source performance without pooling failures away."""
    rows = [
        (int(label), float(score), metadata[str(Path(path).resolve())])
        for label, score, path in zip(labels, scores, paths)
    ]
    payload: dict[str, dict] = {
        "fake_generators": {},
        "real_sources": {},
        "generator_real_source_pairs": {},
    }
    generators = sorted({str(row.get("generator", "unknown")) for label, _, row in rows if label == 1})
    real_sources = sorted({str(row.get("real_source", "unknown")) for label, _, row in rows if label == 0})
    for generator in generators:
        selected = [
            (label, score)
            for label, score, row in rows
            if label == 0 or str(row.get("generator", "unknown")) == generator
        ]
        fake_scores = [score for label, score in selected if label == 1]
        payload["fake_generators"][generator] = {
            "auc_against_all_reals": auc(
                [label for label, _ in selected], [score for _, score in selected]
            ),
            "fake_count": len(fake_scores),
            "true_positive_rate_at_0.5": sum(score >= 0.5 for score in fake_scores) / len(fake_scores),
        }
    for source in real_sources:
        selected = [
            (label, score)
            for label, score, row in rows
            if label == 1 or str(row.get("real_source", "unknown")) == source
        ]
        real_scores = [score for label, score in selected if label == 0]
        payload["real_sources"][source] = {
            "auc_against_all_fakes": auc(
                [label for label, _ in selected], [score for _, score in selected]
            ),
            "real_count": len(real_scores),
            "true_negative_rate_at_0.5": sum(score < 0.5 for score in real_scores) / len(real_scores),
        }
    pair_aucs: list[float] = []
    for generator in generators:
        payload["generator_real_source_pairs"][generator] = {}
        for source in real_sources:
            selected = [
                (label, score)
                for label, score, row in rows
                if (
                    label == 1
                    and str(row.get("generator", "unknown")) == generator
                )
                or (
                    label == 0
                    and str(row.get("real_source", "unknown")) == source
                )
            ]
            pair_auc = auc(
                [label for label, _ in selected], [score for _, score in selected]
            )
            pair_aucs.append(pair_auc)
            payload["generator_real_source_pairs"][generator][source] = {
                "auc": pair_auc,
                "fake_count": sum(label == 1 for label, _ in selected),
                "real_count": sum(label == 0 for label, _ in selected),
            }
    payload["worst_fake_generator_auc"] = min(
        group["auc_against_all_reals"] for group in payload["fake_generators"].values()
    )
    payload["worst_real_source_auc"] = min(
        group["auc_against_all_fakes"] for group in payload["real_sources"].values()
    )
    payload["worst_generator_real_source_pair_auc"] = min(pair_aucs)
    return payload


def evaluate_conditions(
    model: torch.nn.Module,
    dataset_root: str | Path,
    device: torch.device,
    image_size: int,
    batch_size: int,
    workers: int,
    max_per_class: int | None,
    seed: int,
    robust: bool,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    preprocess_mode: str = "stretch",
    codec_normalization: str = "none",
    completed_predictions: Mapping[str, Mapping[str, list]] | None = None,
    prediction_callback: Callable[[str, dict[str, list]], None] | None = None,
    inference_policy: str = "reference",
) -> dict:
    conditions = official_conditions() if robust else [Condition("clean")]
    results: dict[str, dict[str, float | int]] = {}
    pooled_labels: list[float] = []
    pooled_scores: list[float] = []
    pooled_paths: list[str] = []
    clean_labels: list[float] = []
    clean_scores: list[float] = []
    metadata = None
    source_path = Path(dataset_root)
    if source_path.suffix.lower() == ".jsonl":
        metadata = {
            str(record["path"]): record for record in discover_manifest_records(source_path)
        }

    completed_predictions = completed_predictions or {}
    known_conditions = {condition.name for condition in conditions}
    unknown_completed = set(completed_predictions) - known_conditions
    if unknown_completed:
        raise ValueError(
            f"resume data contains unknown conditions: {sorted(unknown_completed)}"
        )

    for condition_index, condition in enumerate(conditions):
        if condition.name in completed_predictions:
            saved = completed_predictions[condition.name]
            labels = [float(value) for value in saved.get("labels", [])]
            scores = [float(value) for value in saved.get("scores", [])]
            paths = [str(value) for value in saved.get("paths", [])]
            if not labels or not (len(labels) == len(scores) == len(paths)):
                raise ValueError(
                    f"invalid resume predictions for condition {condition.name}"
                )
        else:
            # Give every condition its own deterministic random stream. This
            # makes Gaussian-noise results identical whether the full matrix is
            # evaluated in one pass or resumed after an interruption.
            torch.manual_seed(seed + condition_index)
            dataset = binary_dataset(
                dataset_root,
                transform=evaluation_inference_transform(
                    image_size,
                    condition,
                    mean=mean,
                    std=std,
                    preprocess_mode=preprocess_mode,
                    codec_normalization=codec_normalization,
                    inference_policy=inference_policy,
                ),
                max_per_class=max_per_class,
                seed=seed,
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
            )
            labels, scores, paths = predict_loader(model, loader, device)
            if prediction_callback is not None:
                prediction_callback(
                    condition.name,
                    {"labels": labels, "scores": scores, "paths": paths},
                )
        results[condition.name] = {"auc": auc(labels, scores), "count": len(labels)}
        if metadata is not None:
            results[condition.name]["groups"] = grouped_metrics(
                labels, scores, paths, metadata
            )
        if condition.name == "clean":
            clean_labels, clean_scores = labels, scores
        else:
            pooled_labels.extend(labels)
            pooled_scores.extend(scores)
            pooled_paths.extend(paths)

    payload: dict = {"conditions": results}
    if robust:
        payload["official"] = official_metrics(
            clean_labels, clean_scores, pooled_labels, pooled_scores
        ).as_dict()
        if metadata is not None:
            payload["pooled_robust_groups"] = grouped_metrics(
                pooled_labels, pooled_scores, pooled_paths, metadata
            )
    else:
        payload["clean_auc"] = results["clean"]["auc"]
    return payload
