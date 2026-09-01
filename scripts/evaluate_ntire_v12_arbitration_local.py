#!/usr/bin/env python3
"""Resume the frozen NTIRE v12 arbitration on Apple MPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import timm
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import kaggle_evaluate_v12_robustness as workshop


SEED = 20260831
GATE_MANIFEST_SHA256 = "dfd3f196106544d586a3eb32c22f94d213f0ddd0f642f07d5cfc9e1fb08e2bb6"
EXPECTED_ROWS = 1024
EXPECTED_PER_LABEL = 512
EXPECTED_SOURCE_INVENTORY_SHA256 = (
    "a71cab43542cfbe5f95a9d14dd840bb46c94770c8b57d31e09db47493a95340d"
)
V12_TRAIN_MANIFEST_SHA256 = "8eaecdeb6b27220e4a1bff519a1a898321fb4f2577947b36af5440399e677611"
V12_EVAL_MANIFEST_SHA256 = "d61a8575f5330bd01a7351e52c0db2d8886731dee703d49943d381843ee50bd1"
BATCH_SIZE = 1
WORKERS = 0
CANDIDATES = {
    "pe_core": {
        "checkpoint": Path("models/v12_pe_core.pt"),
        "model_name": "vit_pe_core_large_patch14_336",
        "sha256": "f37bd6b445b12257ff29a9e54946c5bf9a9184e86a45dc5490537b9ea325ddd2",
    },
    "dinov2_control": {
        "checkpoint": Path("models/v12_dinov2.pt"),
        "model_name": "vit_large_patch14_dinov2.lvd142m",
        "sha256": "db07f30cbc94e4972f4a8c72c95bbe5df0dcd40b2dbd494ad82dc324cc1e2b5b",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(path)


def validate_gate(manifest: Path, train_manifest: Path, eval_manifest: Path) -> tuple[list[dict], dict]:
    if sha256_file(manifest) != GATE_MANIFEST_SHA256:
        raise RuntimeError("local NTIRE manifest checksum mismatch")
    rows = read_rows(manifest)
    labels = Counter(int(row["label"]) for row in rows)
    if len(rows) != EXPECTED_ROWS or labels != Counter({0: 512, 1: 512}):
        raise RuntimeError(f"unexpected NTIRE gate rows: {len(rows)}, {labels}")
    hashes = [str(row["image_sha256"]) for row in rows]
    if len(set(hashes)) != EXPECTED_ROWS:
        raise RuntimeError("NTIRE gate identities are not unique")
    source_inventory = "\n".join(sorted(str(row["source_filename"]) for row in rows)) + "\n"
    if hashlib.sha256(source_inventory.encode()).hexdigest() != EXPECTED_SOURCE_INVENTORY_SHA256:
        raise RuntimeError("NTIRE source-filename inventory mismatch")
    for index, row in enumerate(rows):
        image = (manifest.parent / row["path"]).resolve()
        if not image.is_file() or sha256_file(image) != row["image_sha256"]:
            raise RuntimeError(f"row {index}: NTIRE image is absent or changed")
        if row.get("training_allowed") is True:
            raise RuntimeError(f"row {index}: audit row explicitly allows training")
        lowered = str(row["path"]).casefold()
        if "demo_only" in lowered or "val2017" in lowered or "dall-e" in lowered:
            raise RuntimeError(f"row {index}: organizer-demo source term")

    compared = {}
    for role, current, expected_sha in (
        ("train", train_manifest, V12_TRAIN_MANIFEST_SHA256),
        ("eval", eval_manifest, V12_EVAL_MANIFEST_SHA256),
    ):
        if sha256_file(current) != expected_sha:
            raise RuntimeError(f"v12 {role} manifest checksum mismatch")
        current_rows = read_rows(current)
        identities = {
            str(value)
            for row in current_rows
            for value in (
                row.get("sha256"),
                row.get("image_sha256"),
                row.get("source_image_sha256"),
            )
            if value
        }
        overlap = len(set(hashes) & identities)
        if overlap:
            raise RuntimeError(f"NTIRE overlaps v12 {role} identities: {overlap}")
        compared[role] = len(current_rows)
    return rows, {
        "rows": len(rows),
        "labels": dict(labels),
        "unique_images": len(set(hashes)),
        "v12_train_rows_compared": compared["train"],
        "v12_eval_rows_compared": compared["eval"],
        "v12_train_identity_overlap": 0,
        "v12_eval_identity_overlap": 0,
        "training_allowed_rows": 0,
        "organizer_demo_rows": 0,
    }


class GateDataset(Dataset):
    def __init__(self, manifest: Path, rows: list[dict], transform) -> None:
        self.manifest = manifest
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = (self.manifest.parent / row["path"]).resolve()
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label"]), index


@torch.inference_mode()
def score(model: torch.nn.Module, dataset: GateDataset, device: torch.device, seed: int) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        generator=generator,
        persistent_workers=WORKERS > 0,
    )
    predictions = []
    for batch_index, (images, labels, indices) in enumerate(loader, 1):
        probabilities = torch.sigmoid(model(images.to(device)).flatten()).cpu()
        predictions.extend(
            {
                "index": int(index),
                "label": int(label),
                "score": float(probability),
                "image_sha256": dataset.rows[int(index)]["image_sha256"],
            }
            for index, label, probability in zip(indices, labels, probabilities)
        )
        if batch_index % 16 == 0 or batch_index == len(loader):
            print(
                json.dumps(
                    {"phase": "evaluate", "batch": batch_index, "batches": len(loader)}
                ),
                flush=True,
            )
    predictions.sort(key=lambda row: row["index"])
    return predictions


def evaluate(candidate: str, manifest: Path, train_manifest: Path, eval_manifest: Path, output_root: Path) -> dict:
    if not torch.backends.mps.is_available():
        raise RuntimeError("frozen local arbitration requires Apple MPS")
    rows, compliance = validate_gate(manifest, train_manifest, eval_manifest)
    specification = CANDIDATES[candidate]
    checkpoint_path = specification["checkpoint"]
    observed_checkpoint_sha = sha256_file(checkpoint_path)
    if observed_checkpoint_sha != specification["sha256"]:
        raise RuntimeError(f"{candidate} checkpoint checksum mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key, expected in {
        "model_name": specification["model_name"],
        "image_size": 224,
        "preprocess_mode": "short_side_crop",
        "codec_normalization": "jpeg_q96",
    }.items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(f"{candidate} checkpoint contract mismatch for {key}")

    device = torch.device("mps")
    model = timm.create_model(
        specification["model_name"], pretrained=False, num_classes=1, img_size=224
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    mean = tuple(checkpoint["normalization_mean"])
    std = tuple(checkpoint["normalization_std"])
    del checkpoint

    output = output_root / candidate
    progress_path = output / "progress.json"
    signature = {
        "candidate": candidate,
        "checkpoint_sha256": observed_checkpoint_sha,
        "manifest_sha256": GATE_MANIFEST_SHA256,
        "rows": len(rows),
        "conditions": [name for name, _ in workshop.conditions()],
        "codec_normalization": "jpeg_q96",
        "device": "mps",
        "arithmetic": "float32",
        "batch_size": BATCH_SIZE,
        "workers": WORKERS,
        "seed": SEED,
    }
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if progress.get("signature") != signature:
            raise RuntimeError(f"incompatible local resume state for {candidate}")
    else:
        progress = {"completed": False, "signature": signature, "conditions": {}}

    started = time.perf_counter()
    clean_predictions = None
    robust_predictions = []
    for index, (condition_name, image_transform) in enumerate(workshop.conditions()):
        prediction_path = output / f"{condition_name}_predictions.jsonl"
        if condition_name in progress["conditions"] and prediction_path.is_file():
            predictions = read_rows(prediction_path)
            if len(predictions) != len(rows):
                raise RuntimeError(f"invalid resume count for {candidate}/{condition_name}")
            print(json.dumps({"resumed": candidate, "condition": condition_name}), flush=True)
        else:
            torch.manual_seed(SEED + index)
            transform = workshop.condition_transform(224, mean, std, image_transform)
            predictions = score(
                model,
                GateDataset(manifest, rows, transform),
                device,
                SEED + index,
            )
            labels = [int(row["label"]) for row in predictions]
            scores = [float(row["score"]) for row in predictions]
            metrics = {"count": len(rows), "clean_auc": float(roc_auc_score(labels, scores))}
            atomic_jsonl(prediction_path, predictions)
            progress["conditions"][condition_name] = metrics
            atomic_json(progress_path, progress)
            print(
                json.dumps(
                    {"saved_condition": condition_name, "candidate": candidate, **metrics}
                ),
                flush=True,
            )
        if condition_name == "clean":
            clean_predictions = predictions
        else:
            robust_predictions.extend(predictions)

    if clean_predictions is None:
        raise RuntimeError("clean predictions missing")
    clean_auc = float(
        roc_auc_score(
            [int(row["label"]) for row in clean_predictions],
            [float(row["score"]) for row in clean_predictions],
        )
    )
    pooled_auc = float(
        roc_auc_score(
            [int(row["label"]) for row in robust_predictions],
            [float(row["score"]) for row in robust_predictions],
        )
    )
    progress.update(
        {
            "completed": True,
            "gate_compliance": compliance,
            "official_style": {
                "clean_auc": clean_auc,
                "pooled_robust_auc": pooled_auc,
                "score": 0.5 * clean_auc + 0.5 * pooled_auc,
            },
            "worst_individual_condition_auc": min(
                value["clean_auc"] for value in progress["conditions"].values()
            ),
            "elapsed_seconds_this_process": time.perf_counter() - started,
            "torch": torch.__version__,
            "device": "mps",
            "boundary": "Frozen one-shot local recovery; no tuning, calibration or weight search.",
        }
    )
    atomic_json(progress_path, progress)
    print("NTIRE_V12_LOCAL_SUMMARY " + json.dumps(progress, sort_keys=True), flush=True)
    return progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/ntire2026/shard_5_quality_route_gate/manifest.jsonl"),
    )
    parser.add_argument(
        "--v12-train-manifest",
        type=Path,
        default=Path("datasets/permissive_mixture_v12_canonical/train.jsonl"),
    )
    parser.add_argument(
        "--v12-eval-manifest",
        type=Path,
        default=Path("datasets/permissive_mixture_v12_canonical/eval_frozen.jsonl"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/ntire-v12-final-arbitration-local"),
    )
    args = parser.parse_args()
    evaluate(
        args.candidate,
        args.manifest,
        args.v12_train_manifest,
        args.v12_eval_manifest,
        args.output_root,
    )


if __name__ == "__main__":
    main()
