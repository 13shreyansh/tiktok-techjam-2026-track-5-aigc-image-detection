#!/usr/bin/env python3
"""Screen the fixed v6/v10 blend on the already-consumed v6 clean gates.

Run this before exposing the untouched low-resolution promotion gate.  A
failure preserves that fresh gate for a later, genuinely new candidate.  This
script performs inference only and cannot change a model, blend or threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import timm
import torch
from torch.utils.data import DataLoader

import kaggle_train_v3 as runner
import kaggle_train_v8_frontier as v8


MODEL_NAME = "vit_pe_core_large_patch14_336"
IMAGE_SIZE = 224
V6_BYTES = 631_645_967
V6_SHA256 = "48ea50773fbd1b7247fff25fde6f985183e29f2eb517b5ac0f6319c1fe38b644"
V6_WEIGHT = 0.75
V10_WEIGHT = 0.25
PHYSICAL_BATCH_SIZE = 64
OUTPUT_ROOT = Path("/kaggle/working/track5-v10-internal-screen")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def locate_v6() -> Path:
    candidates = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if root.exists():
            candidates.extend(
                path
                for path in root.rglob("model.pt")
                if path.is_file() and path.stat().st_size == V6_BYTES
            )
    exact = [path for path in candidates if sha256_file(path) == V6_SHA256]
    if not exact:
        raise RuntimeError("exact v6 checkpoint is absent")
    return sorted(exact, key=str)[0]


def load_model(path: Path, expected_sha256: str) -> tuple[torch.nn.Module, dict]:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"checkpoint changed before load: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["model_name"] != MODEL_NAME or int(checkpoint["image_size"]) != IMAGE_SIZE:
        raise RuntimeError("checkpoint architecture contract mismatch")
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1, img_size=IMAGE_SIZE)
    model.load_state_dict(checkpoint["state_dict"])
    return model.cuda().eval(), checkpoint


@torch.inference_mode()
def score(
    v6_model: torch.nn.Module,
    v10_model: torch.nn.Module,
    dataset: runner.ManifestDataset,
) -> list[dict]:
    loader = DataLoader(
        dataset,
        batch_size=PHYSICAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    predictions = []
    for images, labels, indices in loader:
        original_count = int(images.shape[0])
        if original_count < PHYSICAL_BATCH_SIZE:
            images = torch.cat(
                [images, images[-1:].repeat(PHYSICAL_BATCH_SIZE - original_count, 1, 1, 1)]
            )
        images = images.cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            v6_scores = torch.sigmoid(v6_model(images).flatten())
            v10_scores = torch.sigmoid(v10_model(images).flatten())
            blend_scores = V6_WEIGHT * v6_scores + V10_WEIGHT * v10_scores
        for label, index, v6_score, v10_score, blend_score in zip(
            labels.tolist(),
            indices.tolist(),
            v6_scores[:original_count].float().cpu().tolist(),
            v10_scores[:original_count].float().cpu().tolist(),
            blend_scores[:original_count].float().cpu().tolist(),
        ):
            source = dataset.rows[int(index)]
            predictions.append(
                {
                    "index": int(index),
                    "label": int(label),
                    "image_sha256": source["image_sha256"],
                    "generator": source.get("generator"),
                    "real_source": source.get("real_source"),
                    "family": source.get("family"),
                    "v6_score": float(v6_score),
                    "v10_score": float(v10_score),
                    "score": float(blend_score),
                }
            )
    if [row["index"] for row in predictions] != list(range(len(dataset.rows))):
        raise RuntimeError("internal-screen prediction order mismatch")
    return predictions


def metrics(rows: list[dict], score_key: str) -> dict:
    labels = [int(row["label"]) for row in rows]
    scores = [float(row[score_key]) for row in rows]
    metadata = [
        {
            "label": row["label"],
            "generator": row.get("generator"),
            "real_source": row.get("real_source"),
        }
        for row in rows
    ]
    return {
        "auc": runner.auc(list(zip(labels, scores))),
        "worst_pair_auc": runner.grouped_metrics(metadata, labels, scores)[
            "worst_generator_real_source_pair_auc"
        ],
    }


def subset_by_manifest(predictions: list[dict], rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in predictions:
        grouped.setdefault(row["image_sha256"], []).append(row)
    by_hash = {}
    for image_sha256, duplicates in grouped.items():
        labels = {int(row["label"]) for row in duplicates}
        for score_key in ("v6_score", "v10_score", "score"):
            values = [float(row[score_key]) for row in duplicates]
            if max(values) - min(values) > 1e-6:
                raise RuntimeError(
                    f"duplicate selection score mismatch for {image_sha256}: {score_key}"
                )
        if len(labels) != 1:
            raise RuntimeError(f"duplicate selection label mismatch: {image_sha256}")
        by_hash[image_sha256] = duplicates[0]
    subset = []
    for index, row in enumerate(rows):
        if row["image_sha256"] not in by_hash:
            raise RuntimeError(f"content row absent from selection gate: {row['image_sha256']}")
        selected = {**by_hash[row["image_sha256"]], "index": index}
        if int(selected["label"]) != int(row["label"]):
            raise RuntimeError("content label mismatch")
        subset.append(selected)
    return subset


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v10-checkpoint", type=Path, required=True)
    parser.add_argument("--v10-bytes", type=int, required=True)
    parser.add_argument("--v10-sha256", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("internal screen requires the verified CUDA-FP16 path")
    if args.v10_checkpoint.stat().st_size != args.v10_bytes:
        raise RuntimeError("v10 checkpoint byte count mismatch")
    if sha256_file(args.v10_checkpoint) != args.v10_sha256:
        raise RuntimeError("v10 checkpoint SHA-256 mismatch")

    runner.EXCLUDED_EVAL_SHA256 = v8.EXCLUDED_EVAL_SHA256
    package_sha256, package = runner.validate_package()
    selection_manifest = runner.WORK_ROOT / "manifests/eval_selection.jsonl"
    content_manifest = runner.WORK_ROOT / "manifests/eval_content_holdout.jsonl"
    selection_rows = runner.filter_evaluation_rows(
        read_jsonl(selection_manifest), runner.EXCLUDED_EVAL_SHA256
    )
    content_rows = runner.filter_evaluation_rows(
        read_jsonl(content_manifest), runner.EXCLUDED_EVAL_SHA256
    )
    v6_path = locate_v6()
    v6_model, v6_checkpoint = load_model(v6_path, V6_SHA256)
    v10_model, v10_checkpoint = load_model(args.v10_checkpoint, args.v10_sha256)
    mean = tuple(v6_checkpoint["normalization_mean"])
    std = tuple(v6_checkpoint["normalization_std"])
    if mean != tuple(v10_checkpoint["normalization_mean"]) or std != tuple(
        v10_checkpoint["normalization_std"]
    ):
        raise RuntimeError("v6/v10 normalization mismatch")
    dataset = runner.ManifestDataset(
        selection_manifest,
        runner.eval_transform(mean, std),
        rows=selection_rows,
    )
    selection_predictions = score(v6_model, v10_model, dataset)
    content_predictions = subset_by_manifest(selection_predictions, content_rows)
    summary = {"v6": {}, "blend": {}}
    for candidate, score_key in (("v6", "v6_score"), ("blend", "score")):
        selection = metrics(selection_predictions, score_key)
        content = metrics(content_predictions, score_key)
        summary[candidate] = {
            "selection_clean_auc": selection["auc"],
            "selection_worst_pair_auc": selection["worst_pair_auc"],
            "content_clean_auc": content["auc"],
            "content_worst_pair_auc": content["worst_pair_auc"],
        }
    checks = {
        "selection_clean_auc": summary["blend"]["selection_clean_auc"]
        >= summary["v6"]["selection_clean_auc"] - 0.002,
        "selection_worst_pair_auc": summary["blend"]["selection_worst_pair_auc"]
        >= summary["v6"]["selection_worst_pair_auc"] - 0.01,
        "content_clean_auc": summary["blend"]["content_clean_auc"]
        >= summary["v6"]["content_clean_auc"] - 0.002,
        "content_worst_pair_auc": summary["blend"]["content_worst_pair_auc"]
        >= summary["v6"]["content_worst_pair_auc"] - 0.01,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT_ROOT / "selection_predictions.jsonl", selection_predictions)
    write_jsonl(OUTPUT_ROOT / "content_predictions.jsonl", content_predictions)
    report = {
        **summary,
        "completed": True,
        "passes_internal_screen": all(checks.values()),
        "checks": checks,
        "v6_package_sha256": package_sha256,
        "v6_inventory_sha256": package["inventory_sha256"],
        "v6_checkpoint_sha256": V6_SHA256,
        "v10_checkpoint_sha256": args.v10_sha256,
        "weights": {"v6": V6_WEIGHT, "v10": V10_WEIGHT},
        "arithmetic": "one GPU; physical batch 64; sigmoid and 75/25 blend in FP16; FP32 after blend",
        "fresh_promotion_gate_opened": False,
        "forbidden_demo_resources_used": False,
        "boundary": "Fail any check: reject v10 without opening the fresh promotion gate.",
    }
    (OUTPUT_ROOT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
