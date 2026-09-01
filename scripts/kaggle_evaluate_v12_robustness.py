#!/usr/bin/env python3
"""Run the frozen workshop 20-condition matrix on a clean-screen v12 model.

The 19 transformed conditions are exactly the individually applied workshop
conditions.  Every condition is followed by the same label-independent JPEG
q96 normalization used during v12 training and clean inference.  Results and
predictions are written atomically after every condition so an interrupted
Kaggle session can resume without silently dropping work.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import timm
import torch
from sklearn.metrics import roc_auc_score
from torchvision.transforms import v2

import kaggle_train_v3 as runner


SEED = 20260831
CANDIDATES = {
    "pe_core": {
        "model": "vit_pe_core_large_patch14_336",
        "root": Path("/kaggle/working/track5-v12-permissive"),
    },
    "dinov2_control": {
        "model": "vit_large_patch14_dinov2.lvd142m",
        "root": Path("/kaggle/working/track5-v12-dino-control"),
    },
}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def conditions() -> list[tuple[str, object | None]]:
    values: list[tuple[str, object | None]] = [("clean", None)]
    values += [
        (f"jpeg_q{quality}", runner.JpegCompression(quality))
        for quality in (90, 70, 50, 30)
    ]
    values += [
        (f"blur_sigma_{sigma:g}", runner.GaussianBlurPIL(sigma))
        for sigma in (0.5, 1.0, 2.0)
    ]
    values += [
        (f"resize_{scale:g}", runner.DownUpResize(scale))
        for scale in (0.5, 0.25)
    ]
    values += [
        (f"noise_sigma_{sigma:.2f}", runner.GaussianNoisePIL(sigma))
        for sigma in (0.02, 0.05, 0.10)
    ]
    values += [
        (f"{kind}_{factor:g}", runner.FixedEnhancement(kind, factor))
        for kind in ("brightness", "contrast", "saturation")
        for factor in (0.8, 1.2)
    ]
    values.append(("center_crop_80", runner.CenterCropFraction(0.8)))
    return values


def condition_transform(
    image_size: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    image_transform=None,
):
    operations = []
    if image_transform is not None:
        operations.append(image_transform)
    operations.extend(
        [
            runner.JpegCompression(96),
            v2.Resize(image_size, antialias=True),
            v2.CenterCrop((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std),
        ]
    )
    return v2.Compose(operations)


def pooled_metrics(rows: list[dict], predictions: list[dict]) -> dict:
    labels = [int(row["label"]) for row in predictions]
    scores = [float(row["score"]) for row in predictions]
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "groups": runner.grouped_metrics(
            rows * (len(conditions()) - 1), labels, scores
        ),
    }


def evaluate_candidate(name: str) -> dict:
    # Importing this runtime configuration here prevents test/import order from
    # mutating the shared runner globals before an actual evaluator process.
    import kaggle_train_v12_permissive as v12_runtime

    if name not in CANDIDATES:
        raise ValueError(f"unknown candidate: {name}")
    specification = CANDIDATES[name]
    model_name = specification["model"]
    candidate_root = specification["root"]

    _, package = runner.validate_package()
    eval_manifest = runner.WORK_ROOT / "manifests/eval_frozen.jsonl"
    train_manifest = runner.WORK_ROOT / "manifests/train.jsonl"
    rows = read_rows(eval_manifest)
    compliance = v12_runtime.validate_rows(read_rows(train_manifest), rows)

    model_root = candidate_root / model_name.replace(".", "_")
    clean_report_path = model_root / "report.json"
    checkpoint_path = model_root / "model.pt"
    if not clean_report_path.is_file() or not checkpoint_path.is_file():
        raise RuntimeError(f"missing trained v12 artifact for {name}")
    clean_report = json.loads(clean_report_path.read_text())
    if not clean_report.get("promotion", {}).get("passes_clean_screen"):
        raise RuntimeError(f"{name} did not pass the frozen clean screen")

    checkpoint_sha256 = runner.file_sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != model_name:
        raise RuntimeError(f"checkpoint model mismatch for {name}")
    if checkpoint.get("codec_normalization") != "jpeg_q96":
        raise RuntimeError(f"checkpoint codec contract mismatch for {name}")
    image_size = int(checkpoint["image_size"])
    mean = tuple(checkpoint["normalization_mean"])
    std = tuple(checkpoint["normalization_std"])
    model = timm.create_model(
        model_name, pretrained=False, num_classes=1, img_size=image_size
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda().eval()

    output = model_root / "workshop-20-condition-matrix"
    progress_path = output / "progress.json"
    signature = {
        "candidate": name,
        "checkpoint_sha256": checkpoint_sha256,
        "package_inventory_sha256": package["inventory_sha256"],
        "eval_manifest_sha256": runner.file_sha256(eval_manifest),
        "rows": len(rows),
        "conditions": [condition[0] for condition in conditions()],
        "codec_normalization": "jpeg_q96",
        "seed": SEED,
    }
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if progress.get("signature") != signature:
            raise RuntimeError(f"incompatible matrix resume state for {name}")
    else:
        progress = {"completed": False, "signature": signature, "conditions": {}}

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    clean_predictions = None
    robust_predictions: list[dict] = []
    for index, (condition_name, image_transform) in enumerate(conditions()):
        prediction_path = output / f"{condition_name}_predictions.jsonl"
        if condition_name in progress["conditions"] and prediction_path.is_file():
            predictions = read_rows(prediction_path)
            if len(predictions) != len(rows):
                raise RuntimeError(f"invalid resume count for {name}/{condition_name}")
            print(json.dumps({"resumed": name, "condition": condition_name}), flush=True)
        else:
            torch.manual_seed(SEED + index)
            dataset = runner.ManifestDataset(
                eval_manifest,
                condition_transform(image_size, mean, std, image_transform),
                rows=rows,
            )
            metrics, predictions = runner.evaluate(model, dataset)
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            prediction_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions)
            )
            progress["conditions"][condition_name] = metrics
            atomic_json(progress_path, progress)
            print(
                json.dumps(
                    {
                        "saved_condition": condition_name,
                        "candidate": name,
                        "auc": metrics["clean_auc"],
                    }
                ),
                flush=True,
            )
        if condition_name == "clean":
            clean_predictions = predictions
        else:
            robust_predictions.extend(predictions)

    if clean_predictions is None:
        raise RuntimeError(f"clean predictions missing for {name}")
    clean_labels = [int(row["label"]) for row in clean_predictions]
    clean_scores = [float(row["score"]) for row in clean_predictions]
    clean_auc = float(roc_auc_score(clean_labels, clean_scores))
    robust = pooled_metrics(rows, robust_predictions)
    progress.update(
        {
            "completed": True,
            "workshop_compliance": compliance,
            "official_style": {
                "clean_auc": clean_auc,
                "pooled_robust_auc": robust["auc"],
                "score": 0.5 * clean_auc + 0.5 * robust["auc"],
            },
            "pooled_robust_groups": robust["groups"],
            "worst_individual_condition_auc": min(
                value["clean_auc"] for value in progress["conditions"].values()
            ),
            "elapsed_seconds_this_process": time.time() - started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        }
    )
    atomic_json(progress_path, progress)
    print(json.dumps(progress, indent=2), flush=True)
    return progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    arguments = parser.parse_args()
    evaluate_candidate(arguments.candidate)


if __name__ == "__main__":
    main()
