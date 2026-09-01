#!/usr/bin/env python3
"""Evaluate the frozen v8 candidate on two independent promotion gates.

The input package contains the prompt-held-out Qwen gate and the full balanced
NTIRE shard-5 audit sample.  It contains no organizer demo-only images.  Every
expanded image, manifest and package inventory is verified before inference.
Results are persisted after each workshop-listed individual transformation so
an interrupted Kaggle session can resume without silently dropping conditions.
"""

from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path

import timm
import torch
from sklearn.metrics import roc_auc_score

import kaggle_train_v3 as runner


PACKAGE_NAME = "track5-v8-promotion-gates.zip"
PACKAGE_BYTES = 573_433_989
PACKAGE_SHA256 = "66e9668fd6be0ca86108067bb7782aeaf34a87f8d15b35ab21179b26e14acbec"
INVENTORY_SHA256 = "b1b8e5b4c9882b3e807c368a60633f1d5f971e7b0d81b2390fe3d08227dca02f"
UNIQUE_IMAGES = 1_088
SOURCE_BYTES = 572_706_460
MANIFESTS = {
    "qwen_prompt_holdout": {
        "path": "manifests/combined_gate.jsonl",
        "rows": 576,
        "sha256": "d584b32c5b023a7a88bd1e6455a9dbc9bef852eb4b1fbd1fb5e35039120d98d1",
    },
    "ntire_shard5_full_audit": {
        "path": "manifests/manifest.jsonl",
        "rows": 512,
        "sha256": "8327cb6ed314e9693c107719a875e4c6caf9825734206e3419d6c8c1e573a444",
    },
}
WORK_ROOT = Path("/kaggle/working/track5-v8-promotion-gates")
CANDIDATE_ROOT = Path("/kaggle/working/track5-v8-frontier-candidate")
MODEL_NAME = "vit_pe_core_large_patch14_336"
SEED = 20260830


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"unsafe archive member: {member.filename}")
    archive.extractall(destination)


def validate_package() -> tuple[Path, dict]:
    archives = list(Path("/kaggle/input").rglob(PACKAGE_NAME))
    if len(archives) == 1:
        archive_path = archives[0]
        if archive_path.stat().st_size != PACKAGE_BYTES:
            raise RuntimeError("promotion package byte count mismatch")
        if runner.file_sha256(archive_path) != PACKAGE_SHA256:
            raise RuntimeError("promotion package SHA-256 mismatch")
        if WORK_ROOT.exists():
            shutil.rmtree(WORK_ROOT)
        WORK_ROOT.mkdir(parents=True)
        with zipfile.ZipFile(archive_path) as archive:
            safe_extract(archive, WORK_ROOT)
        root = WORK_ROOT
        transport = "uploaded ZIP byte checksum"
    elif not archives:
        candidates = []
        for report_path in Path("/kaggle/input").rglob("package.json"):
            try:
                report = json.loads(report_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if report.get("inventory_sha256") == INVENTORY_SHA256:
                candidates.append(report_path.parent)
        if len(candidates) != 1:
            raise RuntimeError(f"expected one expanded promotion package: {candidates}")
        root = candidates[0]
        transport = "Kaggle-expanded content-addressed files"
    else:
        raise RuntimeError(f"multiple promotion archives: {archives}")

    report = json.loads((root / "package.json").read_text())
    expected = {
        "unique_images": UNIQUE_IMAGES,
        "source_bytes": SOURCE_BYTES,
        "inventory_sha256": INVENTORY_SHA256,
    }
    observed = {key: report.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(f"promotion package inventory mismatch: {observed}")
    images = sorted(path for path in (root / "images").rglob("*") if path.is_file())
    if len(images) != UNIQUE_IMAGES:
        raise RuntimeError(f"promotion image count mismatch: {len(images)}")
    observed_bytes = 0
    for index, image in enumerate(images, 1):
        if runner.file_sha256(image) != image.stem:
            raise RuntimeError(f"promotion image content mismatch: {image}")
        observed_bytes += image.stat().st_size
        if index % 500 == 0 or index == len(images):
            print(json.dumps({"verified_promotion_images": index}), flush=True)
    if observed_bytes != SOURCE_BYTES:
        raise RuntimeError(f"promotion source byte mismatch: {observed_bytes}")
    for specification in MANIFESTS.values():
        manifest = root / specification["path"]
        if runner.file_sha256(manifest) != specification["sha256"]:
            raise RuntimeError(f"promotion manifest mismatch: {manifest}")
        rows = [line for line in manifest.read_text().splitlines() if line]
        if len(rows) != specification["rows"]:
            raise RuntimeError(f"promotion manifest row mismatch: {manifest}")
    return root, {**report, "runtime_verification": transport}


def conditions() -> list[tuple[str, object | None, float | None]]:
    values: list[tuple[str, object | None, float | None]] = [("clean", None, None)]
    values += [
        (f"jpeg_q{quality}", runner.JpegCompression(quality), None)
        for quality in (90, 70, 50, 30)
    ]
    values += [
        (f"blur_sigma_{sigma:g}", runner.GaussianBlurPIL(sigma), None)
        for sigma in (0.5, 1.0, 2.0)
    ]
    values += [
        (f"resize_{scale:g}", runner.DownUpResize(scale), None)
        for scale in (0.5, 0.25)
    ]
    values += [
        (f"noise_sigma_{sigma:.2f}", None, sigma)
        for sigma in (0.02, 0.05, 0.10)
    ]
    values += [
        (f"{kind}_{factor:g}", runner.FixedEnhancement(kind, factor), None)
        for kind in ("brightness", "contrast", "saturation")
        for factor in (0.8, 1.2)
    ]
    values.append(("center_crop_80", runner.CenterCropFraction(0.8), None))
    return values


def condition_transform(mean, std, image_transform=None, tensor_noise=None):
    from torchvision.transforms import v2

    operations = []
    if image_transform is not None:
        operations.append(image_transform)
    operations.extend(
        [
            v2.Resize(runner.IMAGE_SIZE, antialias=True),
            v2.CenterCrop((runner.IMAGE_SIZE, runner.IMAGE_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    if tensor_noise is not None:
        operations.append(
            v2.Lambda(
                lambda tensor: (
                    tensor + torch.randn_like(tensor) * tensor_noise
                ).clamp(0.0, 1.0)
            )
        )
    operations.append(v2.Normalize(mean, std))
    return v2.Compose(operations)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def evaluate_gate(
    model: torch.nn.Module,
    manifest: Path,
    rows: list[dict],
    mean: tuple[float, ...],
    std: tuple[float, ...],
    output: Path,
    signature: dict,
) -> dict:
    progress_path = output / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if progress.get("signature") != signature:
            raise RuntimeError(f"incompatible promotion resume state: {progress_path}")
    else:
        progress = {"completed": False, "signature": signature, "conditions": {}}
    clean_predictions = None
    pooled_predictions: list[dict] = []
    started = time.time()
    for index, (name, image_transform, tensor_noise) in enumerate(conditions()):
        prediction_path = output / f"{name}_predictions.jsonl"
        if name in progress["conditions"] and prediction_path.is_file():
            predictions = read_rows(prediction_path)
            if len(predictions) != len(rows):
                raise RuntimeError(f"invalid saved prediction count for {name}")
            print(json.dumps({"resumed_promotion_condition": name}), flush=True)
        else:
            torch.manual_seed(SEED + index)
            dataset = runner.ManifestDataset(
                manifest,
                condition_transform(mean, std, image_transform, tensor_noise),
                rows=rows,
            )
            metrics, predictions = runner.evaluate(model, dataset)
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            prediction_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions)
            )
            progress["conditions"][name] = metrics
            atomic_json(progress_path, progress)
            print(
                json.dumps({"saved_promotion_condition": name, "auc": metrics["clean_auc"]}),
                flush=True,
            )
        if name == "clean":
            clean_predictions = predictions
        else:
            pooled_predictions.extend(predictions)
    if clean_predictions is None:
        raise RuntimeError("promotion clean predictions missing")
    clean_labels = [int(row["label"]) for row in clean_predictions]
    clean_scores = [float(row["score"]) for row in clean_predictions]
    robust_labels = [int(row["label"]) for row in pooled_predictions]
    robust_scores = [float(row["score"]) for row in pooled_predictions]
    clean_auc = float(roc_auc_score(clean_labels, clean_scores))
    robust_auc = float(roc_auc_score(robust_labels, robust_scores))
    progress.update(
        {
            "completed": True,
            "official_style": {
                "clean_auc": clean_auc,
                "pooled_robust_auc": robust_auc,
                "score": 0.5 * clean_auc + 0.5 * robust_auc,
            },
            "pooled_robust_groups": runner.grouped_metrics(
                rows * (len(conditions()) - 1), robust_labels, robust_scores
            ),
            "elapsed_seconds_this_process": time.time() - started,
        }
    )
    atomic_json(progress_path, progress)
    return progress


def main() -> None:
    package_root, package = validate_package()
    checkpoint_path = CANDIDATE_ROOT / MODEL_NAME / "model.pt"
    if not checkpoint_path.is_file():
        raise RuntimeError(f"v8 checkpoint missing: {checkpoint_path}")
    checkpoint_sha256 = runner.file_sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["model_name"] != MODEL_NAME:
        raise RuntimeError("v8 checkpoint model mismatch")
    model = timm.create_model(
        MODEL_NAME, pretrained=False, num_classes=1, img_size=runner.IMAGE_SIZE
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda().eval()
    mean = tuple(checkpoint["normalization_mean"])
    std = tuple(checkpoint["normalization_std"])
    output = CANDIDATE_ROOT / MODEL_NAME / "promotion-gates"
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    gates = {}
    for name, specification in MANIFESTS.items():
        manifest = package_root / specification["path"]
        rows = read_rows(manifest)
        gates[name] = evaluate_gate(
            model,
            manifest,
            rows,
            mean,
            std,
            output / name,
            {
                "checkpoint_sha256": checkpoint_sha256,
                "package_inventory_sha256": INVENTORY_SHA256,
                "manifest_sha256": specification["sha256"],
                "rows": specification["rows"],
                "conditions": [condition[0] for condition in conditions()],
                "seed": SEED,
            },
        )
    summary = {
        "completed": all(gate["completed"] for gate in gates.values()),
        "checkpoint_sha256": checkpoint_sha256,
        "package_sha256": PACKAGE_SHA256,
        "package_inventory_sha256": INVENTORY_SHA256,
        "package_runtime_verification": package["runtime_verification"],
        "gates": {
            name: {
                "official_style": gate["official_style"],
                "clean_groups": gate["conditions"]["clean"]["groups"],
                "pooled_robust_groups": gate["pooled_robust_groups"],
                "noise_sigma_0.10": gate["conditions"]["noise_sigma_0.10"],
            }
            for name, gate in gates.items()
        },
        "elapsed_seconds": time.time() - started,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
