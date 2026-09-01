"""Evaluate the saved PE-Core checkpoint on the audit-only external gate.

The Community Forensics pixels are never used for training, model selection, or
threshold calibration.  This script verifies the immutable private package,
runs the workshop's clean plus 19 individual transformations, and checkpoints
every condition so a Kaggle interruption does not invalidate completed work.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import timm
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

import kaggle_stress_eval_v6 as stress
import kaggle_train_v3 as runner
import kaggle_train_v6  # noqa: F401  # apply checksum-pinned v6 constants


EXPECTED_ZIP_SHA256 = (
    "123a0e4bb8ae484a804a6a39a9f20063e62116e260fef813efe44824cc11a084"
)
EXPECTED_INVENTORY_SHA256 = (
    "5b345de1d57badd4a9bbc6b33876a29f2660c9cd173cff54aa9a379038578943"
)
EXPECTED_MANIFEST_SHA256 = (
    "2d770ff99f781320a10a9a15fa03de79d2cab40929b09fb1b4db7e759848398c"
)
EXPECTED_ROWS = 624
EXPECTED_REAL = 312
EXPECTED_FAKE = 312
EXPECTED_GENERATOR_MODELS = 78
WORK_ROOT = Path("/kaggle/working/community-forensics-external-gate")


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination_resolved = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if destination_resolved not in target.parents and target != destination_resolved:
            raise RuntimeError(f"unsafe archive member: {member.filename}")
    archive.extractall(destination)


def find_verified_extracted_root() -> Path:
    candidates = []
    for package_path in Path("/kaggle/input").rglob("package.json"):
        try:
            package = json.loads(package_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        manifest = package_path.parent / "manifests/manifest.jsonl"
        if (
            package.get("inventory_sha256") == EXPECTED_INVENTORY_SHA256
            and manifest.is_file()
            and runner.file_sha256(manifest) == EXPECTED_MANIFEST_SHA256
        ):
            candidates.append(package_path.parent)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one verified extracted external gate, found {candidates}"
        )
    return candidates[0]


def validate_and_extract() -> tuple[Path, list[dict], dict, bool]:
    archives = list(
        Path("/kaggle/input").rglob("community-forensics-external-gate.zip")
    )
    if len(archives) == 1:
        archive_path = archives[0]
        observed_zip_sha256 = runner.file_sha256(archive_path)
        if observed_zip_sha256 != EXPECTED_ZIP_SHA256:
            raise RuntimeError(
                f"external gate ZIP mismatch: {observed_zip_sha256}"
            )
        if WORK_ROOT.exists():
            shutil.rmtree(WORK_ROOT)
        WORK_ROOT.mkdir(parents=True)
        with zipfile.ZipFile(archive_path) as archive:
            safe_extract(archive, WORK_ROOT)
        root = WORK_ROOT
        zip_transport_verified = True
    elif not archives:
        # Kaggle dataset inputs commonly expose the ZIP members directly and
        # omit the uploaded transport archive.  In that case every durable
        # package and image checksum below is still verified, while the report
        # explicitly records that the transport ZIP was unavailable.
        root = find_verified_extracted_root()
        zip_transport_verified = False
    else:
        raise RuntimeError(f"expected at most one external-gate archive: {archives}")

    package = json.loads((root / "package.json").read_text())
    if package.get("inventory_sha256") != EXPECTED_INVENTORY_SHA256:
        raise RuntimeError("external gate inventory mismatch")
    manifest = root / "manifests/manifest.jsonl"
    if runner.file_sha256(manifest) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("external gate manifest mismatch")
    rows = stress.read_rows(manifest)
    counts = {
        label: sum(int(row["label"]) == label for row in rows) for label in (0, 1)
    }
    models = {str(row["generator_model"]) for row in rows if int(row["label"]) == 1}
    if (
        len(rows) != EXPECTED_ROWS
        or counts != {0: EXPECTED_REAL, 1: EXPECTED_FAKE}
        or len(models) != EXPECTED_GENERATOR_MODELS
    ):
        raise RuntimeError(
            f"unexpected gate composition: rows={len(rows)} counts={counts} "
            f"models={len(models)}"
        )

    root = root.resolve()
    for index, row in enumerate(rows, 1):
        image_path = (manifest.parent / row["path"]).resolve()
        if root not in image_path.parents:
            raise RuntimeError(f"row escapes package root: {row['path']}")
        if runner.file_sha256(image_path) != row["image_sha256"]:
            raise RuntimeError(f"image hash mismatch: {row['path']}")
        if index % 100 == 0 or index == len(rows):
            print(json.dumps({"verified_gate_images": index}), flush=True)
    return manifest, rows, package, zip_transport_verified


def original_descriptors(manifest: Path, rows: list[dict]) -> list[dict]:
    descriptors = []
    for row in rows:
        with Image.open((manifest.parent / row["path"]).resolve()) as image:
            descriptors.append(
                {
                    "format": image.format,
                    "width": image.width,
                    "height": image.height,
                    "square": image.width == image.height,
                }
            )
    return descriptors


def metadata_subgroups(predictions: list[dict], descriptors: list[dict]) -> dict:
    result = {}
    selectors = {
        "all": lambda descriptor: True,
        "square": lambda descriptor: descriptor["square"],
        "jpeg": lambda descriptor: descriptor["format"] in {"JPEG", "JPG"},
        "square_jpeg": lambda descriptor: descriptor["square"]
        and descriptor["format"] in {"JPEG", "JPG"},
    }
    for name, selector in selectors.items():
        chosen = [
            prediction
            for prediction, descriptor in zip(predictions, descriptors)
            if selector(descriptor)
        ]
        labels = [int(row["label"]) for row in chosen]
        scores = [float(row["score"]) for row in chosen]
        result[name] = {
            "count": len(chosen),
            "labels": sorted(set(labels)),
            "auc": float(roc_auc_score(labels, scores))
            if len(set(labels)) == 2
            else None,
        }
    return result


def per_generator_model_metrics(rows: list[dict], predictions: list[dict]) -> dict:
    real_scores = [
        float(prediction["score"])
        for row, prediction in zip(rows, predictions)
        if int(row["label"]) == 0
    ]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        if int(row["label"]) == 1:
            grouped[str(row["generator_model"])].append(float(prediction["score"]))

    metrics = {}
    for model_name, fake_scores in sorted(grouped.items()):
        labels = [0] * len(real_scores) + [1] * len(fake_scores)
        scores = real_scores + fake_scores
        metrics[model_name] = {
            "fake_count": len(fake_scores),
            "mean_fake_score": sum(fake_scores) / len(fake_scores),
            "auc_against_all_reals": float(roc_auc_score(labels, scores)),
        }
    return metrics


def main() -> None:
    manifest, rows, package, zip_transport_verified = validate_and_extract()
    descriptors = original_descriptors(manifest, rows)
    model_name = runner.MODEL_NAMES[0]
    model_output = runner.OUTPUT_ROOT / model_name.replace(".", "_")
    checkpoint_path = model_output / "model.pt"
    if not checkpoint_path.is_file():
        raise RuntimeError(f"saved checkpoint is missing: {checkpoint_path}")
    checkpoint_sha256 = runner.file_sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = timm.create_model(
        model_name, pretrained=False, num_classes=1, img_size=runner.IMAGE_SIZE
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda().eval()
    mean = tuple(checkpoint["normalization_mean"])
    std = tuple(checkpoint["normalization_std"])

    output = model_output / "community-forensics-external-audit"
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    signature = {
        "gate_zip_sha256": EXPECTED_ZIP_SHA256,
        "gate_inventory_sha256": package["inventory_sha256"],
        "gate_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "zip_transport_verified": zip_transport_verified,
        "checkpoint_sha256": checkpoint_sha256,
        "gate_rows": len(rows),
        "generator_models": EXPECTED_GENERATOR_MODELS,
        "training_allowed": False,
    }
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        observed = {key: progress.get(key) for key in signature}
        if observed != signature:
            raise RuntimeError(
                f"refusing incompatible resume state: {observed} != {signature}"
            )
    else:
        progress = {"completed": False, **signature, "conditions": {}}

    clean_predictions = None
    pooled_predictions = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for index, (name, image_transform, tensor_noise) in enumerate(stress.conditions()):
        prediction_path = output / f"{name}_predictions.jsonl"
        if name in progress["conditions"] and prediction_path.is_file():
            predictions = stress.read_predictions(prediction_path)
            if len(predictions) != len(rows):
                raise RuntimeError(
                    f"invalid saved prediction count for {name}: {len(predictions)}"
                )
            print(json.dumps({"resumed_condition": name}), flush=True)
        else:
            torch.manual_seed(runner.SEED + index)
            dataset = runner.ManifestDataset(
                manifest,
                stress.condition_transform(mean, std, image_transform, tensor_noise),
                rows=rows,
            )
            metrics, predictions = runner.evaluate(model, dataset)
            stress.write_predictions(prediction_path, predictions)
            progress["conditions"][name] = metrics
            progress["completed"] = False
            stress.write_json(progress_path, progress)
            print(
                json.dumps({"saved_condition": name, "auc": metrics["clean_auc"]}),
                flush=True,
            )
        if name == "clean":
            clean_predictions = predictions
        else:
            pooled_predictions.extend(predictions)

    if clean_predictions is None:
        raise RuntimeError("clean predictions were not produced")
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
            "clean_metadata_subgroups": metadata_subgroups(
                clean_predictions, descriptors
            ),
            "clean_per_generator_model": per_generator_model_metrics(
                rows, clean_predictions
            ),
            "elapsed_seconds": time.time() - started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        }
    )
    stress.write_json(progress_path, progress)
    print(json.dumps(progress, indent=2), flush=True)


if __name__ == "__main__":
    main()
