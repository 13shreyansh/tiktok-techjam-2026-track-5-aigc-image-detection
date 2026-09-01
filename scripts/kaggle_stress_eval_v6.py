"""Run the workshop's full individual-transform matrix on the saved v6 model.

This script deliberately reuses the package and checkpoint already verified by
``kaggle_train_v6.py``.  It evaluates a deterministic source-balanced subset,
stores every condition as it completes, and adds a JPEG-normalized clean
diagnostic to expose reliance on source codec fingerprints.  The diagnostic is
not part of the organizer-style score.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import timm
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision.transforms import v2

import kaggle_train_v3 as runner
import kaggle_train_v6  # noqa: F401  # apply checksum-pinned v6 constants


FAKE_PER_GENERATOR = 192
REAL_PER_SOURCE = 307


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def rank(row: dict) -> str:
    identity = str(row.get("image_sha256") or row.get("archive_member") or row["path"])
    return hashlib.sha256(f"{runner.SEED}:{identity}".encode()).hexdigest()


def balanced_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        label = int(row["label"])
        group = str(
            row.get("generator", "unknown")
            if label == 1
            else row.get("real_source", "unknown")
        )
        grouped[(label, group)].append(row)
    selected = []
    for (label, group), candidates in sorted(grouped.items()):
        limit = FAKE_PER_GENERATOR if label == 1 else REAL_PER_SOURCE
        if len(candidates) < limit:
            raise RuntimeError(f"insufficient {label=} {group=}: {len(candidates)} < {limit}")
        selected.extend(sorted(candidates, key=rank)[:limit])
    selected.sort(
        key=lambda row: (
            int(row["label"]),
            str(row.get("real_source") or row.get("generator")),
            str(row["path"]),
        )
    )
    return selected


def condition_transform(
    mean: tuple[float, ...],
    std: tuple[float, ...],
    image_transform=None,
    tensor_noise: float | None = None,
):
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
                lambda tensor: (tensor + torch.randn_like(tensor) * tensor_noise).clamp(
                    0.0, 1.0
                )
            )
        )
    operations.append(v2.Normalize(mean, std))
    return v2.Compose(operations)


def conditions() -> list[tuple[str, object | None, float | None]]:
    result: list[tuple[str, object | None, float | None]] = [("clean", None, None)]
    result += [(f"jpeg_q{quality}", runner.JpegCompression(quality), None) for quality in (90, 70, 50, 30)]
    result += [(f"blur_sigma_{sigma:g}", runner.GaussianBlurPIL(sigma), None) for sigma in (0.5, 1.0, 2.0)]
    result += [(f"resize_{scale:g}", runner.DownUpResize(scale), None) for scale in (0.5, 0.25)]
    result += [(f"noise_sigma_{sigma:.2f}", None, sigma) for sigma in (0.02, 0.05, 0.10)]
    result += [
        (f"{kind}_{factor:g}", runner.FixedEnhancement(kind, factor), None)
        for kind in ("brightness", "contrast", "saturation")
        for factor in (0.8, 1.2)
    ]
    result.append(("center_crop_80", runner.CenterCropFraction(0.8), None))
    return result


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_predictions(path: Path, predictions: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions)
    )
    temporary.replace(path)


def read_predictions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    model_name = runner.MODEL_NAMES[0]
    model_output = runner.OUTPUT_ROOT / model_name.replace(".", "_")
    checkpoint_path = model_output / "model.pt"
    if not checkpoint_path.is_file():
        raise RuntimeError(f"saved checkpoint is missing: {checkpoint_path}")

    package_sha256, metadata = runner.validate_package()
    eval_manifest = runner.WORK_ROOT / "manifests/eval_selection.jsonl"
    rows = runner.filter_evaluation_rows(
        read_rows(eval_manifest), runner.EXCLUDED_EVAL_SHA256
    )
    selected = balanced_rows(rows)
    if len(selected) != 3071:
        raise RuntimeError(f"balanced gate has {len(selected)} rows, expected 3071")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = timm.create_model(
        model_name, pretrained=False, num_classes=1, img_size=runner.IMAGE_SIZE
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda().eval()
    mean = tuple(checkpoint["normalization_mean"])
    std = tuple(checkpoint["normalization_std"])

    output = model_output / "stress-eval"
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    signature = {
        "package_sha256": package_sha256,
        "inventory_sha256": metadata["inventory_sha256"],
        "checkpoint_sha256": runner.file_sha256(checkpoint_path),
        "gate_rows": len(selected),
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
    for index, (name, image_transform, tensor_noise) in enumerate(conditions()):
        prediction_path = output / f"{name}_predictions.jsonl"
        if name in progress["conditions"] and prediction_path.is_file():
            predictions = read_predictions(prediction_path)
            if len(predictions) != len(selected):
                raise RuntimeError(
                    f"invalid saved prediction count for {name}: {len(predictions)}"
                )
            print(json.dumps({"resumed_condition": name}), flush=True)
        else:
            torch.manual_seed(runner.SEED + index)
            dataset = runner.ManifestDataset(
                eval_manifest,
                condition_transform(mean, std, image_transform, tensor_noise),
                rows=selected,
            )
            metrics, predictions = runner.evaluate(model, dataset)
            write_predictions(prediction_path, predictions)
            progress["conditions"][name] = metrics
            progress["completed"] = False
            write_json(progress_path, progress)
            print(json.dumps({"saved_condition": name, "auc": metrics["clean_auc"]}), flush=True)
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

    # Every image is decoded and re-encoded identically before inference.  A
    # collapse here would be evidence that the clean score relied on source
    # codec traces rather than general AI-generation evidence.
    codec_dataset = runner.ManifestDataset(
        eval_manifest,
        condition_transform(mean, std, runner.JpegCompression(96), None),
        rows=selected,
    )
    codec_metrics, codec_predictions = runner.evaluate(model, codec_dataset)
    progress.update(
        {
            "completed": True,
            "official_style": {
                "clean_auc": clean_auc,
                "pooled_robust_auc": robust_auc,
                "score": 0.5 * clean_auc + 0.5 * robust_auc,
            },
            "pooled_robust_groups": runner.grouped_metrics(
                selected * (len(conditions()) - 1), robust_labels, robust_scores
            ),
            "codec_normalized_jpeg_q96": codec_metrics,
            "elapsed_seconds": time.time() - started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        }
    )
    write_json(progress_path, progress)
    write_predictions(output / "clean_predictions.jsonl", clean_predictions)
    write_predictions(output / "codec_q96_predictions.jsonl", codec_predictions)
    print(json.dumps(progress, indent=2), flush=True)


if __name__ == "__main__":
    main()
