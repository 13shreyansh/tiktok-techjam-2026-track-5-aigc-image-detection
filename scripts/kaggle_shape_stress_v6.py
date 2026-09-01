"""Stress the saved PE-Core candidate against codec and geometry shortcuts.

These diagnostics are intentionally separate from the organizer's 19-condition
score.  Each condition is label-independent and is selected from immutable
image identity, never from the real/fake label.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import timm
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
from torchvision.transforms import v2

import kaggle_stress_eval_v6 as stress
import kaggle_train_v3 as runner
import kaggle_train_v6  # noqa: F401  # apply checksum-pinned v6 constants


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def identity(row: dict) -> str:
    return str(row.get("image_sha256") or row.get("archive_member") or row["path"])


def inspect_originals(manifest: Path, rows: list[dict]) -> list[dict]:
    descriptors = []
    for row in rows:
        path = (manifest.parent / row["path"]).resolve()
        with Image.open(path) as image:
            descriptors.append(
                {
                    "square": image.width == image.height,
                    "jpeg": image.format in {"JPEG", "JPG"},
                }
            )
    return descriptors


def metadata_subgroups(predictions: list[dict], descriptors: list[dict]) -> dict:
    selectors = {
        "all_originals": lambda item: True,
        "square_originals": lambda item: item[1]["square"],
        "jpeg_originals": lambda item: item[1]["jpeg"],
        "square_jpeg_originals": lambda item: item[1]["square"] and item[1]["jpeg"],
    }
    result = {}
    paired = list(zip(predictions, descriptors))
    for name, selector in selectors.items():
        selected = [prediction for prediction, descriptor in paired if selector((prediction, descriptor))]
        labels = [int(row["label"]) for row in selected]
        scores = [float(row["score"]) for row in selected]
        result[name] = {
            "count": len(selected),
            "labels": sorted(set(labels)),
            "auc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
        }
    return result


def centered_crop_to_ratio(image: Image.Image, ratio: float) -> Image.Image:
    width, height = image.size
    current = width / height
    if current > ratio:
        crop_height = height
        crop_width = max(1, round(height * ratio))
    else:
        crop_width = width
        crop_height = max(1, round(width / ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))


def deterministic_square_patch(image: Image.Image, key: str, fraction: float = 0.75) -> Image.Image:
    width, height = image.size
    side = max(1, round(min(width, height) * fraction))
    digest = hashlib.sha256(f"{runner.SEED}:{key}:square-patch".encode()).digest()
    x_fraction = int.from_bytes(digest[:8], "big") / (2**64 - 1)
    y_fraction = int.from_bytes(digest[8:16], "big") / (2**64 - 1)
    left = round((width - side) * x_fraction)
    top = round((height - side) * y_fraction)
    return image.crop((left, top, left + side, top + side))


class GeometryDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        rows: list[dict],
        mean: tuple[float, ...],
        std: tuple[float, ...],
        condition: str,
    ) -> None:
        self.manifest = manifest
        self.rows = rows
        self.condition = condition
        self.jpeg = runner.JpegCompression(96)
        self.finish = v2.Compose(
            [
                v2.Resize((runner.IMAGE_SIZE, runner.IMAGE_SIZE), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean, std),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = (self.manifest.parent / row["path"]).resolve()
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        key = identity(row)
        if self.condition == "jpeg_q96_stretch_full_frame":
            transformed = image
        elif self.condition == "jpeg_q96_square_patch_75":
            transformed = deterministic_square_patch(image, key)
        elif self.condition == "jpeg_q96_forced_aspect":
            ratios = (4 / 3, 3 / 4, 16 / 9, 9 / 16)
            bucket = int(hashlib.sha256(f"{runner.SEED}:{key}:aspect".encode()).hexdigest(), 16)
            transformed = centered_crop_to_ratio(image, ratios[bucket % len(ratios)])
        else:
            raise ValueError(f"unknown condition: {self.condition}")
        tensor = self.finish(self.jpeg(transformed))
        return tensor, int(row["label"]), index


def main() -> None:
    model_name = runner.MODEL_NAMES[0]
    model_output = runner.OUTPUT_ROOT / model_name.replace(".", "_")
    checkpoint_path = model_output / "model.pt"
    if not checkpoint_path.is_file():
        raise RuntimeError(f"saved checkpoint is missing: {checkpoint_path}")

    package_sha256, metadata = runner.validate_package()
    eval_manifest = runner.WORK_ROOT / "manifests/eval_selection.jsonl"
    rows = runner.filter_evaluation_rows(
        stress.read_rows(eval_manifest), runner.EXCLUDED_EVAL_SHA256
    )
    selected = stress.balanced_rows(rows)
    if len(selected) != 3071:
        raise RuntimeError(f"balanced gate has {len(selected)} rows, expected 3071")
    descriptors = inspect_originals(eval_manifest, selected)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = timm.create_model(
        model_name, pretrained=False, num_classes=1, img_size=runner.IMAGE_SIZE
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda().eval()
    mean = tuple(checkpoint["normalization_mean"])
    std = tuple(checkpoint["normalization_std"])

    output = model_output / "shortcut-stress"
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    result = {
        "completed": False,
        "purpose": "diagnostic only; not part of the organizer-style score",
        "package_sha256": package_sha256,
        "inventory_sha256": metadata["inventory_sha256"],
        "checkpoint_sha256": runner.file_sha256(checkpoint_path),
        "gate_rows": len(selected),
        "conditions": {},
    }
    conditions = (
        "jpeg_q96_stretch_full_frame",
        "jpeg_q96_square_patch_75",
        "jpeg_q96_forced_aspect",
    )
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for condition in conditions:
        dataset = GeometryDataset(eval_manifest, selected, mean, std, condition)
        metrics, predictions = runner.evaluate(model, dataset)
        metrics["original_metadata_subgroups"] = metadata_subgroups(
            predictions, descriptors
        )
        result["conditions"][condition] = metrics
        (output / f"{condition}_predictions.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions)
        )
        write_json(progress_path, result)
        print(json.dumps({"saved_condition": condition, "auc": metrics["clean_auc"]}), flush=True)

    result.update(
        {
            "completed": True,
            "elapsed_seconds": time.time() - started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        }
    )
    write_json(progress_path, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
