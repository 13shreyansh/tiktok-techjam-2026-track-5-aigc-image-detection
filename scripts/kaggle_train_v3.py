"""Self-contained Kaggle P100 comparison on the source-controlled v4 mixture.

The private package contains only manifest-selected public/licensed images.  It
is checksum verified before extraction, and the script refuses any train/eval
content overlap.  FLUX and Stable Diffusion 3 are present only in training;
PixArt-Sigma and Imagen are held-out generators; LAION-5B is a held-out real
source. Source-balanced sampling prevents one large branch dominating an epoch.

Fill the two expected package hashes from the package provenance JSON before
running this file in Kaggle.
"""

from __future__ import annotations

import gc
import hashlib
import io
import json
import random
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2


SEED = 20260829
IMAGE_SIZE = 224
PACKAGE_NAME = "family-mixture-v4-dedup.zip"
EXPECTED_ZIP_SHA256 = "90421d04d8f70d391b28e703d0f7aca94393471336696e4138138107379f017e"
EXPECTED_INVENTORY_SHA256 = "4239ddf2846a3b2d05ac03fbec872d7b0f8cbd5d606c59c52bc7c198df288a69"
WORK_ROOT = Path("/kaggle/working/family-mixture-v4")
OUTPUT_ROOT = Path("/kaggle/working/track5-v4-candidates")
INPUT_ROOT = Path("/kaggle/input")
MODEL_NAMES = (
    "vit_large_patch14_dinov2.lvd142m",
    "vit_pe_core_large_patch14_336",
)
PARAMETER_LIMIT = 2_000_000_000
EXPECTED_TRAIN_ROWS = 16910
EXPECTED_EVAL_ROWS = 12326
EXPECTED_CONTENT_EVAL_ROWS = 7916
EXCLUDED_EVAL_SHA256: set[str] = set()
AUGMENTATION_DESCRIPTION = "at_most_one_workshop_listed_transformation"
CODEC_NORMALIZATION = "none"
PREPROCESS_MODE = "short_side_crop"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mounted_root_files(name: str) -> list[Path]:
    """Locate root files in both current and legacy Kaggle mount layouts."""
    patterns = (f"*/{name}", f"datasets/*/*/{name}")
    return sorted(
        {
            path
            for pattern in patterns
            for path in INPUT_ROOT.glob(pattern)
            if path.is_file()
        }
    )


class JpegCompression:
    def __init__(self, quality: int) -> None:
        self.quality = quality

    def __call__(self, image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")


class GaussianBlurPIL:
    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(self.sigma))


class DownUpResize:
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        down = (max(1, round(width * self.scale)), max(1, round(height * self.scale)))
        return image.resize(down, Image.Resampling.BICUBIC).resize(
            (width, height), Image.Resampling.BICUBIC
        )


class CenterCropFraction:
    def __init__(self, fraction: float) -> None:
        self.fraction = fraction

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        crop_width = max(1, round(width * self.fraction))
        crop_height = max(1, round(height * self.fraction))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        return image.crop((left, top, left + crop_width, top + crop_height))


class FixedEnhancement:
    def __init__(self, kind: str, factor: float) -> None:
        self.kind = kind
        self.factor = factor

    def __call__(self, image: Image.Image) -> Image.Image:
        enhancer = {
            "brightness": ImageEnhance.Brightness,
            "contrast": ImageEnhance.Contrast,
            "saturation": ImageEnhance.Color,
        }[self.kind]
        return enhancer(image).enhance(self.factor)


class GaussianNoisePIL:
    def __init__(self, sigma: float) -> None:
        self.sigma = sigma
        self.to_image = v2.ToImage()
        self.to_float = v2.ToDtype(torch.float32, scale=True)
        self.to_pil = v2.ToPILImage()

    def __call__(self, image: Image.Image) -> Image.Image:
        tensor = self.to_float(self.to_image(image))
        return self.to_pil(
            (tensor + torch.randn_like(tensor) * self.sigma).clamp(0.0, 1.0)
        )


class RandomSingleOfficialTransform:
    """Apply at most one workshop-listed redistribution operation."""

    def __init__(self, probability: float = 0.8) -> None:
        self.probability = probability
        self.transforms = (
            [JpegCompression(quality) for quality in (90, 70, 50, 30)]
            + [GaussianBlurPIL(sigma) for sigma in (0.5, 1.0, 2.0)]
            + [DownUpResize(scale) for scale in (0.5, 0.25)]
            + [GaussianNoisePIL(sigma) for sigma in (0.02, 0.05, 0.10)]
            + [
                FixedEnhancement(kind, factor)
                for kind in ("brightness", "contrast", "saturation")
                for factor in (0.8, 1.2)
            ]
            + [CenterCropFraction(0.8)]
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        return random.choice(self.transforms)(image)


def normalization(model: torch.nn.Module) -> tuple[tuple[float, ...], tuple[float, ...]]:
    config = getattr(model, "pretrained_cfg", {})
    return tuple(config.get("mean", (0.485, 0.456, 0.406))), tuple(
        config.get("std", (0.229, 0.224, 0.225))
    )


def train_transform(mean: tuple[float, ...], std: tuple[float, ...]):
    transforms = [RandomSingleOfficialTransform()]
    if CODEC_NORMALIZATION == "jpeg_q96":
        transforms.append(JpegCompression(96))
    elif CODEC_NORMALIZATION != "none":
        raise ValueError(f"unknown codec normalization: {CODEC_NORMALIZATION}")
    transforms.extend(
        [
            v2.Resize(IMAGE_SIZE, antialias=True),
            v2.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std),
        ]
    )
    return v2.Compose(transforms)


def eval_transform(mean: tuple[float, ...], std: tuple[float, ...]):
    # Preserve the input aspect ratio and crop rather than stretching every
    # rectangular real photograph into a square.
    transforms = []
    if CODEC_NORMALIZATION == "jpeg_q96":
        transforms.append(JpegCompression(96))
    elif CODEC_NORMALIZATION != "none":
        raise ValueError(f"unknown codec normalization: {CODEC_NORMALIZATION}")
    transforms.extend(
        [
            v2.Resize(IMAGE_SIZE, antialias=True),
            v2.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std),
        ]
    )
    return v2.Compose(transforms)


class ManifestDataset(Dataset):
    def __init__(self, manifest: Path, transform, rows: list[dict] | None = None) -> None:
        self.manifest = manifest
        self.transform = transform
        self.rows = rows if rows is not None else [
            json.loads(line) for line in manifest.read_text().splitlines() if line
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = (self.manifest.parent / row["path"]).resolve()
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label"]), index


def source_balanced_sampler(rows: list[dict]) -> tuple[WeightedRandomSampler, dict]:
    groups = [
        (
            int(row["label"]),
            str(
                row.get("generator", "unknown")
                if int(row["label"]) == 1
                else row.get("real_source", "unknown")
            ),
        )
        for row in rows
    ]
    counts = Counter(groups)
    groups_per_label = Counter(label for label, _ in counts)
    weights = [
        1.0 / (groups_per_label[label] * counts[(label, group)])
        for label, group in groups
    ]
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(rows),
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    report = {
        "policy": "equal labels; equal named sources within each label",
        "groups_per_label": {
            str(label): groups_per_label[label] for label in sorted(groups_per_label)
        },
        "group_counts": {
            f'{"fake" if label else "real"}:{group}': count
            for (label, group), count in sorted(counts.items())
        },
    }
    return sampler, report


def auc(selected: list[tuple[int, float]]) -> float:
    labels, scores = zip(*selected)
    return float(roc_auc_score(labels, scores))


def filter_evaluation_rows(rows: list[dict], excluded_hashes: set[str]) -> list[dict]:
    """Exclude review-confirmed near duplicates by immutable image hash."""
    if not excluded_hashes:
        return rows
    observed = {row.get("image_sha256") for row in rows} & excluded_hashes
    missing = excluded_hashes - observed
    if missing:
        raise RuntimeError(f"requested evaluation exclusions are absent: {sorted(missing)}")
    return [row for row in rows if row.get("image_sha256") not in excluded_hashes]


def grouped_metrics(rows: list[dict], labels: list[int], scores: list[float]) -> dict:
    generators = sorted(
        {str(row.get("generator", "unknown")) for row in rows if int(row["label"]) == 1}
    )
    real_sources = sorted(
        {str(row.get("real_source", "unknown")) for row in rows if int(row["label"]) == 0}
    )
    fake_groups = {}
    real_groups = {}
    pairs: dict[str, dict] = defaultdict(dict)
    pair_values = []
    for generator in generators:
        selected = [
            (label, score)
            for row, label, score in zip(rows, labels, scores)
            if label == 0 or str(row.get("generator", "unknown")) == generator
        ]
        fake_groups[generator] = auc(selected)
    for source in real_sources:
        selected = [
            (label, score)
            for row, label, score in zip(rows, labels, scores)
            if label == 1 or str(row.get("real_source", "unknown")) == source
        ]
        real_groups[source] = auc(selected)
    for generator in generators:
        for source in real_sources:
            selected = [
                (label, score)
                for row, label, score in zip(rows, labels, scores)
                if (label == 1 and str(row.get("generator", "unknown")) == generator)
                or (label == 0 and str(row.get("real_source", "unknown")) == source)
            ]
            value = auc(selected)
            pairs[generator][source] = value
            pair_values.append(value)
    return {
        "fake_generator_auc": fake_groups,
        "real_source_auc": real_groups,
        "generator_real_source_pair_auc": dict(pairs),
        "worst_fake_generator_auc": min(fake_groups.values()),
        "worst_real_source_auc": min(real_groups.values()),
        "worst_generator_real_source_pair_auc": min(pair_values),
    }


@torch.inference_mode()
def evaluate(model: torch.nn.Module, dataset: ManifestDataset) -> tuple[dict, list[dict]]:
    loader = DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True
    )
    labels: list[int] = []
    scores: list[float] = []
    indices: list[int] = []
    model.eval()
    for batch_number, (images, batch_labels, batch_indices) in enumerate(loader, 1):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            probabilities = torch.sigmoid(
                model(images.cuda(non_blocking=True)).flatten()
            )
        labels.extend(int(value) for value in batch_labels.tolist())
        scores.extend(float(value) for value in probabilities.float().cpu().tolist())
        indices.extend(int(value) for value in batch_indices.tolist())
        if batch_number % 25 == 0 or batch_number == len(loader):
            print(
                json.dumps(
                    {
                        "phase": "evaluate",
                        "batch": batch_number,
                        "batches": len(loader),
                        "images": len(labels),
                    }
                ),
                flush=True,
            )
    predictions = [
        {
            "index": index,
            "label": label,
            "score": score,
            **{
                key: dataset.rows[index][key]
                for key in ("generator", "real_source", "family", "image_sha256")
                if key in dataset.rows[index]
            },
        }
        for index, label, score in zip(indices, labels, scores)
    ]
    result = {
        "count": len(labels),
        "clean_auc": float(roc_auc_score(labels, scores)),
        "groups": grouped_metrics(dataset.rows, labels, scores),
    }
    return result, predictions


def select_evaluation_from_predictions(
    selection_predictions: list[dict], content_rows: list[dict]
) -> tuple[dict, list[dict]]:
    """Derive a subset gate without running the frozen backbone a second time."""
    by_hash: dict[str, list[dict]] = {}
    for prediction in selection_predictions:
        image_sha256 = prediction.get("image_sha256")
        if not image_sha256:
            raise RuntimeError("selection prediction is missing image_sha256")
        by_hash.setdefault(image_sha256, []).append(prediction)
    canonical_by_hash: dict[str, dict] = {}
    for image_sha256, duplicates in by_hash.items():
        labels = {int(prediction["label"]) for prediction in duplicates}
        scores = [float(prediction["score"]) for prediction in duplicates]
        if len(labels) != 1:
            raise RuntimeError(
                f"conflicting labels for duplicate selection image: {image_sha256}"
            )
        if max(scores) - min(scores) > 1e-6:
            raise RuntimeError(
                f"conflicting scores for duplicate selection image: {image_sha256}"
            )
        canonical_by_hash[image_sha256] = duplicates[0]

    subset_predictions: list[dict] = []
    subset_rows: list[dict] = []
    for index, row in enumerate(content_rows):
        image_sha256 = row.get("image_sha256")
        if not image_sha256 or image_sha256 not in canonical_by_hash:
            raise RuntimeError(
                f"content image is absent from selection predictions: {image_sha256}"
            )
        selected = {
            **canonical_by_hash[image_sha256],
            **{
                key: value
                for key, value in row.items()
                if key not in {"score", "index"}
            },
            "index": index,
        }
        if int(selected["label"]) != int(row["label"]):
            raise RuntimeError(f"label mismatch for content image: {image_sha256}")
        subset_predictions.append(selected)
        subset_rows.append(row)

    labels = [int(prediction["label"]) for prediction in subset_predictions]
    scores = [float(prediction["score"]) for prediction in subset_predictions]
    result = {
        "count": len(labels),
        "clean_auc": float(roc_auc_score(labels, scores)),
        "groups": grouped_metrics(subset_rows, labels, scores),
    }
    return result, subset_predictions


def validate_package() -> tuple[str, dict]:
    global WORK_ROOT
    # Kaggle mounts every attached dataset directly below /kaggle/input.  Do
    # not recursively walk every image in every unrelated attachment merely
    # to locate one root-level package file; historical notebooks can contain
    # hundreds of thousands of mounted files.
    packages = mounted_root_files(PACKAGE_NAME)
    if len(packages) == 1:
        package = packages[0]
        observed_sha256 = file_sha256(package)
        if observed_sha256 != EXPECTED_ZIP_SHA256:
            raise RuntimeError(f"package checksum mismatch: {observed_sha256}")
        if not WORK_ROOT.exists():
            WORK_ROOT.mkdir(parents=True)
            with zipfile.ZipFile(package) as archive:
                archive.extractall(WORK_ROOT)
        metadata = json.loads((WORK_ROOT / "package.json").read_text())
        metadata["runtime_verification"] = "uploaded ZIP byte checksum"
    elif len(packages) == 0:
        # Kaggle expands uploaded ZIP datasets. In that case verify every
        # extracted image against its content-addressed filename, plus every
        # packaged manifest against the embedded immutable metadata.
        candidates = []
        for report_path in mounted_root_files("package.json"):
            report = json.loads(report_path.read_text())
            if report.get("inventory_sha256") == EXPECTED_INVENTORY_SHA256:
                candidates.append((report_path.parent, report))
        if len(candidates) != 1:
            raise RuntimeError(
                "expected one extracted dataset with inventory "
                f"{EXPECTED_INVENTORY_SHA256}, found {[str(path) for path, _ in candidates]}"
            )
        WORK_ROOT, metadata = candidates[0]
        image_paths = sorted(path for path in (WORK_ROOT / "images").rglob("*") if path.is_file())
        if len(image_paths) != int(metadata["unique_images"]):
            raise RuntimeError(
                f"extracted image count mismatch: {len(image_paths)} != {metadata['unique_images']}"
            )
        observed_bytes = 0
        for index, image_path in enumerate(image_paths, start=1):
            observed = file_sha256(image_path)
            if observed != image_path.stem:
                raise RuntimeError(f"content-address mismatch: {image_path}")
            observed_bytes += image_path.stat().st_size
            if index % 2000 == 0 or index == len(image_paths):
                print(f"verified extracted images {index}/{len(image_paths)}", flush=True)
        if observed_bytes != int(metadata["source_bytes"]):
            raise RuntimeError(
                f"extracted byte count mismatch: {observed_bytes} != {metadata['source_bytes']}"
            )
        for manifest in metadata["manifests"]:
            manifest_path = WORK_ROOT / manifest["packaged_manifest"]
            if file_sha256(manifest_path) != manifest["sha256"]:
                raise RuntimeError(f"manifest checksum mismatch: {manifest_path}")
        observed_sha256 = EXPECTED_ZIP_SHA256
        metadata["runtime_verification"] = (
            "Kaggle-expanded content: every image SHA-256 plus manifest checksums; "
            "package SHA-256 is the locally recorded upload source"
        )
    else:
        raise RuntimeError(f"expected at most one {PACKAGE_NAME}, found {packages}")
    if metadata["inventory_sha256"] != EXPECTED_INVENTORY_SHA256:
        raise RuntimeError("package inventory mismatch")
    return observed_sha256, metadata


def train_candidate(
    model_name: str,
    train_manifest: Path,
    eval_manifest: Path,
    content_eval_manifest: Path,
    package_sha256: str,
    inventory_sha256: str,
) -> dict:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model = timm.create_model(
        model_name, pretrained=True, num_classes=1, img_size=IMAGE_SIZE
    ).cuda()
    mean, std = normalization(model)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.get_classifier().parameters():
        parameter.requires_grad = True
    parameters = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    if parameters["total"] >= PARAMETER_LIMIT:
        raise RuntimeError(f"parameter limit exceeded: {parameters}")

    train_dataset = ManifestDataset(train_manifest, train_transform(mean, std))
    eval_rows = [
        json.loads(line) for line in eval_manifest.read_text().splitlines() if line
    ]
    content_eval_rows = [
        json.loads(line)
        for line in content_eval_manifest.read_text().splitlines()
        if line
    ]
    eval_rows = filter_evaluation_rows(eval_rows, EXCLUDED_EVAL_SHA256)
    content_eval_rows = filter_evaluation_rows(
        content_eval_rows, EXCLUDED_EVAL_SHA256
    )
    eval_dataset = ManifestDataset(
        eval_manifest, eval_transform(mean, std), rows=eval_rows
    )
    sampler, sampling_report = source_balanced_sampler(train_dataset.rows)
    loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )
    learning_rate = 0.001
    optimizer = torch.optim.AdamW(
        model.get_classifier().parameters(), lr=learning_rate, weight_decay=0.0001
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    model.train()
    running_loss = 0.0
    seen = 0
    for step, (images, labels, _) in enumerate(loader, 1):
        images = images.cuda(non_blocking=True)
        labels = labels.to(device="cuda", dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = criterion(model(images).flatten(), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += float(loss.detach()) * len(labels)
        seen += len(labels)
        if step % 20 == 0 or step == len(loader):
            print(
                json.dumps(
                    {
                        "model": model_name,
                        "step": step,
                        "steps": len(loader),
                        "loss": running_loss / seen,
                    }
                ),
                flush=True,
            )

    # Persist the trained state before the comparatively long evaluation pass.
    # Kaggle sessions can reconnect or be interrupted after training; delaying
    # this write until evaluation completes previously lost a fully trained
    # PE-Core candidate.
    model_output = OUTPUT_ROOT / model_name.replace(".", "_")
    model_output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "state_dict": model.state_dict(),
        "model_name": model_name,
        "image_size": IMAGE_SIZE,
        "normalization_mean": mean,
        "normalization_std": std,
        "preprocess_mode": PREPROCESS_MODE,
        "codec_normalization": CODEC_NORMALIZATION,
        "parameters": parameters,
        "seed": SEED,
    }
    checkpoint_path = model_output / "model.pt"
    torch.save(checkpoint, checkpoint_path)
    print(
        json.dumps(
            {
                "phase": "checkpoint_saved",
                "model": model_name,
                "path": str(checkpoint_path),
                "final_loss": running_loss / seen,
            }
        ),
        flush=True,
    )

    evaluation, predictions = evaluate(model, eval_dataset)
    content_evaluation, content_predictions = select_evaluation_from_predictions(
        predictions, content_eval_rows
    )
    report = {
        "seed": SEED,
        "model": model_name,
        "image_size": IMAGE_SIZE,
        "preprocess_mode": PREPROCESS_MODE,
        "codec_normalization": CODEC_NORMALIZATION,
        "augmentation": AUGMENTATION_DESCRIPTION,
        "sampling": sampling_report,
        "learning_rate": learning_rate,
        "parameters": parameters,
        "package_sha256": package_sha256,
        "inventory_sha256": inventory_sha256,
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset),
        "excluded_eval_sha256": sorted(EXCLUDED_EVAL_SHA256),
        "final_loss": running_loss / seen,
        "selection_clean": evaluation,
        "content_holdout_clean": content_evaluation,
        "elapsed_seconds": time.time() - started,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    (model_output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (model_output / "selection_predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions)
    )
    (model_output / "content_holdout_predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in content_predictions)
    )
    print(json.dumps(report, indent=2), flush=True)
    del model, optimizer, scaler, loader, train_dataset, eval_dataset
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    package_sha256, metadata = validate_package()
    train_manifest = WORK_ROOT / "manifests/train.jsonl"
    eval_manifest = WORK_ROOT / "manifests/eval_selection.jsonl"
    content_eval_manifest = WORK_ROOT / "manifests/eval_content_holdout.jsonl"
    train_rows = [json.loads(line) for line in train_manifest.read_text().splitlines() if line]
    eval_rows = [json.loads(line) for line in eval_manifest.read_text().splitlines() if line]
    content_eval_rows = [
        json.loads(line)
        for line in content_eval_manifest.read_text().splitlines()
        if line
    ]
    train_hashes = {row["image_sha256"] for row in train_rows}
    eval_hashes = {row["image_sha256"] for row in eval_rows}
    content_eval_hashes = {row["image_sha256"] for row in content_eval_rows}
    overlap = train_hashes & (eval_hashes | content_eval_hashes)
    if overlap:
        raise RuntimeError(f"train/eval content overlap: {len(overlap)}")
    if (
        len(train_rows) != EXPECTED_TRAIN_ROWS
        or len(eval_rows) != EXPECTED_EVAL_ROWS
        or len(content_eval_rows) != EXPECTED_CONTENT_EVAL_ROWS
    ):
        raise RuntimeError(
            "unexpected manifest sizes: "
            f"train={len(train_rows)}, eval={len(eval_rows)}, "
            f"content_eval={len(content_eval_rows)}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    reports = []
    for model_name in MODEL_NAMES:
        reports.append(
            train_candidate(
                model_name,
                train_manifest,
                eval_manifest,
                content_eval_manifest,
                package_sha256,
                metadata["inventory_sha256"],
            )
        )
    summary = {
        "models": [
            {
                "model": report["model"],
                "clean_auc": report["selection_clean"]["clean_auc"],
                "worst_fake_generator_auc": report["selection_clean"]["groups"][
                    "worst_fake_generator_auc"
                ],
                "worst_real_source_auc": report["selection_clean"]["groups"][
                    "worst_real_source_auc"
                ],
                "worst_generator_real_source_pair_auc": report["selection_clean"][
                    "groups"
                ]["worst_generator_real_source_pair_auc"],
                "content_holdout_clean_auc": report["content_holdout_clean"][
                    "clean_auc"
                ],
                "content_holdout_worst_pair_auc": report["content_holdout_clean"][
                    "groups"
                ]["worst_generator_real_source_pair_auc"],
            }
            for report in reports
        ]
    }
    (OUTPUT_ROOT / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
