"""Self-contained Kaggle P100 gate for PE-Core-L on family mixture v2.

Paste this file into a Kaggle cell after installing the pinned PyTorch/timm
versions documented below. The uploaded ZIP stays private and is checksum
verified before extraction. This script intentionally evaluates only the two
manifests packaged with v2; it does not touch organizer demo-only data.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


SEED = 20260829
MODEL_NAME = "vit_pe_core_large_patch14_336"
IMAGE_SIZE = 224
EXPECTED_ZIP_SHA256 = "d6db56897fc4fa855349e5a5992481b0c81d2c39cb5ed91f7638f6e7171fb709"
EXPECTED_INVENTORY_SHA256 = "66257154698bcfb1bb5a7dbc91a2a58d951425cb237bc014cc3f3cc583e3ac23"
WORK_ROOT = Path("/kaggle/working/family-mixture-v2")
OUTPUT_ROOT = Path("/kaggle/working/pe-core-family-mixture-v2")
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class JpegCompression:
    def __init__(self, quality: int) -> None:
        self.quality = quality

    def __call__(self, image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")


class DownUpResize:
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        down = (max(1, round(width * self.scale)), max(1, round(height * self.scale)))
        return image.resize(down, Image.Resampling.BICUBIC).resize(
            (width, height), Image.Resampling.BICUBIC
        )


class GaussianBlurPIL:
    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(self.sigma))


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


class RandomRedistribution:
    def __init__(self, probability: float = 0.8) -> None:
        self.probability = probability
        self.transforms = (
            [JpegCompression(quality) for quality in (90, 70, 50, 30)]
            + [GaussianBlurPIL(sigma) for sigma in (0.5, 1.0, 2.0)]
            + [DownUpResize(scale) for scale in (0.5, 0.25)]
            + [CenterCropFraction(0.8)]
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        return random.choice(self.transforms)(image)


class RandomGaussianNoise(torch.nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= 0.25:
            return tensor
        sigma = random.choice((0.02, 0.05, 0.10))
        return (tensor + torch.randn_like(tensor) * sigma).clamp(0.0, 1.0)


TRAIN_TRANSFORM = v2.Compose(
    [
        RandomRedistribution(),
        v2.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.65, 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)], p=0.5),
        v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0))], p=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        RandomGaussianNoise(),
        v2.Normalize(MEAN, STD),
    ]
)
EVAL_TRANSFORM = v2.Compose(
    [
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(MEAN, STD),
    ]
)


class ManifestDataset(Dataset):
    def __init__(self, manifest: Path, transform) -> None:
        self.manifest = manifest
        self.transform = transform
        self.rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = (self.manifest.parent / row["path"]).resolve()
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label"]), index


def grouped_metrics(rows: list[dict], labels: list[int], scores: list[float]) -> dict:
    generators = sorted({str(row.get("generator", "unknown")) for row in rows if int(row["label"]) == 1})
    real_sources = sorted({str(row.get("real_source", "unknown")) for row in rows if int(row["label"]) == 0})
    fake_groups = {}
    real_groups = {}
    pairs: dict[str, dict] = defaultdict(dict)
    for generator in generators:
        selected = [
            (label, score)
            for row, label, score in zip(rows, labels, scores)
            if label == 0 or str(row.get("generator", "unknown")) == generator
        ]
        fake_groups[generator] = float(roc_auc_score(*zip(*selected)))
    for source in real_sources:
        selected = [
            (label, score)
            for row, label, score in zip(rows, labels, scores)
            if label == 1 or str(row.get("real_source", "unknown")) == source
        ]
        real_groups[source] = float(roc_auc_score(*zip(*selected)))
    pair_values = []
    for generator in generators:
        for source in real_sources:
            selected = [
                (label, score)
                for row, label, score in zip(rows, labels, scores)
                if (label == 1 and str(row.get("generator", "unknown")) == generator)
                or (label == 0 and str(row.get("real_source", "unknown")) == source)
            ]
            value = float(roc_auc_score(*zip(*selected)))
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
def evaluate(model: torch.nn.Module, dataset: ManifestDataset) -> dict:
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    labels: list[int] = []
    scores: list[float] = []
    model.eval()
    for images, batch_labels, _ in loader:
        probabilities = torch.sigmoid(model(images.cuda(non_blocking=True)).flatten())
        labels.extend(int(value) for value in batch_labels.tolist())
        scores.extend(float(value) for value in probabilities.cpu().tolist())
    return {
        "count": len(labels),
        "clean_auc": float(roc_auc_score(labels, scores)),
        "groups": grouped_metrics(dataset.rows, labels, scores),
    }


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    packages = list(Path("/kaggle/input").rglob("family-mixture-v2.zip"))
    if len(packages) != 1:
        raise RuntimeError(f"expected one family-mixture-v2.zip, found {packages}")
    package = packages[0]
    observed_sha256 = file_sha256(package)
    if observed_sha256 != EXPECTED_ZIP_SHA256:
        raise RuntimeError(f"package checksum mismatch: {observed_sha256}")
    if not WORK_ROOT.exists():
        WORK_ROOT.mkdir(parents=True)
        with zipfile.ZipFile(package) as archive:
            archive.extractall(WORK_ROOT)
    package_metadata = json.loads((WORK_ROOT / "package.json").read_text())
    if package_metadata["inventory_sha256"] != EXPECTED_INVENTORY_SHA256:
        raise RuntimeError("package inventory mismatch")

    train_dataset = ManifestDataset(WORK_ROOT / "manifests/train.jsonl", TRAIN_TRANSFORM)
    known_dataset = ManifestDataset(
        WORK_ROOT / "manifests/eval_heldout_generators_known_reals.jsonl", EVAL_TRANSFORM
    )
    ffhq_dataset = ManifestDataset(
        WORK_ROOT / "manifests/eval_heldout_generators_ffhq_reals.jsonl", EVAL_TRANSFORM
    )
    if len(train_dataset) != 8160:
        raise RuntimeError(f"unexpected train rows: {len(train_dataset)}")

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1, img_size=IMAGE_SIZE).cuda()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.get_classifier().parameters():
        parameter.requires_grad = True
    parameters = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    if parameters["total"] >= 2_000_000_000:
        raise RuntimeError(f"parameter limit exceeded: {parameters}")

    loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    optimizer = torch.optim.AdamW(model.get_classifier().parameters(), lr=0.001, weight_decay=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    model.train()
    running_loss = 0.0
    seen = 0
    for step, (images, labels, _) in enumerate(loader, 1):
        images = images.cuda(non_blocking=True)
        labels = labels.to(device="cuda", dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images).flatten(), labels)
        loss.backward()
        optimizer.step()
        running_loss += float(loss) * len(labels)
        seen += len(labels)
        if step % 20 == 0 or step == len(loader):
            print(json.dumps({"step": step, "steps": len(loader), "loss": running_loss / seen}))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report = {
        "seed": SEED,
        "model": MODEL_NAME,
        "image_size": IMAGE_SIZE,
        "parameters": parameters,
        "package_sha256": observed_sha256,
        "inventory_sha256": package_metadata["inventory_sha256"],
        "train_rows": len(train_dataset),
        "final_loss": running_loss / seen,
        "known_real_source_gate": evaluate(model, known_dataset),
        "ffhq_real_source_gate": evaluate(model, ffhq_dataset),
        "elapsed_seconds": time.time() - started,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_name": MODEL_NAME,
        "image_size": IMAGE_SIZE,
        "normalization_mean": MEAN,
        "normalization_std": STD,
        "parameters": parameters,
        "seed": SEED,
    }
    torch.save(checkpoint, OUTPUT_ROOT / "model.pt")
    (OUTPUT_ROOT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
