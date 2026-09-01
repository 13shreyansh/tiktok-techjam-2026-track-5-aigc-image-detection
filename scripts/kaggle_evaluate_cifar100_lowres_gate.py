#!/usr/bin/env python3
"""Score the frozen CIFAR-100 authentic gate with exact v6 and v6/v9 blend.

This is a diagnosis-only evaluator.  It creates no training rows and computes
no threshold, calibration parameter, blend weight or promotion decision.  The
two conditions exactly mirror the existing clean and Gaussian sigma-0.10
workshop paths.  A fixed physical batch of 64, including padding of the final
logical batch, preserves the verified one-GPU CUDA-FP16 blend contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


PACKAGE_NAME = "cifar100-lowres-gate.zip"
PACKAGE_BYTES = 2_772_682
PACKAGE_SHA256 = "f6520c1b36e81d04ef60ece5386927a5acc4dba0d5bdd577d43ab24b4dfde67b"
PACKAGE_INVENTORY_SHA256 = "225d798db8160de0e6dd5d4ebdbde7463a0519db34c6291652b24b356d9b0a70"
MANIFEST_SHA256 = "7d6cb7214641e75fb3ffdf91f15a81acdeebe8d4df04a3b03245eab9c35b0386"
EXPECTED_ROWS = 1000
MODEL_NAME = "vit_pe_core_large_patch14_336"
# Both exact checkpoints record ``image_size=224`` and were trained/evaluated
# through the established runner contract at that resolution.  The timm model
# name contains ``_336`` as an upstream architecture label, but reconstructing
# it at 336 creates a 577-token positional grid and cannot load the exact
# 257-token (224/14 squared plus class token) checkpoint state.
IMAGE_SIZE = 224
V6_BYTES = 631_645_967
V6_SHA256 = "48ea50773fbd1b7247fff25fde6f985183e29f2eb517b5ac0f6319c1fe38b644"
V9_BYTES = 1_263_202_267
V9_SHA256 = "dd6b26c7849489447c7e96823f5b5e87c31623ca8ae0d28a5b162bb2dcb65075"
V6_WEIGHT = 0.75
V9_WEIGHT = 0.25
PHYSICAL_BATCH_SIZE = 64
OUTPUT_ROOT = Path("/kaggle/working/track5-cifar100-lowres-gate-eval")
EXTRACT_ROOT = OUTPUT_ROOT / "package"
CONDITIONS = (
    ("clean", None, 20260830),
    ("noise_sigma_0.10", 0.10, 20260842),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2) + "\n")
    partial.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    partial.replace(path)


def locate_gate_source() -> tuple[Path, bool]:
    """Find either the exact ZIP or Kaggle's content-equivalent extraction."""
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if not root.exists():
            continue
        for path in root.rglob(PACKAGE_NAME):
            if (
                path.is_file()
                and path.stat().st_size == PACKAGE_BYTES
                and sha256_file(path) == PACKAGE_SHA256
            ):
                return path, True
    extracted = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if not root.exists():
            continue
        for path in root.rglob("package.json"):
            try:
                package = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            manifests = package.get("manifests", [])
            if (
                package.get("inventory_sha256") == PACKAGE_INVENTORY_SHA256
                and len(manifests) == 1
                and manifests[0].get("sha256") == MANIFEST_SHA256
            ):
                extracted.append(path)
    if len(extracted) != 1:
        raise RuntimeError(
            "expected one exact ZIP or one content-equivalent Kaggle extraction; "
            f"found extracted candidates: {extracted}"
        )
    return extracted[0], False


def locate_checkpoint(expected_bytes: int, expected_sha256: str) -> Path:
    candidates = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if not root.exists():
            continue
        for path in root.rglob("*.pt"):
            if path.is_file() and path.stat().st_size == expected_bytes:
                candidates.append(path)
    exact = [path for path in candidates if sha256_file(path) == expected_sha256]
    if not exact:
        raise RuntimeError(
            f"checkpoint absent: expected {expected_bytes} bytes and {expected_sha256}"
        )
    return sorted(exact, key=str)[0]


def validate_and_extract(
    source_path: Path, archive_reverified: bool
) -> tuple[Path, list[dict], dict]:
    if archive_reverified:
        if source_path.stat().st_size != PACKAGE_BYTES or sha256_file(source_path) != PACKAGE_SHA256:
            raise RuntimeError("CIFAR-100 gate package bytes/hash mismatch")
        with zipfile.ZipFile(source_path) as archive:
            package = json.loads(archive.read("package.json"))
            if not EXTRACT_ROOT.exists():
                EXTRACT_ROOT.mkdir(parents=True)
                archive.extractall(EXTRACT_ROOT)
        package_root = EXTRACT_ROOT
    else:
        package = json.loads(source_path.read_text())
        package_root = source_path.parent
    if package["inventory_sha256"] != PACKAGE_INVENTORY_SHA256:
        raise RuntimeError("CIFAR-100 package inventory mismatch")
    manifest_record = package["manifests"]
    if len(manifest_record) != 1 or manifest_record[0]["sha256"] != MANIFEST_SHA256:
        raise RuntimeError("CIFAR-100 packaged manifest metadata mismatch")
    manifest_path = package_root / "manifests/manifest.jsonl"
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("extracted CIFAR-100 manifest hash mismatch")
    rows = read_jsonl(manifest_path)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} gate rows, found {len(rows)}")
    if any(int(row["label"]) != 0 or row.get("real_source") != "CIFAR100-test" for row in rows):
        raise RuntimeError("gate contains a non-authentic or wrong-source row")
    observed_hashes = set()
    for row in rows:
        path = (manifest_path.parent / row["path"]).resolve()
        observed = sha256_file(path)
        if observed != row["image_sha256"]:
            raise RuntimeError(f"image hash mismatch: {path}")
        observed_hashes.add(observed)
    if len(observed_hashes) != EXPECTED_ROWS:
        raise RuntimeError("gate image hashes are not unique")
    return manifest_path, rows, package


class GateDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        rows: list[dict],
        mean: tuple[float, ...],
        std: tuple[float, ...],
        noise_sigma: float | None,
    ) -> None:
        self.manifest_path = manifest_path
        self.rows = rows
        operations = [
            v2.Resize(IMAGE_SIZE, antialias=True),
            v2.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
        if noise_sigma is not None:
            operations.append(
                v2.Lambda(
                    lambda tensor: (
                        tensor + torch.randn_like(tensor) * noise_sigma
                    ).clamp(0.0, 1.0)
                )
            )
        operations.append(v2.Normalize(mean, std))
        self.transform = v2.Compose(operations)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = self.manifest_path.parent / row["path"]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label"]), index


def load_model(path: Path, expected_sha256: str) -> tuple[torch.nn.Module, dict]:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"checkpoint changed before load: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["model_name"] != MODEL_NAME:
        raise RuntimeError(f"checkpoint model mismatch: {checkpoint['model_name']}")
    if int(checkpoint["image_size"]) != IMAGE_SIZE:
        raise RuntimeError(
            f"checkpoint image-size mismatch: {checkpoint['image_size']} != {IMAGE_SIZE}"
        )
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1, img_size=IMAGE_SIZE)
    model.load_state_dict(checkpoint["state_dict"])
    return model.cuda().eval(), checkpoint


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "minimum": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
        "fraction_at_or_above_0.5": float(np.mean(array >= 0.5)),
    }


@torch.inference_mode()
def score_condition(
    v6_model: torch.nn.Module,
    v9_model: torch.nn.Module,
    dataset: GateDataset,
    rows: list[dict],
) -> tuple[dict, list[dict]]:
    loader = DataLoader(
        dataset,
        batch_size=PHYSICAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    tensor_digest = hashlib.sha256()
    predictions = []
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize(0)
    started = time.perf_counter()
    forward_seconds = 0.0
    for images, labels, indices in loader:
        original_count = int(images.shape[0])
        tensor_digest.update(images.contiguous().numpy().tobytes())
        if original_count < PHYSICAL_BATCH_SIZE:
            images = torch.cat(
                [images, images[-1:].repeat(PHYSICAL_BATCH_SIZE - original_count, 1, 1, 1)]
            )
        if int(images.shape[0]) != PHYSICAL_BATCH_SIZE:
            raise RuntimeError(f"physical batch mismatch: {tuple(images.shape)}")
        images = images.cuda(non_blocking=True)
        torch.cuda.synchronize(0)
        forward_started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            v6_scores = torch.sigmoid(v6_model(images).flatten())
            v9_scores = torch.sigmoid(v9_model(images).flatten())
            blend_scores = V6_WEIGHT * v6_scores + V9_WEIGHT * v9_scores
        v6_cpu = v6_scores[:original_count].float().cpu().tolist()
        blend_cpu = blend_scores[:original_count].float().cpu().tolist()
        torch.cuda.synchronize(0)
        forward_seconds += time.perf_counter() - forward_started
        for label, index, v6_score, blend_score in zip(
            labels.tolist(), indices.tolist(), v6_cpu, blend_cpu
        ):
            source = rows[int(index)]
            predictions.append(
                {
                    "index": int(index),
                    "label": int(label),
                    "image_sha256": source["image_sha256"],
                    "fine_label": int(source["fine_label"]),
                    "coarse_label": int(source["coarse_label"]),
                    "real_source": source["real_source"],
                    "v6_score": float(v6_score),
                    "score": float(blend_score),
                }
            )
    if [row["index"] for row in predictions] != list(range(len(rows))):
        raise RuntimeError("prediction order mismatch")
    v6_values = [row["v6_score"] for row in predictions]
    blend_values = [row["score"] for row in predictions]
    elapsed = time.perf_counter() - started
    return {
        "rows": len(predictions),
        "input_tensor_sha256": tensor_digest.hexdigest(),
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "physical_batches": math.ceil(len(rows) / PHYSICAL_BATCH_SIZE),
        "wall_seconds_including_decode_and_input_hash": elapsed,
        "paired_model_forward_seconds": forward_seconds,
        "images_per_forward_second": len(rows) / forward_seconds,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "v6_authentic_scores": summarize(v6_values),
        "blend_authentic_scores": summarize(blend_values),
    }, predictions


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CIFAR-100 gate requires the verified CUDA path")
    package_source, archive_reverified = locate_gate_source()
    manifest_path, rows, package = validate_and_extract(
        package_source, archive_reverified
    )
    v6_path = locate_checkpoint(V6_BYTES, V6_SHA256)
    v9_path = locate_checkpoint(V9_BYTES, V9_SHA256)
    load_started = time.perf_counter()
    v6_model, v6_checkpoint = load_model(v6_path, V6_SHA256)
    v9_model, v9_checkpoint = load_model(v9_path, V9_SHA256)
    torch.cuda.synchronize(0)
    load_seconds = time.perf_counter() - load_started
    mean = tuple(v6_checkpoint["normalization_mean"])
    std = tuple(v6_checkpoint["normalization_std"])
    if mean != tuple(v9_checkpoint["normalization_mean"]) or std != tuple(
        v9_checkpoint["normalization_std"]
    ):
        raise RuntimeError("v6/v9 normalization mismatch")

    condition_reports = {}
    output_files = {}
    for name, noise_sigma, seed in CONDITIONS:
        torch.manual_seed(seed)
        dataset = GateDataset(manifest_path, rows, mean, std, noise_sigma)
        condition_report, predictions = score_condition(v6_model, v9_model, dataset, rows)
        prediction_path = OUTPUT_ROOT / f"{name}_predictions.jsonl"
        write_jsonl(prediction_path, predictions)
        condition_reports[name] = {**condition_report, "seed": seed, "noise_sigma": noise_sigma}
        output_files[name] = {
            "path": str(prediction_path),
            "bytes": prediction_path.stat().st_size,
            "sha256": sha256_file(prediction_path),
        }

    report = {
        "completed": True,
        "purpose": "New authentic-side diagnosis only; no training, tuning, calibration, threshold selection, blend selection or promotion.",
        "package": {
            "expected_archive_bytes": PACKAGE_BYTES,
            "expected_archive_sha256": PACKAGE_SHA256,
            "archive_reverified_in_runtime": archive_reverified,
            "inventory_sha256": package["inventory_sha256"],
            "manifest_sha256": MANIFEST_SHA256,
            "rows": len(rows),
        },
        "models": {
            "v6_checkpoint_sha256": sha256_file(v6_path),
            "v9_checkpoint_sha256": sha256_file(v9_path),
            "blend": "75/25 in CUDA FP16 before FP32 conversion",
            "load_seconds": load_seconds,
        },
        "conditions": condition_reports,
        "outputs": output_files,
        "command": [sys.executable, *sys.argv],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "forbidden_demo_resources_used": False,
    }
    report_path = OUTPUT_ROOT / "report.json"
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
