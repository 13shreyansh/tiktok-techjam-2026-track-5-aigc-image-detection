#!/usr/bin/env python3
"""Evaluate the frozen v6/v10 blend on the untouched low-resolution gate.

The candidate hash and byte count are mandatory command-line inputs because
they do not exist until the already-frozen v10 training run finishes.  This
script performs inference only: it contains no parameter update, training, threshold
selection, calibration or blend search.  The organizer demo-only resources are
never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


PACKAGE_NAME = "cifar100-qwen-lowres-promotion-gate.zip"
PACKAGE_BYTES = 146_586_784
PACKAGE_SHA256 = "1dd7d460a8a9096d17499dd79cc74d7c4aa7bf0cb63a30435fd6d3e709b32cbf"
PACKAGE_INVENTORY_SHA256 = "b2357d4487faad00e41084329ecbd0c52a284fd61291de675039efcccf3efac2"
REAL_MANIFEST_SHA256 = "6020a655544065e08f0fd613df5410e125a8a56cf8d32e6f1b68ab263ebc41aa"
FAKE_MANIFEST_SHA256 = "3a34e0f8310bffce976bf2c49bb3ec7f5ac1ff8d35f5cc05229dbf1bae967ea2"
EXPECTED_REAL_ROWS = 1_000
EXPECTED_FAKE_ROWS = 144
EXPECTED_GENERATORS = 18
EXPECTED_PROMPTS = (85, 202, 214, 316, 489, 551, 728, 987)
MODEL_NAME = "vit_pe_core_large_patch14_336"
IMAGE_SIZE = 224
V6_BYTES = 631_645_967
V6_SHA256 = "48ea50773fbd1b7247fff25fde6f985183e29f2eb517b5ac0f6319c1fe38b644"
V6_WEIGHT = 0.75
V10_WEIGHT = 0.25
PHYSICAL_BATCH_SIZE = 64
OUTPUT_ROOT = Path("/kaggle/working/track5-v10-lowres-promotion")
EXTRACT_ROOT = OUTPUT_ROOT / "package"
CONDITIONS = (
    ("clean", None, 20260831),
    ("noise_sigma_0.10", 0.10, 20260843),
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


def locate_exact_file(name: str, expected_bytes: int, expected_sha256: str) -> Path:
    candidates = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if root.exists():
            candidates.extend(path for path in root.rglob(name) if path.is_file())
    exact = [
        path
        for path in candidates
        if path.stat().st_size == expected_bytes and sha256_file(path) == expected_sha256
    ]
    if not exact:
        raise RuntimeError(
            f"exact file absent: {name}, bytes={expected_bytes}, sha256={expected_sha256}"
        )
    return sorted(exact, key=str)[0]


def locate_gate() -> tuple[Path, bool]:
    try:
        return locate_exact_file(PACKAGE_NAME, PACKAGE_BYTES, PACKAGE_SHA256), True
    except RuntimeError:
        pass
    candidates = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if not root.exists():
            continue
        for path in root.rglob("package.json"):
            try:
                package = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if package.get("inventory_sha256") == PACKAGE_INVENTORY_SHA256:
                candidates.append(path.parent)
    if len(candidates) != 1:
        raise RuntimeError(f"expected one content-equivalent promotion package: {candidates}")
    return candidates[0], False


def validate_gate(source: Path, archive_reverified: bool) -> tuple[Path, list[dict], dict]:
    if archive_reverified:
        EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source) as archive:
            archive.extractall(EXTRACT_ROOT)
        root = EXTRACT_ROOT
    else:
        root = source
    package = json.loads((root / "package.json").read_text())
    if package.get("inventory_sha256") != PACKAGE_INVENTORY_SHA256:
        raise RuntimeError("promotion package inventory mismatch")
    records = {record["packaged_manifest"]: record for record in package["manifests"]}
    expected = {
        "manifests/manifest.jsonl": (EXPECTED_REAL_ROWS, REAL_MANIFEST_SHA256),
        "manifests/qwen_image_bench_holdout_v2-manifest.jsonl": (
            EXPECTED_FAKE_ROWS,
            FAKE_MANIFEST_SHA256,
        ),
    }
    if set(records) != set(expected):
        raise RuntimeError(f"unexpected promotion manifests: {sorted(records)}")
    rows = []
    for relative, (count, digest) in expected.items():
        manifest = root / relative
        if records[relative]["rows"] != count or records[relative]["sha256"] != digest:
            raise RuntimeError(f"packaged manifest metadata mismatch: {relative}")
        if sha256_file(manifest) != digest:
            raise RuntimeError(f"manifest content mismatch: {relative}")
        rows.extend(read_jsonl(manifest))
    labels = Counter(int(row["label"]) for row in rows)
    if labels != Counter({0: EXPECTED_REAL_ROWS, 1: EXPECTED_FAKE_ROWS}):
        raise RuntimeError(f"unexpected gate labels: {labels}")
    generators = Counter(str(row.get("generator")) for row in rows if int(row["label"]) == 1)
    prompts = Counter(int(row["prompt_id"]) for row in rows if int(row["label"]) == 1)
    if len(generators) != EXPECTED_GENERATORS or set(generators.values()) != {8}:
        raise RuntimeError(f"unexpected fake-generator composition: {generators}")
    if tuple(sorted(prompts)) != EXPECTED_PROMPTS or set(prompts.values()) != {18}:
        raise RuntimeError(f"unexpected prompt composition: {prompts}")
    hashes = []
    for row in rows:
        image = (root / "manifests" / row["path"]).resolve()
        if sha256_file(image) != row["image_sha256"]:
            raise RuntimeError(f"promotion image mismatch: {image}")
        hashes.append(row["image_sha256"])
        row["_absolute_path"] = str(image)
    if len(set(hashes)) != len(rows):
        raise RuntimeError("promotion gate contains duplicate image bytes")
    return root, rows, package


class PromotionDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        mean: tuple[float, ...],
        std: tuple[float, ...],
        noise_sigma: float | None,
    ) -> None:
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
        with Image.open(row["_absolute_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label"]), index


def load_model(path: Path, expected_sha256: str) -> tuple[torch.nn.Module, dict]:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"checkpoint changed before load: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["model_name"] != MODEL_NAME or int(checkpoint["image_size"]) != IMAGE_SIZE:
        raise RuntimeError("checkpoint architecture contract mismatch")
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1, img_size=IMAGE_SIZE)
    model.load_state_dict(checkpoint["state_dict"])
    return model.cuda().eval(), checkpoint


def auc(labels: list[int], scores: list[float]) -> float:
    return float(roc_auc_score(np.asarray(labels), np.asarray(scores)))


def condition_metrics(rows: list[dict]) -> dict:
    labels = [int(row["label"]) for row in rows]
    result = {}
    for candidate, score_key in (("v6", "v6_score"), ("blend", "score")):
        scores = [float(row[score_key]) for row in rows]
        real = np.asarray([score for score, label in zip(scores, labels) if label == 0])
        fake = np.asarray([score for score, label in zip(scores, labels) if label == 1])
        result[candidate] = {
            "auc": auc(labels, scores),
            "real_mean": float(real.mean()),
            "fake_mean": float(fake.mean()),
            "mean_score_inversion": bool(real.mean() > fake.mean()),
            "illustrative_real_fraction_at_or_above_0.5": float(np.mean(real >= 0.5)),
        }
    real_rows = [row for row in rows if int(row["label"]) == 0]
    fake_rows = [row for row in rows if int(row["label"]) == 1]
    result["blend"]["worst_generator_auc_against_all_reals"] = min(
        auc(
            [0] * len(real_rows) + [1] * len(group),
            [float(row["score"]) for row in real_rows + group],
        )
        for generator in sorted({row["generator"] for row in fake_rows})
        for group in [[row for row in fake_rows if row["generator"] == generator]]
    )
    result["blend"]["worst_prompt_auc_against_all_reals"] = min(
        auc(
            [0] * len(real_rows) + [1] * len(group),
            [float(row["score"]) for row in real_rows + group],
        )
        for prompt in EXPECTED_PROMPTS
        for group in [[row for row in fake_rows if int(row["prompt_id"]) == prompt]]
    )
    return result


@torch.inference_mode()
def score_condition(
    v6_model: torch.nn.Module,
    v10_model: torch.nn.Module,
    dataset: PromotionDataset,
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
    for images, labels, indices in loader:
        original_count = int(images.shape[0])
        tensor_digest.update(images.contiguous().numpy().tobytes())
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
            source = rows[int(index)]
            predictions.append(
                {
                    "index": int(index),
                    "label": int(label),
                    "image_sha256": source["image_sha256"],
                    "real_source": source.get("real_source"),
                    "generator": source.get("generator"),
                    "prompt_id": source.get("prompt_id"),
                    "fine_label": source.get("fine_label"),
                    "v6_score": float(v6_score),
                    "v10_score": float(v10_score),
                    "score": float(blend_score),
                }
            )
    if [row["index"] for row in predictions] != list(range(len(rows))):
        raise RuntimeError("promotion prediction order mismatch")
    torch.cuda.synchronize(0)
    return {
        "rows": len(predictions),
        "input_tensor_sha256": tensor_digest.hexdigest(),
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "physical_batches": math.ceil(len(rows) / PHYSICAL_BATCH_SIZE),
        "wall_seconds_including_decode_and_input_hash": time.perf_counter() - started,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "metrics": condition_metrics(predictions),
    }, predictions


def promotion_decision(conditions: dict, internal_screen: dict) -> dict:
    clean = conditions["clean"]["metrics"]
    noise = conditions["noise_sigma_0.10"]["metrics"]
    checks = {
        "fresh_clean_auc_drop_from_v6_at_most_0.01": (
            clean["blend"]["auc"] >= clean["v6"]["auc"] - 0.01
        ),
        "fresh_noise_auc_at_least_0.60": noise["blend"]["auc"] >= 0.60,
        "fresh_noise_auc_improvement_over_v6_at_least_0.05": (
            noise["blend"]["auc"] >= noise["v6"]["auc"] + 0.05
        ),
        "fresh_noise_authentic_mean_not_above_fake_mean": (
            not noise["blend"]["mean_score_inversion"]
        ),
        "selection_clean_auc_drop_from_v6_at_most_0.002": (
            internal_screen["blend"]["selection_clean_auc"]
            >= internal_screen["v6"]["selection_clean_auc"] - 0.002
        ),
        "selection_worst_pair_drop_from_v6_at_most_0.01": (
            internal_screen["blend"]["selection_worst_pair_auc"]
            >= internal_screen["v6"]["selection_worst_pair_auc"] - 0.01
        ),
        "content_clean_auc_drop_from_v6_at_most_0.002": (
            internal_screen["blend"]["content_clean_auc"]
            >= internal_screen["v6"]["content_clean_auc"] - 0.002
        ),
        "content_worst_pair_drop_from_v6_at_most_0.01": (
            internal_screen["blend"]["content_worst_pair_auc"]
            >= internal_screen["v6"]["content_worst_pair_auc"] - 0.01
        ),
    }
    return {
        "checks": checks,
        "passes_all_frozen_rules": all(checks.values()),
        "boundary": "Fail any check: reject v10 and do not tune on this consumed gate.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v6-checkpoint", type=Path, required=True)
    parser.add_argument("--v10-checkpoint", type=Path, required=True)
    parser.add_argument("--v10-bytes", type=int, required=True)
    parser.add_argument("--v10-sha256", required=True)
    parser.add_argument("--internal-screen", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("promotion requires the verified CUDA-FP16 path")
    if args.v6_checkpoint.stat().st_size != V6_BYTES:
        raise RuntimeError("v6 checkpoint byte count mismatch")
    if sha256_file(args.v6_checkpoint) != V6_SHA256:
        raise RuntimeError("v6 checkpoint SHA-256 mismatch")
    if args.v10_checkpoint.stat().st_size != args.v10_bytes:
        raise RuntimeError("v10 checkpoint byte count mismatch")
    if sha256_file(args.v10_checkpoint) != args.v10_sha256:
        raise RuntimeError("v10 checkpoint SHA-256 mismatch")
    internal_screen = json.loads(args.internal_screen.read_text())
    gate_source, archive_reverified = locate_gate()
    _, rows, package = validate_gate(gate_source, archive_reverified)
    v6_model, v6_checkpoint = load_model(args.v6_checkpoint, V6_SHA256)
    v10_model, v10_checkpoint = load_model(args.v10_checkpoint, args.v10_sha256)
    mean = tuple(v6_checkpoint["normalization_mean"])
    std = tuple(v6_checkpoint["normalization_std"])
    if mean != tuple(v10_checkpoint["normalization_mean"]) or std != tuple(
        v10_checkpoint["normalization_std"]
    ):
        raise RuntimeError("v6/v10 normalization mismatch")
    reports = {}
    outputs = {}
    for name, noise_sigma, seed in CONDITIONS:
        torch.manual_seed(seed)
        dataset = PromotionDataset(rows, mean, std, noise_sigma)
        condition, predictions = score_condition(v6_model, v10_model, dataset, rows)
        prediction_path = OUTPUT_ROOT / f"{name}_predictions.jsonl"
        write_jsonl(prediction_path, predictions)
        reports[name] = {**condition, "seed": seed, "noise_sigma": noise_sigma}
        outputs[name] = {
            "path": str(prediction_path),
            "bytes": prediction_path.stat().st_size,
            "sha256": sha256_file(prediction_path),
        }
    report = {
        "completed": True,
        "purpose": "One-shot evaluation of the predeclared fixed v6/v10 blend; no learning or result-driven selection.",
        "package": {
            "archive_reverified_in_runtime": archive_reverified,
            "archive_sha256": PACKAGE_SHA256,
            "inventory_sha256": package["inventory_sha256"],
            "rows": len(rows),
        },
        "models": {
            "v6_checkpoint_sha256": V6_SHA256,
            "v10_checkpoint_sha256": args.v10_sha256,
            "weights": {"v6": V6_WEIGHT, "v10": V10_WEIGHT},
            "arithmetic": "one GPU; physical batch 64; sigmoid and 75/25 blend in FP16; FP32 after blend",
        },
        "conditions": reports,
        "internal_screen": internal_screen,
        "promotion": promotion_decision(reports, internal_screen),
        "outputs": outputs,
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
