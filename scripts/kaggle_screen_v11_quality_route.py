#!/usr/bin/env python3
"""Screen the frozen v11 quality route without opening its fresh gate."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import timm
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

import kaggle_evaluate_community_forensics as community
import kaggle_evaluate_v8_promotion_gates as v8_gates
import kaggle_train_v3 as runner
import kaggle_train_v8_frontier as v8
from quality_routed_multihead import (
    GENERAL_V6_WEIGHT,
    GENERAL_V9_WEIGHT,
    ROUTING_THRESHOLD,
    haar_noise_estimate,
    quality_routed_scores,
)


MODEL_NAME = "vit_pe_core_large_patch14_336"
IMAGE_SIZE = 224
PHYSICAL_BATCH_SIZE = 64
OUTPUT_ROOT = Path("/kaggle/working/track5-v11-quality-route-screen")
CHECKPOINTS = {
    "v6": {
        "bytes": 631_645_967,
        "sha256": "48ea50773fbd1b7247fff25fde6f985183e29f2eb517b5ac0f6319c1fe38b644",
    },
    "v9": {
        "bytes": 1_263_202_267,
        "sha256": "dd6b26c7849489447c7e96823f5b5e87c31623ca8ae0d28a5b162bb2dcb65075",
    },
    "v10": {
        "bytes": 1_263_202_267,
        "sha256": "633d6e0dada31dc1bf3e97e1cf1b534b7a54a21d3c0f7bd3aee9609e8c5f71f9",
    },
}
CONDITIONS = (("clean", None, 20260831), ("noise_sigma_0.10", 0.10, 20260843))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_checkpoint(name: str) -> Path:
    specification = CHECKPOINTS[name]
    candidates = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if root.exists():
            candidates.extend(
                path
                for path in root.rglob("*.pt")
                if path.is_file() and path.stat().st_size == specification["bytes"]
            )
    exact = [path for path in candidates if sha256_file(path) == specification["sha256"]]
    if len(exact) != 1:
        raise RuntimeError(f"expected one exact {name} checkpoint: {exact}")
    return exact[0]


class ConditionDataset(Dataset):
    def __init__(self, manifest: Path, rows: list[dict], mean, std, sigma, seed) -> None:
        self.manifest = manifest
        self.rows = rows
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]
        self.sigma = sigma
        self.seed = int(seed)
        self.base = v2.Compose(
            [
                v2.Resize(IMAGE_SIZE, antialias=True),
                v2.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = (self.manifest.parent / row["path"]).resolve()
        with Image.open(image_path) as image:
            pixels = self.base(image.convert("RGB"))
        if self.sigma is not None:
            generator = torch.Generator().manual_seed(self.seed + index)
            noise = torch.randn(pixels.shape, generator=generator, dtype=pixels.dtype)
            pixels = (pixels + noise * self.sigma).clamp(0.0, 1.0)
        estimate = haar_noise_estimate(pixels.unsqueeze(0))[0]
        return (pixels - self.mean) / self.std, estimate, int(row["label"]), index


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_shared_model(paths: dict[str, Path]):
    checkpoints = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    for name, checkpoint in checkpoints.items():
        if (
            checkpoint["model_name"] != MODEL_NAME
            or int(checkpoint["image_size"]) != IMAGE_SIZE
        ):
            raise RuntimeError(f"{name} architecture mismatch")
    means = {tuple(checkpoint["normalization_mean"]) for checkpoint in checkpoints.values()}
    stds = {tuple(checkpoint["normalization_std"]) for checkpoint in checkpoints.values()}
    if len(means) != 1 or len(stds) != 1:
        raise RuntimeError("checkpoint normalization mismatch")
    v9_state = checkpoints["v9"]["state_dict"]
    for name in ("v6", "v10"):
        other = checkpoints[name]["state_dict"]
        unequal = [
            key
            for key in v9_state
            if key not in {"head.weight", "head.bias"}
            and not torch.equal(other[key].half(), v9_state[key].half())
        ]
        if unequal:
            raise RuntimeError(f"{name}/v9 FP16 backbone mismatch: {unequal[:3]}")

    model = timm.create_model(
        MODEL_NAME, pretrained=False, num_classes=1, img_size=IMAGE_SIZE
    )
    model.load_state_dict(v9_state)
    model = model.half().cuda().eval()
    heads = {
        name: (
            checkpoint["state_dict"]["head.weight"].half().cuda(),
            checkpoint["state_dict"]["head.bias"].half().cuda(),
        )
        for name, checkpoint in checkpoints.items()
    }
    mean = next(iter(means))
    std = next(iter(stds))
    del checkpoints, v9_state
    gc.collect()
    return model, heads, mean, std


@torch.inference_mode()
def verify_shared_logits(model, heads, paths) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(20260831)
    image = torch.randn(
        (1, 3, IMAGE_SIZE, IMAGE_SIZE),
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        features = model.forward_head(model.forward_features(image), pre_logits=True)
        shared = {name: F.linear(features, *head) for name, head in heads.items()}
    report = {}
    for name, path in paths.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        separate = timm.create_model(
            MODEL_NAME, pretrained=False, num_classes=1, img_size=IMAGE_SIZE
        )
        separate.load_state_dict(checkpoint["state_dict"])
        separate = separate.half().cuda().eval()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            separate_logit = separate(image)
        difference = (separate_logit - shared[name]).abs()
        report[name] = {
            "exact": bool(torch.equal(separate_logit, shared[name])),
            "maximum_absolute_logit_difference": float(difference.max()),
        }
        del checkpoint, separate, separate_logit, difference
        torch.cuda.empty_cache()
        gc.collect()
    if not all(value["exact"] for value in report.values()):
        raise RuntimeError(f"shared-head contract mismatch: {report}")
    return report


@torch.inference_mode()
def score(model, heads, dataset: ConditionDataset) -> list[dict]:
    loader = DataLoader(
        dataset,
        batch_size=PHYSICAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    predictions = []
    for images, estimates, labels, indices in loader:
        logical = int(images.shape[0])
        if logical < PHYSICAL_BATCH_SIZE:
            pad = PHYSICAL_BATCH_SIZE - logical
            images = torch.cat([images, images[-1:].repeat(pad, 1, 1, 1)])
            estimates = torch.cat([estimates, estimates[-1:].repeat(pad)])
        images = images.half().cuda(non_blocking=True)
        estimates = estimates.cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            features = model.forward_head(model.forward_features(images), pre_logits=True)
            head_scores = {
                name: torch.sigmoid(F.linear(features, *head).flatten())
                for name, head in heads.items()
            }
            routed, general, route_mask = quality_routed_scores(
                estimates,
                head_scores["v6"],
                head_scores["v9"],
                head_scores["v10"],
            )
        for offset, (label, index) in enumerate(zip(labels.tolist(), indices.tolist())):
            row = dataset.rows[int(index)]
            predictions.append(
                {
                    "index": int(index),
                    "label": int(label),
                    "image_sha256": row.get("image_sha256"),
                    "generator": row.get("generator") or row.get("generator_model"),
                    "real_source": row.get("real_source"),
                    "noise_estimate": float(estimates[offset].float().cpu()),
                    "routed_to_v10": bool(route_mask[offset].cpu()),
                    "general_score": float(general[offset].float().cpu()),
                    "v10_score": float(head_scores["v10"][offset].float().cpu()),
                    "score": float(routed[offset].float().cpu()),
                }
            )
    return predictions


def metrics(rows: list[dict], score_key: str) -> dict:
    labels = [int(row["label"]) for row in rows]
    scores = [float(row[score_key]) for row in rows]
    real = [score for label, score in zip(labels, scores) if label == 0]
    fake = [score for label, score in zip(labels, scores) if label == 1]
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "real_mean": sum(real) / len(real),
        "fake_mean": sum(fake) / len(fake),
        "mean_score_inversion": (sum(real) / len(real)) > (sum(fake) / len(fake)),
    }


def screen_decision(reports: dict) -> dict:
    checks = {}
    for gate, conditions in reports.items():
        clean = conditions["clean"]
        noise = conditions["noise_sigma_0.10"]
        checks[f"{gate}_clean_drop_at_most_0_002"] = (
            clean["routed"]["auc"] >= clean["general"]["auc"] - 0.002
        )
        checks[f"{gate}_noise_non_regression"] = (
            noise["routed"]["auc"] >= noise["general"]["auc"]
        )
        checks[f"{gate}_clean_route_rate_at_most_0_05"] = clean["route_rate"] <= 0.05
        checks[f"{gate}_noise_route_rate_at_least_0_95"] = noise["route_rate"] >= 0.95
    return {"checks": checks, "passes": all(checks.values())}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("v11 screen requires CUDA")
    paths = {name: locate_checkpoint(name) for name in CHECKPOINTS}
    model, heads, mean, std = load_shared_model(paths)
    shared_check = verify_shared_logits(model, heads, paths)

    runner.EXCLUDED_EVAL_SHA256 = v8.EXCLUDED_EVAL_SHA256
    runner.validate_package()
    internal_manifest = runner.WORK_ROOT / "manifests/eval_selection.jsonl"
    internal_rows = runner.filter_evaluation_rows(
        read_rows(internal_manifest), runner.EXCLUDED_EVAL_SHA256
    )
    v8_root, _ = v8_gates.validate_package()
    ntire_spec = v8_gates.MANIFESTS["ntire_shard5_full_audit"]
    ntire_manifest = v8_root / ntire_spec["path"]
    ntire_rows = read_rows(ntire_manifest)
    community_manifest, community_rows, _, _ = community.validate_and_extract()
    gates = {
        "internal": (internal_manifest, internal_rows),
        "ntire": (ntire_manifest, ntire_rows),
        "community": (community_manifest, community_rows),
    }

    reports = {}
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for gate, (manifest, rows) in gates.items():
        reports[gate] = {}
        for condition, sigma, seed in CONDITIONS:
            torch.manual_seed(seed)
            dataset = ConditionDataset(manifest, rows, mean, std, sigma, seed)
            predictions = score(model, heads, dataset)
            output = OUTPUT_ROOT / f"{gate}-{condition}-predictions.jsonl"
            output.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions)
            )
            reports[gate][condition] = {
                "rows": len(predictions),
                "general": metrics(predictions, "general_score"),
                "v10": metrics(predictions, "v10_score"),
                "routed": metrics(predictions, "score"),
                "route_rate": sum(row["routed_to_v10"] for row in predictions)
                / len(predictions),
                "noise_estimate_minimum": min(row["noise_estimate"] for row in predictions),
                "noise_estimate_maximum": max(row["noise_estimate"] for row in predictions),
                "predictions_sha256": sha256_file(output),
            }
            print(json.dumps({"gate": gate, "condition": condition, **reports[gate][condition]}), flush=True)

    result = {
        "status": "screened_before_fresh_gate",
        "router_threshold": ROUTING_THRESHOLD,
        "general_weights": {"v6": GENERAL_V6_WEIGHT, "v9": GENERAL_V9_WEIGHT},
        "shared_logit_check": shared_check,
        "reports": reports,
        "screen": screen_decision(reports),
        "fresh_gate_opened": False,
        "organizer_demo_rows": 0,
        "checkpoint_paths": {name: str(path) for name, path in paths.items()},
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    report_path = OUTPUT_ROOT / "report.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

