#!/usr/bin/env python3
"""Reproduce the training-only transform identification audit for v11."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision.transforms import v2

from quality_routed_multihead import ROUTING_THRESHOLD, haar_noise_estimate


SEED = 20260831
GROUP_CAP = 80
IMAGE_SIZE = 224
SIGMAS = (0.0, 0.02, 0.05, 0.10)
DEFAULT_MANIFESTS = (
    Path("datasets/family_mixture_v6/train.jsonl"),
    Path("datasets/cifake_lowres_repair_v10/manifest.jsonl"),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/audits/quality-router-training-only.json"),
    )
    args = parser.parse_args()

    groups: dict[tuple[int, str], list[tuple[Path, dict]]] = defaultdict(list)
    for manifest in DEFAULT_MANIFESTS:
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            image = (manifest.parent / row["path"]).resolve()
            if not image.is_file():
                continue
            source = str(row.get("real_source") or row.get("generator") or "unknown")
            groups[(int(row["label"]), source)].append((image, row))

    rng = random.Random(SEED)
    selected = []
    for group in sorted(groups):
        candidates = list(groups[group])
        rng.shuffle(candidates)
        selected.extend(candidates[: min(GROUP_CAP, len(candidates))])
    rng.shuffle(selected)

    transform = v2.Compose(
        [
            v2.Resize(IMAGE_SIZE, antialias=True),
            v2.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    values = {sigma: [] for sigma in SIGMAS}
    generator = torch.Generator().manual_seed(SEED)
    selected_hashes = hashlib.sha256()
    for image_path, row in selected:
        selected_hashes.update(str(row.get("image_sha256") or image_path).encode())
        selected_hashes.update(b"\n")
        with Image.open(image_path) as image:
            clean = transform(image.convert("RGB"))
        for sigma in SIGMAS:
            if sigma:
                noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
                pixels = (clean + noise * sigma).clamp(0.0, 1.0)
            else:
                pixels = clean
            values[sigma].append(float(haar_noise_estimate(pixels.unsqueeze(0))[0]))

    clean = np.asarray(values[0.0])
    sigma_10 = np.asarray(values[0.10])
    report = {
        "purpose": "training-only label-blind transform identification; no candidate selection",
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "group_cap": GROUP_CAP,
        "rows": len(selected),
        "source_groups": len(groups),
        "selected_row_inventory_sha256": selected_hashes.hexdigest(),
        "manifests": {
            str(path): file_sha256(path) for path in DEFAULT_MANIFESTS
        },
        "estimator": "median(abs(diagonal 2x2 Haar coefficient))/0.67448975",
        "routing_threshold": ROUTING_THRESHOLD,
        "conditions": {
            f"sigma_{sigma:.2f}": percentile_summary(values[sigma])
            for sigma in SIGMAS
        },
        "clean_vs_sigma_0_10_auc": float(
            roc_auc_score(
                np.r_[np.zeros(len(clean)), np.ones(len(sigma_10))],
                np.r_[clean, sigma_10],
            )
        ),
        "routing_rates": {
            f"sigma_{sigma:.2f}": float(
                (np.asarray(values[sigma]) >= ROUTING_THRESHOLD).mean()
            )
            for sigma in SIGMAS
        },
        "clean_sigma_0_10_separation": bool(clean.max() < sigma_10.min()),
        "uses_ai_labels_to_route": False,
        "uses_model_scores_to_route": False,
        "uses_consumed_gate_to_set_threshold": False,
        "organizer_demo_rows": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

