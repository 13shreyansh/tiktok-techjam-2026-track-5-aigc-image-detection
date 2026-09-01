#!/usr/bin/env python3
"""Evaluate the exact v6/v9 blend on a second unseen-prompt Qwen gate.

The 16 prompt IDs were acquired and frozen before the first score was read.
They are disjoint from both the first Qwen audit and v9 training prompts.
Images are downloaded from the pinned public benchmark revision.  The real
half is reused from the checksum-verified sealed package, then both labels are
processed by the same JPEG-q96 full-frame stretch rule.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import statistics
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import kaggle_evaluate_frontier_ensemble_promotion as promotion
import kaggle_evaluate_v8_promotion_gates as sealed
import kaggle_shape_stress_v6 as shape
import kaggle_train_v3 as runner


REPOSITORY = "Qwen/Qwen-Image-Bench"
REVISION = "d2493deb153b020cf169c7e3f57d15e4dd697038"
API_URL = f"https://huggingface.co/api/datasets/{REPOSITORY}?blobs=true"
PROMPT_IDS = (22, 183, 309, 327, 338, 420, 488, 536, 568, 631, 721, 772, 838, 859, 896, 900)
MODELS = (
    "FLUX.2-pro", "FLUX.2_max", "GLM-Image", "GPT-Image-1", "GPT-Image-1.5",
    "HunyuanImage-3.0", "Imagen-4.0", "Imagen-4.0-Ultra", "Qwen-Image",
    "Qwen-Image-2.0-pro", "Qwen-Image-2512", "Seedream-4.0", "Seedream-4.5",
    "Seedream-5.0", "gpt-image-2", "kling_v2_1", "nano-banana-2.0",
    "nano-banana-pro",
)
EXPECTED_REAL_HASH_INVENTORY = "985d0842c9f38a4771cb247cf48753edf6b9564f9d41eb2b1fdc7bf0af85e0c7"
EXPECTED_FAKE_HASH_INVENTORY = "c7f565e333aa09954243d0c41e90fb447c02045a8d57df513ff711b1e7c1caaa"
EXPECTED_ROWS = 576
PHYSICAL_BATCH_SIZE = 64
OUTPUT_ROOT = Path("/kaggle/working/qwen-unseen-prompt-holdout")
OUTPUT = Path("/kaggle/working/track5-qwen-unseen-prompt-ensemble.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_inventory(rows: list[dict]) -> str:
    values = "\n".join(sorted(str(row["image_sha256"]) for row in rows))
    return hashlib.sha256(values.encode()).hexdigest()


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "track5-audit/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def download(entry: dict) -> dict:
    destination = OUTPUT_ROOT / entry["source_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or destination.stat().st_size != entry["bytes"]:
        encoded = urllib.parse.quote(entry["source_path"], safe="/")
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{encoded}?download=true"
        temporary = destination.with_name(destination.name + ".partial")
        request = urllib.request.Request(url, headers={"User-Agent": "track5-audit/1"})
        with urllib.request.urlopen(request, timeout=240) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if temporary.stat().st_size != entry["bytes"]:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"download size mismatch: {entry['source_path']}")
        temporary.replace(destination)
    return {
        "path": str(destination.relative_to(OUTPUT_ROOT)),
        "label": 1,
        "generator": entry["generator"],
        "generator_model": entry["generator"],
        "family": "frontier-2026-image-generation",
        "prompt_id": entry["prompt_id"],
        "image_sha256": sha256(destination),
        "source_path": entry["source_path"],
        "source_blob_id": entry["blob_id"],
    }


def acquire_fake_rows() -> list[dict]:
    metadata = fetch_json(API_URL)
    if metadata.get("sha") != REVISION or metadata.get("private") or metadata.get("gated"):
        raise RuntimeError("Qwen source identity/access changed")
    if metadata.get("cardData", {}).get("license") != "apache-2.0":
        raise RuntimeError("Qwen source licence declaration changed")
    by_key = {}
    for sibling in metadata.get("siblings", []):
        source_path = str(sibling.get("rfilename", ""))
        parts = source_path.split("/")
        if len(parts) != 3 or parts[0] != "images" or parts[1] not in MODELS:
            continue
        prompt_id = int(parts[2].split("_", 1)[0])
        if prompt_id not in PROMPT_IDS:
            continue
        key = (parts[1], prompt_id)
        if key in by_key:
            raise RuntimeError(f"duplicate source cell: {key}")
        by_key[key] = {
            "source_path": source_path,
            "generator": parts[1],
            "prompt_id": prompt_id,
            "bytes": int(sibling.get("size") or sibling.get("lfs", {}).get("size")),
            "blob_id": sibling.get("blobId") or sibling.get("lfs", {}).get("oid"),
        }
    entries = [by_key[(model, prompt)] for model in MODELS for prompt in PROMPT_IDS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        rows = list(executor.map(download, entries))
    rows.sort(key=lambda row: (row["generator_model"], row["prompt_id"]))
    if hash_inventory(rows) != EXPECTED_FAKE_HASH_INVENTORY:
        raise RuntimeError("unseen fake-image SHA-256 inventory mismatch")
    return rows


def balanced_effects(rows: list[dict], score_key: str, real_scores: list[float]) -> dict:
    by_prompt = defaultdict(list)
    by_generator = defaultdict(list)
    for row in rows:
        by_prompt[int(row["prompt_id"])].append(float(row[score_key]))
        by_generator[str(row["generator_model"])].append(float(row[score_key]))
    if {len(v) for v in by_prompt.values()} != {len(MODELS)}:
        raise RuntimeError("prompt grid is unbalanced")
    if {len(v) for v in by_generator.values()} != {len(PROMPT_IDS)}:
        raise RuntimeError("generator grid is unbalanced")
    scores = [float(row[score_key]) for row in rows]
    grand = statistics.fmean(scores)
    total_ss = sum((value - grand) ** 2 for value in scores)
    prompt_ss = len(MODELS) * sum(
        (statistics.fmean(values) - grand) ** 2 for values in by_prompt.values()
    )
    generator_ss = len(PROMPT_IDS) * sum(
        (statistics.fmean(values) - grand) ** 2 for values in by_generator.values()
    )

    def auc(values: list[float]) -> float:
        labels = [0] * len(real_scores) + [1] * len(values)
        return float(roc_auc_score(labels, real_scores + values))

    return {
        "score_mean": grand,
        "sum_of_squares_share": {
            "prompt": prompt_ss / total_ss,
            "generator": generator_ss / total_ss,
            "interaction": max(0.0, total_ss - prompt_ss - generator_ss) / total_ss,
        },
        "prompts": sorted(
            [
                {"prompt_id": key, "mean_score": statistics.fmean(values), "auc_against_all_reals": auc(values)}
                for key, values in by_prompt.items()
            ],
            key=lambda item: item["auc_against_all_reals"],
        ),
    }


@torch.inference_mode()
def score(v6_model, v9_model, dataset) -> list[dict]:
    loader = DataLoader(dataset, batch_size=PHYSICAL_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    predictions = []
    for images, labels, indices in loader:
        if int(images.shape[0]) != PHYSICAL_BATCH_SIZE:
            raise RuntimeError(f"physical batch contract changed: {tuple(images.shape)}")
        images = images.to("cuda:0", non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            v6_scores = torch.sigmoid(v6_model(images).flatten())
            v9_scores = torch.sigmoid(v9_model(images).flatten())
            blend_scores = promotion.V6_WEIGHT * v6_scores + promotion.V9_WEIGHT * v9_scores
        for index, label, v6_value, blend_value in zip(
            indices.tolist(), labels.tolist(), v6_scores.float().cpu().tolist(), blend_scores.float().cpu().tolist()
        ):
            source = dataset.rows[int(index)]
            predictions.append({
                **source,
                "index": int(index),
                "label": int(label),
                "v6_score": float(v6_value),
                "score": float(blend_value),
            })
    return predictions


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen ensemble contract")
    sealed_root, sealed_package = sealed.validate_package()
    old_manifest = sealed_root / sealed.MANIFESTS["qwen_prompt_holdout"]["path"]
    old_rows = promotion.read_jsonl(old_manifest)
    real_rows = [dict(row) for row in old_rows if int(row["label"]) == 0]
    if len(real_rows) != 288 or hash_inventory(real_rows) != EXPECTED_REAL_HASH_INVENTORY:
        raise RuntimeError("sealed real-image inventory mismatch")

    started = time.time()
    fake_rows = acquire_fake_rows()
    manifest = OUTPUT_ROOT / "combined_gate.jsonl"
    combined = []
    for row in real_rows:
        source = (old_manifest.parent / row["path"]).resolve()
        combined.append({**row, "path": str(source)})
    combined.extend(fake_rows)
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in combined))
    if len(combined) != EXPECTED_ROWS:
        raise RuntimeError("combined row count mismatch")

    v6_path = promotion.selected_v6_path()
    v9_path = promotion.V9_ROOT / promotion.MODEL_NAME / "model.pt"
    v6_model, v6_checkpoint = promotion.load_model(v6_path, promotion.V6_SHA256, "cuda:0")
    v9_model, v9_checkpoint = promotion.load_model(v9_path, promotion.V9_SHA256, "cuda:0")
    mean = tuple(v6_checkpoint["normalization_mean"])
    std = tuple(v6_checkpoint["normalization_std"])
    if mean != tuple(v9_checkpoint["normalization_mean"]) or std != tuple(v9_checkpoint["normalization_std"]):
        raise RuntimeError("checkpoint normalization mismatch")
    dataset = shape.GeometryDataset(manifest, combined, mean, std, "jpeg_q96_stretch_full_frame")
    torch.cuda.reset_peak_memory_stats(0)
    predictions = score(v6_model, v9_model, dataset)
    labels = [int(row["label"]) for row in predictions]
    fake_predictions = [row for row in predictions if int(row["label"]) == 1]
    real_predictions = [row for row in predictions if int(row["label"]) == 0]

    candidates = {}
    for name, key in (("v6", "v6_score"), ("blend", "score")):
        values = [float(row[key]) for row in predictions]
        real_scores = [float(row[key]) for row in real_predictions]
        groups = runner.grouped_metrics(combined, labels, values)
        candidates[name] = {
            "auc": float(roc_auc_score(labels, values)),
            "worst_generator_auc": groups["worst_fake_generator_auc"],
            "worst_real_source_auc": groups["worst_real_source_auc"],
            "worst_pair_auc": groups["worst_generator_real_source_pair_auc"],
            "dependence": balanced_effects(fake_predictions, key, real_scores),
        }
    delta = {
        "auc": candidates["blend"]["auc"] - candidates["v6"]["auc"],
        "worst_generator_auc": candidates["blend"]["worst_generator_auc"] - candidates["v6"]["worst_generator_auc"],
        "worst_pair_auc": candidates["blend"]["worst_pair_auc"] - candidates["v6"]["worst_pair_auc"],
        "worst_prompt_auc": candidates["blend"]["dependence"]["prompts"][0]["auc_against_all_reals"] - candidates["v6"]["dependence"]["prompts"][0]["auc_against_all_reals"],
    }
    report = {
        "completed": True,
        "scope": "second 16-prompt by 18-generator clean Qwen holdout, unseen before v6 baseline",
        "source_revision": REVISION,
        "prompt_ids": list(PROMPT_IDS),
        "rows": len(predictions),
        "real_hash_inventory": EXPECTED_REAL_HASH_INVENTORY,
        "fake_hash_inventory": EXPECTED_FAKE_HASH_INVENTORY,
        "sealed_package_inventory_sha256": sealed_package["inventory_sha256"],
        "v6_checkpoint_sha256": promotion.V6_SHA256,
        "v9_checkpoint_sha256": promotion.V9_SHA256,
        "arithmetic_contract": "one GPU; physical batch 64; sigmoid and 75/25 blend in FP16; FP32 conversion only after blend",
        "preprocessing": "both labels JPEG q96 then full-frame stretch",
        "candidates": candidates,
        "delta_blend_minus_v6": delta,
        "elapsed_seconds_including_download_load_decode_and_inference": time.time() - started,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
