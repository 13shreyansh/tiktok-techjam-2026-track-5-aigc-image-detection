#!/usr/bin/env python3
"""Evaluate the one preselected v6/v9 blend on all frozen promotion gates.

The script deliberately runs v6 and the 75/25 blend on the same decoded and
transformed tensors.  V9 standalone metrics are not exposed on the sealed
holdout.  Conditions are the workshop's individual transformations, never
chains.  Every condition is persisted so an interrupted Kaggle session can
resume without silently dropping evidence.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import timm
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import kaggle_evaluate_community_forensics as external
import kaggle_evaluate_v8_promotion_gates as sealed
import kaggle_stress_eval_v6 as stress
import kaggle_train_v3 as runner
import kaggle_train_v6  # noqa: F401  # checksum-pinned v6 package constants


MODEL_NAME = "vit_pe_core_large_patch14_336"
V6_SHA256 = "48ea50773fbd1b7247fff25fde6f985183e29f2eb517b5ac0f6319c1fe38b644"
V9_SHA256 = "dd6b26c7849489447c7e96823f5b5e87c31623ca8ae0d28a5b162bb2dcb65075"
V6_WEIGHT = 0.75
V9_WEIGHT = 0.25
INFERENCE_POLICY = {
    "physical_batch_size": 128,
    "model_devices": ["cuda:0", "cuda:1"],
    "autocast_dtype": "float16",
    "score_conversion": "per_model_float32_cpu",
    "blend_location_dtype": "cpu_float32",
}
V9_ROOT = Path("/kaggle/working/track5-v9-frontier-capped-candidate")
OUTPUT_ROOT = Path("/kaggle/working/track5-frontier-ensemble-promotion")
SEED = 20260830
NON_FRONTIER_TOLERANCES = {
    "clean_auc": -0.002,
    "pooled_robust_auc": -0.002,
    "official_score": -0.002,
    "noise_sigma_0.10_auc": -0.01,
}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    temporary.replace(path)


def selected_v6_path() -> Path:
    matches = [
        path
        for path in Path("/kaggle/input").rglob("model.pt")
        if "track5-v6-selected-fp16-checkpoint" in str(path)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one selected v6 checkpoint: {matches}")
    if runner.file_sha256(matches[0]) != V6_SHA256:
        raise RuntimeError("selected v6 checkpoint SHA-256 mismatch")
    return matches[0]


def load_model(
    path: Path, expected_sha256: str, device: str
) -> tuple[torch.nn.Module, dict]:
    if runner.file_sha256(path) != expected_sha256:
        raise RuntimeError(f"checkpoint SHA-256 mismatch: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["model_name"] != MODEL_NAME:
        raise RuntimeError(f"checkpoint model mismatch: {path}")
    model = timm.create_model(
        MODEL_NAME, pretrained=False, num_classes=1, img_size=runner.IMAGE_SIZE
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def metrics(rows: list[dict], predictions: list[dict], score_key: str) -> dict:
    labels = [int(row["label"]) for row in predictions]
    scores = [float(row[score_key]) for row in predictions]
    return {
        "count": len(predictions),
        "auc": float(roc_auc_score(labels, scores)),
        "groups": runner.grouped_metrics(rows, labels, scores),
    }


@torch.inference_mode()
def evaluate_pair(
    v6_model: torch.nn.Module,
    v9_model: torch.nn.Module,
    dataset: runner.ManifestDataset,
) -> tuple[dict, list[dict]]:
    loader = DataLoader(
        dataset,
        batch_size=INFERENCE_POLICY["physical_batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    predictions = []
    for batch_number, (images, labels, indices) in enumerate(loader, 1):
        # The same transformed CPU batch is copied to both GPUs.  Launch both
        # forwards before synchronizing either result, so the two independent
        # encoders run concurrently without changing the numerical policy.
        images_v6 = images.to("cuda:0", non_blocking=True)
        images_v9 = images.to("cuda:1", non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            v6_scores = torch.sigmoid(v6_model(images_v6).flatten())
            v9_scores = torch.sigmoid(v9_model(images_v9).flatten())
        v6_scores_cpu = v6_scores.float().cpu()
        v9_scores_cpu = v9_scores.float().cpu()
        blend_scores_cpu = V6_WEIGHT * v6_scores_cpu + V9_WEIGHT * v9_scores_cpu
        for index, label, v6_score, blend_score in zip(
            indices.tolist(),
            labels.tolist(),
            v6_scores_cpu.tolist(),
            blend_scores_cpu.tolist(),
        ):
            source = dataset.rows[int(index)]
            predictions.append(
                {
                    "index": int(index),
                    "label": int(label),
                    "v6_score": float(v6_score),
                    "score": float(blend_score),
                    **{
                        key: source[key]
                        for key in (
                            "generator",
                            "generator_model",
                            "real_source",
                            "family",
                            "image_sha256",
                        )
                        if key in source
                    },
                }
            )
        if batch_number % 25 == 0 or batch_number == len(loader):
            print(
                json.dumps(
                    {
                        "phase": "paired_evaluate",
                        "batch": batch_number,
                        "batches": len(loader),
                        "images": len(predictions),
                    }
                ),
                flush=True,
            )
    return {
        "v6": metrics(dataset.rows, predictions, "v6_score"),
        "blend": metrics(dataset.rows, predictions, "score"),
    }, predictions


def gate_summary(rows: list[dict], conditions: dict, pooled: list[dict]) -> dict:
    summary = {}
    for candidate, score_key in (("v6", "v6_score"), ("blend", "score")):
        clean = conditions["clean"][candidate]
        pooled_metrics = metrics(
            rows * (len(stress.conditions()) - 1), pooled, score_key
        )
        summary[candidate] = {
            "clean_auc": clean["auc"],
            "pooled_robust_auc": pooled_metrics["auc"],
            "official_score": 0.5 * clean["auc"] + 0.5 * pooled_metrics["auc"],
            "clean_worst_pair_auc": clean["groups"][
                "worst_generator_real_source_pair_auc"
            ],
            "pooled_worst_pair_auc": pooled_metrics["groups"][
                "worst_generator_real_source_pair_auc"
            ],
            "noise_sigma_0.10_auc": conditions["noise_sigma_0.10"][candidate][
                "auc"
            ],
        }
    summary["delta_blend_minus_v6"] = {
        key: summary["blend"][key] - summary["v6"][key]
        for key in summary["v6"]
    }
    return summary


def evaluate_gate(
    name: str,
    manifest: Path,
    rows: list[dict],
    v6_model: torch.nn.Module,
    v9_model: torch.nn.Module,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    signature: dict,
) -> dict:
    output = OUTPUT_ROOT / name
    progress_path = output / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if progress.get("signature") != signature:
            raise RuntimeError(f"incompatible resume state: {progress_path}")
    else:
        progress = {"completed": False, "signature": signature, "conditions": {}}
    pooled = []
    started = time.time()
    for condition_index, (condition, image_transform, tensor_noise) in enumerate(
        stress.conditions()
    ):
        prediction_path = output / f"{condition}_predictions.jsonl"
        if condition in progress["conditions"] and prediction_path.is_file():
            predictions = read_jsonl(prediction_path)
            if len(predictions) != len(rows):
                raise RuntimeError(f"invalid resume count for {name}/{condition}")
            print(json.dumps({"resumed": name, "condition": condition}), flush=True)
        else:
            torch.manual_seed(SEED + condition_index)
            dataset = runner.ManifestDataset(
                manifest,
                stress.condition_transform(mean, std, image_transform, tensor_noise),
                rows=rows,
            )
            condition_metrics, predictions = evaluate_pair(v6_model, v9_model, dataset)
            write_jsonl(prediction_path, predictions)
            progress["conditions"][condition] = condition_metrics
            atomic_json(progress_path, progress)
            print(
                json.dumps(
                    {
                        "saved_gate": name,
                        "condition": condition,
                        "v6_auc": condition_metrics["v6"]["auc"],
                        "blend_auc": condition_metrics["blend"]["auc"],
                    }
                ),
                flush=True,
            )
        if condition != "clean":
            pooled.extend(predictions)
    progress.update(
        {
            "completed": True,
            "summary": gate_summary(rows, progress["conditions"], pooled),
            "elapsed_seconds_this_process": time.time() - started,
        }
    )
    atomic_json(progress_path, progress)
    return progress


def assessment(name: str, summary: dict) -> dict:
    delta = summary["delta_blend_minus_v6"]
    if name == "qwen_prompt_holdout":
        checks = {
            "clean_improved": delta["clean_auc"] > 0.0,
            "worst_pair_improved": delta["clean_worst_pair_auc"] > 0.0,
            "official_score_within_tolerance": delta["official_score"] >= -0.002,
        }
    else:
        checks = {
            key: delta[key] >= floor
            for key, floor in NON_FRONTIER_TOLERANCES.items()
        }
        checks["clean_worst_pair_auc"] = delta["clean_worst_pair_auc"] >= -0.01
        checks["pooled_worst_pair_auc"] = delta["pooled_worst_pair_auc"] >= -0.01
    return {"checks": checks, "passes": all(checks.values())}


def main() -> None:
    if torch.cuda.device_count() < 2:
        raise RuntimeError("paired evaluator requires the verified two-T4 session")
    package_sha256, package = runner.validate_package()
    internal_manifest = runner.WORK_ROOT / "manifests/eval_selection.jsonl"
    internal_rows = stress.balanced_rows(
        runner.filter_evaluation_rows(
            read_jsonl(internal_manifest), runner.EXCLUDED_EVAL_SHA256
        )
    )
    if len(internal_rows) != 3071:
        raise RuntimeError(f"internal gate row mismatch: {len(internal_rows)}")
    external_manifest, external_rows, external_package, zip_verified = (
        external.validate_and_extract()
    )
    sealed_root, sealed_package = sealed.validate_package()

    v6_model, v6_checkpoint = load_model(selected_v6_path(), V6_SHA256, "cuda:0")
    v9_path = V9_ROOT / MODEL_NAME / "model.pt"
    v9_model, v9_checkpoint = load_model(v9_path, V9_SHA256, "cuda:1")
    mean = tuple(v6_checkpoint["normalization_mean"])
    std = tuple(v6_checkpoint["normalization_std"])
    if mean != tuple(v9_checkpoint["normalization_mean"]) or std != tuple(
        v9_checkpoint["normalization_std"]
    ):
        raise RuntimeError("v6/v9 normalization mismatch")

    gate_inputs = [
        (
            "qwen_prompt_holdout",
            sealed_root / sealed.MANIFESTS["qwen_prompt_holdout"]["path"],
            sealed.MANIFESTS["qwen_prompt_holdout"]["sha256"],
        ),
        (
            "ntire_shard5_full_audit",
            sealed_root / sealed.MANIFESTS["ntire_shard5_full_audit"]["path"],
            sealed.MANIFESTS["ntire_shard5_full_audit"]["sha256"],
        ),
        ("internal_3071", internal_manifest, runner.file_sha256(internal_manifest)),
        ("community_forensics_624", external_manifest, external.EXPECTED_MANIFEST_SHA256),
    ]
    row_map = {
        "qwen_prompt_holdout": read_jsonl(gate_inputs[0][1]),
        "ntire_shard5_full_audit": read_jsonl(gate_inputs[1][1]),
        "internal_3071": internal_rows,
        "community_forensics_624": external_rows,
    }
    for device_index in (0, 1):
        torch.cuda.reset_peak_memory_stats(device_index)
    started = time.time()
    gates = {}
    for name, manifest, manifest_sha256 in gate_inputs:
        signature = {
            "name": name,
            "manifest_sha256": manifest_sha256,
            "rows": len(row_map[name]),
            "v6_checkpoint_sha256": V6_SHA256,
            "v9_checkpoint_sha256": V9_SHA256,
            "v6_weight": V6_WEIGHT,
            "v9_weight": V9_WEIGHT,
            "inference_policy": INFERENCE_POLICY,
            "conditions": [condition[0] for condition in stress.conditions()],
            "seed": SEED,
        }
        gates[name] = evaluate_gate(
            name,
            manifest,
            row_map[name],
            v6_model,
            v9_model,
            mean,
            std,
            signature,
        )

    assessments = {
        name: assessment(name, progress["summary"])
        for name, progress in gates.items()
    }
    summary = {
        "completed": all(progress["completed"] for progress in gates.values()),
        "preselected_blend": {"v6_weight": V6_WEIGHT, "v9_weight": V9_WEIGHT},
        "sealed_holdout_opened_once": True,
        "v9_standalone_sealed_metrics_reported": False,
        "v6_checkpoint_sha256": V6_SHA256,
        "v9_checkpoint_sha256": V9_SHA256,
        "internal_package_sha256": package_sha256,
        "internal_package_inventory_sha256": package["inventory_sha256"],
        "promotion_package_inventory_sha256": sealed_package["inventory_sha256"],
        "external_package_inventory_sha256": external_package["inventory_sha256"],
        "external_zip_transport_verified": zip_verified,
        "gates": {name: progress["summary"] for name, progress in gates.items()},
        "assessments": assessments,
        "passes_all_promotion_gates": all(
            result["passes"] for result in assessments.values()
        ),
        "elapsed_seconds": time.time() - started,
        "cuda_peak_allocated_bytes": {
            str(device_index): torch.cuda.max_memory_allocated(device_index)
            for device_index in (0, 1)
        },
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(index) for index in (0, 1)],
    }
    atomic_json(OUTPUT_ROOT / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
