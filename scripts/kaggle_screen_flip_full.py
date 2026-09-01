"""Run paired reference/flip inference gates on the selected v6 classifier head.

This is an inference-only ablation.  It reconstructs the public PE-Core-L
backbone under the pinned Kaggle environment, loads the immutable selected v6
classifier head, and evaluates the reference and horizontal-flip-mean policies
on exactly the same decoded and transformed tensors.  It never trains or
calibrates on either gate.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import timm
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import kaggle_evaluate_community_forensics as external
import kaggle_stress_eval_v6 as stress
import kaggle_train_v3 as runner
import kaggle_train_v6  # noqa: F401  # apply checksum-pinned v6 constants


HEAD_PATH = Path("/kaggle/working/selected-v6-classifier-head.pt")
OUTPUT_ROOT = Path("/kaggle/working/track5-v6-paired-flip-gate")
EXPECTED_HEAD_SHA256 = (
    "152e8555754613c230a42e150809e1879d707624d8709b067bb0bc8b4d11a56d"
)
EXPECTED_SOURCE_CHECKPOINT_SHA256 = (
    "4b8f3ac4776b0fddc689252de760d661916d9377374484703487538e8268766a"
)
EXPECTED_MODEL_NAME = "vit_pe_core_large_patch14_336"
EXPECTED_IMAGE_SIZE = 224
EXPECTED_PARAMETERS = 315_776_001

HISTORICAL_REFERENCE = {
    "internal": {
        "clean_auc": 0.991079,
        "pooled_robust_auc": 0.978019,
        "official_score": 0.984549,
        "noise_sigma_0.10": 0.867729,
    },
    "external": {
        "clean_auc": 0.992634,
        "pooled_robust_auc": 0.972330,
        "official_score": 0.982482,
        "noise_sigma_0.10": 0.797466,
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_selected_model() -> tuple[torch.nn.Module, tuple[float, ...], tuple[float, ...], dict]:
    observed_head_sha256 = file_sha256(HEAD_PATH)
    if observed_head_sha256 != EXPECTED_HEAD_SHA256:
        raise RuntimeError(
            f"classifier-head mismatch: {observed_head_sha256} != {EXPECTED_HEAD_SHA256}"
        )
    artifact = torch.load(HEAD_PATH, map_location="cpu", weights_only=True)
    expected_metadata = {
        "format_version": "track5-classifier-head-v1",
        "model_name": EXPECTED_MODEL_NAME,
        "image_size": EXPECTED_IMAGE_SIZE,
        "preprocess_mode": "short_side_crop",
        "source_checkpoint_sha256": EXPECTED_SOURCE_CHECKPOINT_SHA256,
    }
    observed_metadata = {key: artifact.get(key) for key in expected_metadata}
    if observed_metadata != expected_metadata:
        raise RuntimeError(
            f"classifier-head metadata mismatch: {observed_metadata} != {expected_metadata}"
        )

    model = timm.create_model(
        EXPECTED_MODEL_NAME,
        pretrained=True,
        num_classes=1,
        img_size=EXPECTED_IMAGE_SIZE,
    )
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
        raise RuntimeError("unexpected reconstructed parameter count")
    classifier = model.get_classifier()
    classifier.load_state_dict(artifact["classifier_state_dict"], strict=True)
    mean, std = runner.normalization(model)
    model.cuda().eval()
    reconstruction = {
        "policy": "public pinned PE-Core-L backbone plus immutable selected v6 head",
        "not_byte_identical_checkpoint": True,
        "timm": timm.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "parameters": EXPECTED_PARAMETERS,
        "head_sha256": observed_head_sha256,
        "source_checkpoint_sha256": EXPECTED_SOURCE_CHECKPOINT_SHA256,
    }
    return model, tuple(mean), tuple(std), reconstruction


@torch.inference_mode()
def evaluate_pair(model: torch.nn.Module, dataset: runner.ManifestDataset) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    labels: list[int] = []
    indices: list[int] = []
    reference_scores: list[float] = []
    flip_scores: list[float] = []
    for batch_number, (images, batch_labels, batch_indices) in enumerate(loader, 1):
        images = images.cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            reference = torch.sigmoid(model(images).flatten())
            flipped = torch.sigmoid(model(torch.flip(images, dims=(3,))).flatten())
            averaged = (reference + flipped) * 0.5
        labels.extend(int(value) for value in batch_labels.tolist())
        indices.extend(int(value) for value in batch_indices.tolist())
        reference_scores.extend(float(value) for value in reference.float().cpu().tolist())
        flip_scores.extend(float(value) for value in averaged.float().cpu().tolist())
        if batch_number % 25 == 0 or batch_number == len(loader):
            print(
                json.dumps(
                    {
                        "phase": "paired_evaluate",
                        "batch": batch_number,
                        "batches": len(loader),
                        "images": len(labels),
                    }
                ),
                flush=True,
            )

    def predictions(scores: list[float]) -> list[dict]:
        return [
            {
                "index": index,
                "label": label,
                "score": score,
                **{
                    key: dataset.rows[index][key]
                    for key in (
                        "generator",
                        "generator_model",
                        "real_source",
                        "family",
                        "image_sha256",
                    )
                    if key in dataset.rows[index]
                },
            }
            for index, label, score in zip(indices, labels, scores)
        ]

    def metrics(scores: list[float]) -> dict:
        return {
            "count": len(labels),
            "auc": float(roc_auc_score(labels, scores)),
            "groups": runner.grouped_metrics(dataset.rows, labels, scores),
        }

    return {
        "reference": {
            "metrics": metrics(reference_scores),
            "predictions": predictions(reference_scores),
        },
        "reference_flip_mean": {
            "metrics": metrics(flip_scores),
            "predictions": predictions(flip_scores),
        },
    }


def read_predictions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_predictions(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(path)


def run_gate(
    gate_name: str,
    model: torch.nn.Module,
    manifest: Path,
    rows: list[dict],
    mean: tuple[float, ...],
    std: tuple[float, ...],
    signature: dict,
) -> dict:
    output = OUTPUT_ROOT / gate_name
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        observed = {key: progress.get(key) for key in signature}
        if observed != signature:
            raise RuntimeError(f"incompatible resume state: {observed} != {signature}")
    else:
        progress = {"completed": False, **signature, "conditions": {}}

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for index, (name, image_transform, tensor_noise) in enumerate(stress.conditions()):
        paths = {
            policy: output / f"{name}_{policy}_predictions.jsonl"
            for policy in ("reference", "reference_flip_mean")
        }
        if name in progress["conditions"] and all(path.is_file() for path in paths.values()):
            pair = {
                policy: {
                    "metrics": progress["conditions"][name][policy],
                    "predictions": read_predictions(path),
                }
                for policy, path in paths.items()
            }
            if any(len(value["predictions"]) != len(rows) for value in pair.values()):
                raise RuntimeError(f"invalid resume predictions for {gate_name}:{name}")
            print(json.dumps({"resumed_gate": gate_name, "condition": name}), flush=True)
        else:
            torch.manual_seed(runner.SEED + index)
            dataset = runner.ManifestDataset(
                manifest,
                stress.condition_transform(mean, std, image_transform, tensor_noise),
                rows=rows,
            )
            pair = evaluate_pair(model, dataset)
            progress["conditions"][name] = {
                policy: value["metrics"] for policy, value in pair.items()
            }
            for policy, path in paths.items():
                write_predictions(path, pair[policy]["predictions"])
            write_json(progress_path, progress)
            print(
                json.dumps(
                    {
                        "saved_gate": gate_name,
                        "condition": name,
                        "reference_auc": pair["reference"]["metrics"]["auc"],
                        "flip_auc": pair["reference_flip_mean"]["metrics"]["auc"],
                    }
                ),
                flush=True,
            )

    policies = {}
    for policy in ("reference", "reference_flip_mean"):
        clean = progress["conditions"]["clean"][policy]["auc"]
        robust_labels: list[int] = []
        robust_scores: list[float] = []
        for name, _, _ in stress.conditions():
            if name == "clean":
                continue
            for row in read_predictions(output / f"{name}_{policy}_predictions.jsonl"):
                robust_labels.append(int(row["label"]))
                robust_scores.append(float(row["score"]))
        robust = float(roc_auc_score(robust_labels, robust_scores))
        policies[policy] = {
            "clean_auc": clean,
            "pooled_robust_auc": robust,
            "official_score": 0.5 * clean + 0.5 * robust,
            "noise_sigma_0.10": progress["conditions"]["noise_sigma_0.10"][policy]["auc"],
        }

    deltas = {
        name: progress["conditions"][name]["reference_flip_mean"]["auc"]
        - progress["conditions"][name]["reference"]["auc"]
        for name, _, _ in stress.conditions()
    }
    reference = policies["reference"]
    candidate = policies["reference_flip_mean"]
    historical = HISTORICAL_REFERENCE[gate_name]
    historical_reference_check = {
        key: reference[key] - historical[key]
        for key in historical
    }
    gate_pass = (
        candidate["official_score"] > reference["official_score"]
        and candidate["clean_auc"] >= reference["clean_auc"] - 0.002
        and deltas["noise_sigma_0.10"] > 0.0
        and min(deltas.values()) >= -0.01
    )
    progress.update(
        {
            "completed": True,
            "policies": policies,
            "condition_auc_deltas_flip_minus_reference": deltas,
            "worst_condition_delta": min(deltas.values()),
            "historical_reference_delta": historical_reference_check,
            "paired_gate_pass": gate_pass,
            "elapsed_seconds_this_invocation": time.time() - started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
    )
    write_json(progress_path, progress)
    return progress


def main() -> None:
    model, mean, std, reconstruction = load_selected_model()
    package_sha256, metadata = runner.validate_package()
    internal_manifest = runner.WORK_ROOT / "manifests/eval_selection.jsonl"
    internal_rows = runner.filter_evaluation_rows(
        stress.read_rows(internal_manifest), runner.EXCLUDED_EVAL_SHA256
    )
    internal_rows = stress.balanced_rows(internal_rows)
    if len(internal_rows) != 3071:
        raise RuntimeError(f"unexpected internal gate size: {len(internal_rows)}")
    common_signature = {
        "policy": "reference versus reference_flip_mean",
        "head_sha256": EXPECTED_HEAD_SHA256,
        "source_checkpoint_sha256": EXPECTED_SOURCE_CHECKPOINT_SHA256,
        "model_name": EXPECTED_MODEL_NAME,
        "image_size": EXPECTED_IMAGE_SIZE,
    }
    internal = run_gate(
        "internal",
        model,
        internal_manifest,
        internal_rows,
        mean,
        std,
        {
            **common_signature,
            "rows": len(internal_rows),
            "package_sha256": package_sha256,
            "inventory_sha256": metadata["inventory_sha256"],
        },
    )

    external_manifest, external_rows, package, zip_transport_verified = (
        external.validate_and_extract()
    )
    independent = run_gate(
        "external",
        model,
        external_manifest,
        external_rows,
        mean,
        std,
        {
            **common_signature,
            "rows": len(external_rows),
            "gate_manifest_sha256": external.EXPECTED_MANIFEST_SHA256,
            "gate_inventory_sha256": package["inventory_sha256"],
            "zip_transport_verified": zip_transport_verified,
            "training_allowed": False,
        },
    )
    summary = {
        "completed": True,
        "reconstruction": reconstruction,
        "internal": {
            key: internal[key]
            for key in (
                "policies",
                "condition_auc_deltas_flip_minus_reference",
                "worst_condition_delta",
                "historical_reference_delta",
                "paired_gate_pass",
            )
        },
        "external": {
            key: independent[key]
            for key in (
                "policies",
                "condition_auc_deltas_flip_minus_reference",
                "worst_condition_delta",
                "historical_reference_delta",
                "paired_gate_pass",
            )
        },
        "promotion_screen_pass": bool(
            internal["paired_gate_pass"] and independent["paired_gate_pass"]
        ),
        "promotion_decision": "screen_only_not_automatic",
    }
    write_json(OUTPUT_ROOT / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
