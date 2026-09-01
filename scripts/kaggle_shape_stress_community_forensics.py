"""Run label-independent codec/geometry controls on the external PE gate."""

from __future__ import annotations

import json
import time
from pathlib import Path

import timm
import torch

import kaggle_evaluate_community_forensics as external
import kaggle_shape_stress_v6 as shape
import kaggle_stress_eval_v6 as stress
import kaggle_train_v3 as runner
import kaggle_train_v6  # noqa: F401  # apply checksum-pinned v6 constants


def main() -> None:
    manifest, rows, package, zip_transport_verified = external.validate_and_extract()
    descriptors = shape.inspect_originals(manifest, rows)
    model_name = runner.MODEL_NAMES[0]
    model_output = runner.OUTPUT_ROOT / model_name.replace(".", "_")
    checkpoint_path = model_output / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = timm.create_model(
        model_name, pretrained=False, num_classes=1, img_size=runner.IMAGE_SIZE
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda().eval()
    mean = tuple(checkpoint["normalization_mean"])
    std = tuple(checkpoint["normalization_std"])

    output = model_output / "community-forensics-external-shape-audit"
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    signature = {
        "completed": False,
        "training_allowed": False,
        "gate_inventory_sha256": package["inventory_sha256"],
        "gate_manifest_sha256": external.EXPECTED_MANIFEST_SHA256,
        "zip_transport_verified": zip_transport_verified,
        "checkpoint_sha256": runner.file_sha256(checkpoint_path),
        "gate_rows": len(rows),
        "conditions": {},
    }
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        for key, value in signature.items():
            if key != "conditions" and progress.get(key) != value:
                raise RuntimeError(f"incompatible resume state for {key}")
    else:
        progress = signature

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for condition in (
        "jpeg_q96_stretch_full_frame",
        "jpeg_q96_square_patch_75",
        "jpeg_q96_forced_aspect",
    ):
        prediction_path = output / f"{condition}_predictions.jsonl"
        if condition in progress["conditions"] and prediction_path.is_file():
            predictions = stress.read_predictions(prediction_path)
            if len(predictions) != len(rows):
                raise RuntimeError(
                    f"invalid saved prediction count for {condition}: {len(predictions)}"
                )
            print(json.dumps({"resumed_condition": condition}), flush=True)
        else:
            dataset = shape.GeometryDataset(
                manifest, rows, mean, std, condition
            )
            metrics, predictions = runner.evaluate(model, dataset)
            metrics["original_metadata_subgroups"] = shape.metadata_subgroups(
                predictions, descriptors
            )
            metrics["per_generator_model"] = external.per_generator_model_metrics(
                rows, predictions
            )
            progress["conditions"][condition] = metrics
            stress.write_predictions(prediction_path, predictions)
            stress.write_json(progress_path, progress)
            print(
                json.dumps({"saved_condition": condition, "auc": metrics["clean_auc"]}),
                flush=True,
            )

    progress.update(
        {
            "completed": True,
            "elapsed_seconds": time.time() - started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        }
    )
    stress.write_json(progress_path, progress)
    print(json.dumps(progress, indent=2), flush=True)


if __name__ == "__main__":
    main()
