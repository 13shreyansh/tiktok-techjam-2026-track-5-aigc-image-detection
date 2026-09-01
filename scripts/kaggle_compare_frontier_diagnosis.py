#!/usr/bin/env python3
"""Compare selected v6 and rejected v9 on the already-open frontier audit.

This is diagnosis, not a promotion gate.  The prompt/pixel-sealed Qwen holdout
must remain unopened because v9 failed the predeclared frozen internal floors.
"""

from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path

import timm
import torch

import kaggle_train_v3 as runner


PACKAGE_NAME = "qwen-frontier-diagnosis-gate.zip"
PACKAGE_BYTES = 331_050_025
PACKAGE_SHA256 = "fad723b52a849550e01e4ff3cc7b08bd4aeec1bb35ec515ce9cec0c64b508cc4"
INVENTORY_SHA256 = "ae99a1e98c01f0a91a26aa7bc9728bbd7c8cbba0c708b55647e8e5375ecc8a48"
MANIFEST_SHA256 = "bfc516ede0b70da5fb572bfc4294d16b49af51d7725b125d0d5c4b66813e3e7a"
UNIQUE_IMAGES = 576
SOURCE_BYTES = 330_663_624
MODEL_NAME = "vit_pe_core_large_patch14_336"
WORK_ROOT = Path("/kaggle/working/qwen-frontier-diagnosis-gate")
OUTPUT_ROOT = Path("/kaggle/working/track5-frontier-diagnosis-comparison")
CANDIDATES = {
    "v6-selected-fp16": {
        "checkpoint_sha256": "48ea50773fbd1b7247fff25fde6f985183e29f2eb517b5ac0f6319c1fe38b644",
        "location": "input-v6",
    },
    "v9-frontier-capped": {
        "checkpoint_sha256": "dd6b26c7849489447c7e96823f5b5e87c31623ca8ae0d28a5b162bb2dcb65075",
        "location": "working-v9",
    },
}


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"unsafe archive member: {member.filename}")
    archive.extractall(destination)


def validate_package() -> tuple[Path, str]:
    archives = list(Path("/kaggle/input").rglob(PACKAGE_NAME))
    if len(archives) == 1:
        archive = archives[0]
        if archive.stat().st_size != PACKAGE_BYTES:
            raise RuntimeError("diagnosis package byte mismatch")
        if runner.file_sha256(archive) != PACKAGE_SHA256:
            raise RuntimeError("diagnosis package SHA-256 mismatch")
        if WORK_ROOT.exists():
            shutil.rmtree(WORK_ROOT)
        WORK_ROOT.mkdir(parents=True)
        with zipfile.ZipFile(archive) as handle:
            safe_extract(handle, WORK_ROOT)
        root = WORK_ROOT
        transport = "uploaded ZIP byte checksum"
    elif not archives:
        candidates = []
        for report_path in Path("/kaggle/input").rglob("package.json"):
            try:
                report = json.loads(report_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if report.get("inventory_sha256") == INVENTORY_SHA256:
                candidates.append(report_path.parent)
        if len(candidates) != 1:
            raise RuntimeError(f"expected one expanded diagnosis package: {candidates}")
        root = candidates[0]
        transport = "Kaggle-expanded content-addressed files"
    else:
        raise RuntimeError(f"multiple diagnosis packages: {archives}")

    report = json.loads((root / "package.json").read_text())
    expected = {
        "unique_images": UNIQUE_IMAGES,
        "source_bytes": SOURCE_BYTES,
        "inventory_sha256": INVENTORY_SHA256,
    }
    if {key: report.get(key) for key in expected} != expected:
        raise RuntimeError("diagnosis inventory mismatch")
    images = sorted(path for path in (root / "images").rglob("*") if path.is_file())
    if len(images) != UNIQUE_IMAGES:
        raise RuntimeError(f"diagnosis image count mismatch: {len(images)}")
    observed_bytes = 0
    for index, image in enumerate(images, 1):
        if runner.file_sha256(image) != image.stem:
            raise RuntimeError(f"diagnosis image mismatch: {image}")
        observed_bytes += image.stat().st_size
        if index % 200 == 0 or index == len(images):
            print(json.dumps({"verified_diagnosis_images": index}), flush=True)
    if observed_bytes != SOURCE_BYTES:
        raise RuntimeError(f"diagnosis byte mismatch: {observed_bytes}")
    manifest = root / "manifests/combined_gate.jsonl"
    if runner.file_sha256(manifest) != MANIFEST_SHA256:
        raise RuntimeError("diagnosis manifest mismatch")
    if len([line for line in manifest.read_text().splitlines() if line]) != UNIQUE_IMAGES:
        raise RuntimeError("diagnosis manifest row mismatch")
    return manifest, transport


def checkpoint_path(name: str) -> Path:
    if CANDIDATES[name]["location"] == "working-v9":
        return (
            Path("/kaggle/working/track5-v9-frontier-capped-candidate")
            / MODEL_NAME
            / "model.pt"
        )
    matches = [
        path
        for path in Path("/kaggle/input").rglob("model.pt")
        if "track5-v6-selected-fp16-checkpoint" in str(path)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one selected v6 checkpoint: {matches}")
    return matches[0]


def evaluate_candidate(name: str, manifest: Path, rows: list[dict]) -> dict:
    path = checkpoint_path(name)
    observed_sha256 = runner.file_sha256(path)
    if observed_sha256 != CANDIDATES[name]["checkpoint_sha256"]:
        raise RuntimeError(f"{name} checkpoint mismatch: {observed_sha256}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["model_name"] != MODEL_NAME:
        raise RuntimeError(f"{name} model mismatch")
    model = timm.create_model(
        MODEL_NAME, pretrained=False, num_classes=1, img_size=runner.IMAGE_SIZE
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda().eval()
    mean = tuple(checkpoint["normalization_mean"])
    std = tuple(checkpoint["normalization_std"])
    dataset = runner.ManifestDataset(
        manifest, runner.eval_transform(mean, std), rows=rows
    )
    started = time.time()
    metrics, predictions = runner.evaluate(model, dataset)
    output = OUTPUT_ROOT / name
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions)
    )
    report = {
        "candidate": name,
        "checkpoint_sha256": observed_sha256,
        "diagnosis_only": True,
        "metrics": metrics,
        "elapsed_seconds": time.time() - started,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    del model, dataset
    torch.cuda.empty_cache()
    return report


def main() -> None:
    manifest, transport = validate_package()
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    torch.cuda.reset_peak_memory_stats()
    reports = {
        name: evaluate_candidate(name, manifest, rows) for name in CANDIDATES
    }
    summary = {
        "diagnosis_only": True,
        "sealed_holdout_opened": False,
        "package_sha256": PACKAGE_SHA256,
        "package_runtime_verification": transport,
        "results": {
            name: {
                "clean_auc": report["metrics"]["clean_auc"],
                "groups": report["metrics"]["groups"],
                "checkpoint_sha256": report["checkpoint_sha256"],
            }
            for name, report in reports.items()
        },
        "clean_auc_delta_v9_minus_v6": (
            reports["v9-frontier-capped"]["metrics"]["clean_auc"]
            - reports["v6-selected-fp16"]["metrics"]["clean_auc"]
        ),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
