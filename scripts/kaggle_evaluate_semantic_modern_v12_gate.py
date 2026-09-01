#!/usr/bin/env python3
"""One-shot v12 audit on a frozen content-matched modern-generator gate.

The gate was frozen without candidate detector scores. This evaluator verifies
the package, the audit-only row contract, separation from v12 train/evaluation
identities and the exact checkpoint before scoring clean plus the 19 workshop
transformations. It never trains, tunes, calibrates or selects a threshold.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import timm
import torch
from sklearn.metrics import roc_auc_score

import kaggle_evaluate_v12_robustness as workshop
import kaggle_train_v3 as runner


SEED = 20260901
GATE_PACKAGE_NAME = "semantic-matched-modern-v6.zip"
GATE_PACKAGE_SHA256 = "52f6749bed16015a1511e6cc6e9e7072d50350b3755148e1c3bea6d645288d69"
GATE_INVENTORY_SHA256 = "86da500fcbbfe76730940bbbf82d789d6773e90086b3f48ab0889adb29ad8496"
GATE_MANIFEST_SHA256 = "8f3372c1e37d0b508e2333e72ee65de18a18fa97b5270c751a07cb2f8368a792"
V12_INVENTORY_SHA256 = "ec78d74e62d8e1b1f75e661f2ea3338fa95be11e96694a3ed168b463fe314fa6"
CANONICALIZATION = "exif_transpose_center_square_resize336_jpeg_q96_subsampling0"
EXPECTED_PROMPTS = {445, 467, 474, 539, 653, 759, 851, 870}
EXPECTED_GENERATORS = {
    "FLUX.2-pro",
    "FLUX.2_max",
    "GLM-Image",
    "GPT-Image-1",
    "GPT-Image-1.5",
    "HunyuanImage-3.0",
    "Imagen-4.0",
    "Imagen-4.0-Ultra",
    "Qwen-Image",
    "Qwen-Image-2.0-pro",
    "Qwen-Image-2512",
    "Seedream-4.0",
    "Seedream-4.5",
    "Seedream-5.0",
    "gpt-image-2",
    "kling_v2_1",
    "nano-banana-2.0",
    "nano-banana-pro",
}
GATE_WORK_ROOT = Path("/kaggle/working/semantic-matched-modern-v6")
GATE_STAGING_ROOT = Path("/kaggle/working/semantic-gate-package")

CANDIDATES = {
    "pe_core": {
        "model": "vit_pe_core_large_patch14_336",
        "root": Path("/kaggle/working/track5-v12-permissive"),
        "checkpoint_sha256": "f37bd6b445b12257ff29a9e54946c5bf9a9184e86a45dc5490537b9ea325ddd2",
    },
    "dinov2_control": {
        "model": "vit_large_patch14_dinov2.lvd142m",
        "root": Path("/kaggle/working/track5-v12-dino-control"),
        "checkpoint_sha256": "db07f30cbc94e4972f4a8c72c95bbe5df0dcd40b2dbd494ad82dc324cc1e2b5b",
    },
}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def validate_gate_package() -> tuple[Path, dict]:
    """Verify the frozen gate from a mounted input or an exact working copy.

    Kaggle working storage is session-local.  A recovered private package may
    therefore arrive through the authenticated Kaggle CLI instead of the
    notebook input mount.  This fallback changes only the package locator: the
    same frozen archive hash, inventory hash and runner validation still apply.
    The input root is restored immediately so v12 identity manifests must still
    come from the separately mounted canonical-v12 input.
    """

    staged_package = GATE_STAGING_ROOT / GATE_PACKAGE_NAME
    previous = {
        "package_name": runner.PACKAGE_NAME,
        "zip_sha256": runner.EXPECTED_ZIP_SHA256,
        "inventory_sha256": runner.EXPECTED_INVENTORY_SHA256,
        "work_root": runner.WORK_ROOT,
        "input_root": runner.INPUT_ROOT,
    }
    runner.PACKAGE_NAME = GATE_PACKAGE_NAME
    runner.EXPECTED_ZIP_SHA256 = GATE_PACKAGE_SHA256
    runner.EXPECTED_INVENTORY_SHA256 = GATE_INVENTORY_SHA256
    runner.WORK_ROOT = GATE_WORK_ROOT
    if staged_package.is_file():
        observed = runner.file_sha256(staged_package)
        if observed != GATE_PACKAGE_SHA256:
            raise RuntimeError(
                f"semantic gate working-copy checksum mismatch: {observed}"
            )
        runner.INPUT_ROOT = GATE_STAGING_ROOT.parent
    try:
        _, metadata = runner.validate_package()
        validated_root = Path(runner.WORK_ROOT)
        return validated_root, metadata
    finally:
        runner.PACKAGE_NAME = previous["package_name"]
        runner.EXPECTED_ZIP_SHA256 = previous["zip_sha256"]
        runner.EXPECTED_INVENTORY_SHA256 = previous["inventory_sha256"]
        runner.WORK_ROOT = previous["work_root"]
        runner.INPUT_ROOT = previous["input_root"]


def validate_gate_rows(rows: list[dict]) -> dict:
    labels = Counter(int(row["label"]) for row in rows)
    if len(rows) != 288 or labels != Counter({0: 144, 1: 144}):
        raise RuntimeError(f"unexpected gate balance: rows={len(rows)}, labels={labels}")
    if len({row["image_sha256"] for row in rows}) != 288:
        raise RuntimeError("gate canonical images are not unique")
    if len({row["source_image_sha256"] for row in rows}) != 288:
        raise RuntimeError("gate source images are not unique")

    by_pair: dict[str, list[dict]] = defaultdict(list)
    prompt_labels = Counter()
    fake_generators = Counter()
    for index, row in enumerate(rows):
        if row.get("workflow_purpose") != "semantic-matched-modern-audit":
            raise RuntimeError(f"row {index}: purpose mismatch")
        if row.get("training_allowed") is not False or row.get(
            "organizer_demo_row"
        ) is not False:
            raise RuntimeError(f"row {index}: audit-use contract mismatch")
        if row.get("canonicalization") != CANONICALIZATION:
            raise RuntimeError(f"row {index}: canonicalization mismatch")
        if row.get("canonical_format") != "JPEG" or (
            row.get("canonical_width"), row.get("canonical_height")
        ) != (336, 336):
            raise RuntimeError(f"row {index}: canonical image contract mismatch")
        prompt = int(row["semantic_prompt_id"])
        if prompt not in EXPECTED_PROMPTS:
            raise RuntimeError(f"row {index}: unexpected prompt {prompt}")
        generator = str(row["paired_generator"])
        if generator not in EXPECTED_GENERATORS:
            raise RuntimeError(f"row {index}: unexpected generator {generator}")
        by_pair[str(row["pair_id"])].append(row)
        prompt_labels[(prompt, int(row["label"]))] += 1
        if int(row["label"]) == 1:
            fake_generators[generator] += 1

    if len(by_pair) != 144:
        raise RuntimeError(f"unexpected pair count: {len(by_pair)}")
    for pair_id, pair in by_pair.items():
        if len(pair) != 2 or {int(row["label"]) for row in pair} != {0, 1}:
            raise RuntimeError(f"invalid pair: {pair_id}")
        if len({row["semantic_prompt_id"] for row in pair}) != 1 or len(
            {row["paired_generator"] for row in pair}
        ) != 1:
            raise RuntimeError(f"pair metadata mismatch: {pair_id}")
    if any(prompt_labels[(prompt, label)] != 18 for prompt in EXPECTED_PROMPTS for label in (0, 1)):
        raise RuntimeError(f"prompt balance mismatch: {prompt_labels}")
    if fake_generators != Counter({generator: 8 for generator in EXPECTED_GENERATORS}):
        raise RuntimeError(f"generator balance mismatch: {fake_generators}")
    return {
        "rows": 288,
        "labels": dict(labels),
        "pairs": 144,
        "prompts": sorted(EXPECTED_PROMPTS),
        "generators": sorted(EXPECTED_GENERATORS),
        "unique_source_images": 288,
        "unique_canonical_images": 288,
        "organizer_demo_rows": 0,
        "training_allowed_rows": 0,
    }


def locate_v12_root() -> Path:
    candidates = []
    for package_path in runner.mounted_root_files("package.json"):
        metadata = json.loads(package_path.read_text())
        if metadata.get("inventory_sha256") == V12_INVENTORY_SHA256:
            candidates.append(package_path.parent)
    if len(candidates) != 1:
        raise RuntimeError(f"expected one mounted v12 root, found {candidates}")
    return candidates[0]


def validate_identity_separation(gate_rows: list[dict], v12_root: Path) -> dict:
    gate_source = {row["source_image_sha256"] for row in gate_rows}
    gate_canonical = {row["image_sha256"] for row in gate_rows}
    compared_rows = 0
    v12_source: set[str] = set()
    v12_canonical: set[str] = set()
    for name in ("train.jsonl", "eval_frozen.jsonl"):
        path = v12_root / "manifests" / name
        if not path.is_file():
            raise RuntimeError(f"missing mounted v12 manifest: {path}")
        rows = read_rows(path)
        compared_rows += len(rows)
        v12_source.update(row["source_image_sha256"] for row in rows)
        v12_canonical.update(row["image_sha256"] for row in rows)
    if gate_source & v12_source or gate_canonical & v12_canonical:
        raise RuntimeError("semantic gate overlaps v12 train/evaluation identities")
    return {
        "v12_rows_compared": compared_rows,
        "source_identity_overlap": 0,
        "canonical_identity_overlap": 0,
        "v12_inventory_sha256": V12_INVENTORY_SHA256,
    }


def paired_accuracy(rows: list[dict], scores: list[float]) -> float:
    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    for row, score in zip(rows, scores):
        grouped[str(row["pair_id"])][int(row["label"])] = float(score)
    values = []
    for pair_id, pair in grouped.items():
        if set(pair) != {0, 1}:
            raise RuntimeError(f"missing pair score: {pair_id}")
        values.append(1.0 if pair[1] > pair[0] else 0.5 if pair[1] == pair[0] else 0.0)
    return float(sum(values) / len(values))


def semantic_metrics(rows: list[dict], predictions: list[dict]) -> dict:
    if len(rows) != len(predictions):
        raise RuntimeError("semantic prediction row count mismatch")
    ordered = sorted(predictions, key=lambda row: int(row["index"]))
    if [int(row["index"]) for row in ordered] != list(range(len(rows))):
        raise RuntimeError("semantic prediction indices are incomplete")
    if any(int(pred["label"]) != int(row["label"]) for row, pred in zip(rows, ordered)):
        raise RuntimeError("semantic prediction label mismatch")
    scores = [float(row["score"]) for row in ordered]
    labels = [int(row["label"]) for row in rows]

    by_prompt = {}
    for prompt in sorted(EXPECTED_PROMPTS):
        indices = [i for i, row in enumerate(rows) if int(row["semantic_prompt_id"]) == prompt]
        subset_rows = [rows[i] for i in indices]
        subset_scores = [scores[i] for i in indices]
        by_prompt[str(prompt)] = {
            "rows": len(indices),
            "auc": float(roc_auc_score([int(row["label"]) for row in subset_rows], subset_scores)),
            "paired_accuracy": paired_accuracy(subset_rows, subset_scores),
        }

    by_generator = {}
    for generator in sorted(EXPECTED_GENERATORS):
        indices = [i for i, row in enumerate(rows) if str(row["paired_generator"]) == generator]
        subset_rows = [rows[i] for i in indices]
        subset_scores = [scores[i] for i in indices]
        by_generator[generator] = {
            "rows": len(indices),
            "auc": float(roc_auc_score([int(row["label"]) for row in subset_rows], subset_scores)),
            "paired_accuracy": paired_accuracy(subset_rows, subset_scores),
            "small_sample_warning": "eight matched pairs",
        }
    return {
        "overall_auc": float(roc_auc_score(labels, scores)),
        "paired_accuracy": paired_accuracy(rows, scores),
        "by_prompt": by_prompt,
        "worst_prompt_auc": min(value["auc"] for value in by_prompt.values()),
        "by_generator": by_generator,
        "worst_generator_auc_diagnostic_only": min(value["auc"] for value in by_generator.values()),
    }


def gate_decision(metrics: dict, official_score: float, worst_condition_auc: float) -> dict:
    floors = {
        "clean_overall_roc_auc": 0.70,
        "paired_accuracy": 0.65,
        "worst_prompt_roc_auc": 0.55,
        "official_style_score": 0.65,
        "worst_individual_transform_roc_auc": 0.55,
    }
    observed = {
        "clean_overall_roc_auc": metrics["overall_auc"],
        "paired_accuracy": metrics["paired_accuracy"],
        "worst_prompt_roc_auc": metrics["worst_prompt_auc"],
        "official_style_score": official_score,
        "worst_individual_transform_roc_auc": worst_condition_auc,
    }
    passes = {key: observed[key] >= floor for key, floor in floors.items()}
    return {
        "floors": floors,
        "observed": observed,
        "passes": passes,
        "passes_all_frozen_floors": all(passes.values()),
        "boundary": "A pass means only that this candidate survived this small frozen audit; it is not hidden-set proof.",
    }


def evaluate_candidate(name: str) -> dict:
    if name not in CANDIDATES:
        raise ValueError(f"unknown candidate: {name}")
    specification = CANDIDATES[name]
    gate_root, package = validate_gate_package()
    manifest = gate_root / "manifests/eval_semantic_matched.jsonl"
    if runner.file_sha256(manifest) != GATE_MANIFEST_SHA256:
        raise RuntimeError("semantic gate manifest checksum mismatch")
    rows = read_rows(manifest)
    compliance = validate_gate_rows(rows)
    separation = validate_identity_separation(rows, locate_v12_root())

    model_name = specification["model"]
    model_root = specification["root"] / model_name.replace(".", "_")
    checkpoint_path = model_root / "model.pt"
    report_path = model_root / "report.json"
    if not checkpoint_path.is_file() or not report_path.is_file():
        raise RuntimeError(f"missing frozen v12 artifact for {name}")
    if not json.loads(report_path.read_text()).get("promotion", {}).get("passes_clean_screen"):
        raise RuntimeError(f"{name} did not pass the frozen v12 clean screen")
    checkpoint_sha256 = runner.file_sha256(checkpoint_path)
    if checkpoint_sha256 != specification["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint checksum mismatch for {name}: {checkpoint_sha256}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != model_name or checkpoint.get("codec_normalization") != "jpeg_q96":
        raise RuntimeError(f"checkpoint contract mismatch for {name}")
    image_size = int(checkpoint["image_size"])
    mean = tuple(checkpoint["normalization_mean"])
    std = tuple(checkpoint["normalization_std"])
    model = timm.create_model(model_name, pretrained=False, num_classes=1, img_size=image_size)
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda().eval()

    output = model_root / "semantic-matched-modern-v6-gate"
    progress_path = output / "progress.json"
    signature = {
        "candidate": name,
        "checkpoint_sha256": checkpoint_sha256,
        "package_inventory_sha256": package["inventory_sha256"],
        "manifest_sha256": runner.file_sha256(manifest),
        "rows": len(rows),
        "conditions": [condition[0] for condition in workshop.conditions()],
        "codec_normalization": "jpeg_q96",
        "seed": SEED,
    }
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if progress.get("signature") != signature:
            raise RuntimeError(f"incompatible gate resume state for {name}")
    else:
        progress = {"completed": False, "signature": signature, "conditions": {}}

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    clean_predictions = None
    robust_predictions: list[dict] = []
    for index, (condition_name, image_transform) in enumerate(workshop.conditions()):
        prediction_path = output / f"{condition_name}_predictions.jsonl"
        if condition_name in progress["conditions"] and prediction_path.is_file():
            predictions = read_rows(prediction_path)
            if len(predictions) != len(rows):
                raise RuntimeError(f"invalid resume count for {name}/{condition_name}")
            print(json.dumps({"resumed": name, "condition": condition_name}), flush=True)
        else:
            torch.manual_seed(SEED + index)
            dataset = runner.ManifestDataset(
                manifest,
                workshop.condition_transform(image_size, mean, std, image_transform),
                rows=rows,
            )
            metrics, predictions = runner.evaluate(model, dataset)
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            prediction_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions))
            progress["conditions"][condition_name] = metrics
            workshop.atomic_json(progress_path, progress)
            print(json.dumps({"saved_condition": condition_name, "candidate": name, "auc": metrics["clean_auc"]}), flush=True)
        if condition_name == "clean":
            clean_predictions = predictions
        else:
            robust_predictions.extend(predictions)

    if clean_predictions is None:
        raise RuntimeError(f"clean predictions missing for {name}")
    clean_semantic = semantic_metrics(rows, clean_predictions)
    robust = workshop.pooled_metrics(rows, robust_predictions)
    worst_condition_auc = min(value["clean_auc"] for value in progress["conditions"].values())
    official_score = 0.5 * clean_semantic["overall_auc"] + 0.5 * robust["auc"]
    progress.update(
        {
            "completed": True,
            "gate_compliance": compliance,
            "identity_separation": separation,
            "clean_semantic_metrics": clean_semantic,
            "official_style": {
                "clean_auc": clean_semantic["overall_auc"],
                "pooled_robust_auc": robust["auc"],
                "score": official_score,
            },
            "pooled_robust_groups": robust["groups"],
            "worst_individual_condition_auc": worst_condition_auc,
            "frozen_gate_decision": gate_decision(clean_semantic, official_score, worst_condition_auc),
            "elapsed_seconds_this_process": time.time() - started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "boundary": "One-shot modern-generator audit only; no row was used for training, calibration or setting selection.",
        }
    )
    workshop.atomic_json(progress_path, progress)
    print("SEMANTIC_GATE_SUMMARY " + json.dumps(progress, sort_keys=True), flush=True)
    return progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    args = parser.parse_args()
    evaluate_candidate(args.candidate)


if __name__ == "__main__":
    main()
