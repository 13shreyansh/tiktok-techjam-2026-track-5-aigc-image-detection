#!/usr/bin/env python3
"""Run the candidate directory-to-JSON ensemble and compare saved scores."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import kaggle_evaluate_frontier_ensemble_promotion as promotion
import kaggle_evaluate_v8_promotion_gates as sealed


CANDIDATE_ROOT = Path("/kaggle/working/ensemble_candidate_v2")
RUNNER = CANDIDATE_ROOT / "src/aigc_detector/predict_ensemble.py"
RUNNER_SHA256 = "a1341afc3c62afa07f6d887c982394605c1fae5da427d486364863fab1b0d33b"
SOURCE_ARCHIVE_SHA256 = "9dec87af70455a797c76181a8faf0846baf932f0c2dcd4ab6cdb3bb57faca301"
INPUT_DIRECTORY = CANDIDATE_ROOT / "e2e_input_full_v2"
OUTPUT = CANDIDATE_ROOT / "e2e_predictions_v2.json"
REPORT = CANDIDATE_ROOT / "e2e_report_v2.json"
EXPECTED_ROWS = 576


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if file_sha256(RUNNER) != RUNNER_SHA256:
        raise RuntimeError("candidate runner checksum mismatch")
    sealed_root, _ = sealed.validate_package()
    manifest = sealed_root / sealed.MANIFESTS["qwen_prompt_holdout"]["path"]
    rows = promotion.read_jsonl(manifest)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"Qwen row mismatch: {len(rows)}")
    if INPUT_DIRECTORY.exists() or OUTPUT.exists() or REPORT.exists():
        raise RuntimeError("refusing to reuse an existing candidate E2E artifact")
    INPUT_DIRECTORY.mkdir()
    for index, row in enumerate(rows):
        source = (manifest.parent / row["path"]).resolve()
        (INPUT_DIRECTORY / f"{index:04d}{source.suffix.lower()}").symlink_to(
            source
        )

    v6 = promotion.selected_v6_path()
    v9 = promotion.V9_ROOT / promotion.MODEL_NAME / "model.pt"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{CANDIDATE_ROOT / 'src'}:/kaggle/working"
    command = [
        "/usr/bin/python3",
        "-m",
        "aigc_detector.predict_ensemble",
        str(INPUT_DIRECTORY),
        "--v6-checkpoint",
        str(v6),
        "--v9-checkpoint",
        str(v9),
        "--output",
        str(OUTPUT),
        "--device",
        "cuda:0",
    ]
    started = time.perf_counter()
    result = subprocess.run(
        command, env=environment, text=True, capture_output=True, check=False
    )
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "returncode": result.returncode,
                "elapsed_seconds": elapsed,
                "stdout": result.stdout,
                "stderr_tail": result.stderr[-2000:],
            }
        ),
        flush=True,
    )
    result.check_returncode()
    predictions = json.loads(OUTPUT.read_text())
    saved = promotion.read_jsonl(
        promotion.OUTPUT_ROOT / "qwen_prompt_holdout/clean_predictions.jsonl"
    )
    if len(predictions) != len(saved) or len(saved) != EXPECTED_ROWS:
        raise RuntimeError("candidate output count mismatch")
    drifts = [
        abs(float(observed["pred"]) - float(reference["score"]))
        for observed, reference in zip(predictions, saved)
    ]
    report = {
        "completed": True,
        "passes": max(drifts) == 0.0,
        "rows": len(rows),
        "runner_sha256": file_sha256(RUNNER),
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "manifest_sha256": file_sha256(manifest),
        "v6_checkpoint_sha256": file_sha256(v6),
        "v9_checkpoint_sha256": file_sha256(v9),
        "command": command,
        "returncode": result.returncode,
        "elapsed_seconds_including_checkpoint_hash_load_decode_and_inference": elapsed,
        "stdout": result.stdout.strip(),
        "output_sha256": file_sha256(OUTPUT),
        "max_absolute_score_drift_vs_saved_promotion": max(drifts),
        "mean_absolute_score_drift_vs_saved_promotion": sum(drifts) / len(drifts),
        "finite_probabilities_in_unit_interval": all(
            0.0 <= float(row["pred"]) <= 1.0 for row in predictions
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
