import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import summarize_v12_errors as error_summary


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_summary_keeps_auc_separate_from_illustrative_threshold(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {"label": 0, "image_sha256": "r0", "source_filename": "real-a.jpg"},
        {"label": 0, "image_sha256": "r1", "source_filename": "real-b.jpg"},
        {"label": 1, "image_sha256": "f0", "source_filename": "fake-a.jpg"},
        {"label": 1, "image_sha256": "f1", "source_filename": "fake-b.jpg"},
    ]
    scored = [
        {"index": 0, "label": 0, "score": 0.1, "image_sha256": "r0"},
        {"index": 1, "label": 0, "score": 0.8, "image_sha256": "r1"},
        {"index": 2, "label": 1, "score": 0.4, "image_sha256": "f0"},
        {"index": 3, "label": 1, "score": 0.9, "image_sha256": "f1"},
    ]
    write_jsonl(manifest, rows)
    write_jsonl(predictions, scored)

    result = error_summary.summarize(manifest, predictions, 0.5)

    assert result["clean_roc_auc"] == pytest.approx(0.75)
    assert result["threshold_summary"]["false_positives"] == 1
    assert result["threshold_summary"]["false_negatives"] == 1
    assert result["highest_scoring_authentic"][0]["source_filename"] == "real-b.jpg"
    assert "not selected" in result["boundary"]


def test_summary_rejects_identity_drift(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        manifest,
        [
            {"label": 0, "image_sha256": "r0"},
            {"label": 1, "image_sha256": "f0"},
        ],
    )
    write_jsonl(
        predictions,
        [
            {"index": 0, "label": 0, "score": 0.1, "image_sha256": "changed"},
            {"index": 1, "label": 1, "score": 0.9, "image_sha256": "f0"},
        ],
    )

    with pytest.raises(RuntimeError, match="identity mismatch"):
        error_summary.summarize(manifest, predictions, 0.5)
