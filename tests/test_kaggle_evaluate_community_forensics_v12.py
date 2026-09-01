import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kaggle_evaluate_community_forensics_v12 as audit  # noqa: E402


def test_validate_gate_rows_requires_balanced_audit_only_contract():
    rows = []
    for index in range(281):
        rows.append(
            {
                "label": 0,
                "path": f"../images/real-{index}.jpg",
                "image_sha256": f"{index:064x}",
                "training_allowed": False,
                "real_source": "real-source",
            }
        )
    for index in range(312):
        rows.append(
            {
                "label": 1,
                "path": f"../images/fake-{index}.jpg",
                "image_sha256": f"{index + 281:064x}",
                "training_allowed": False,
                "generator": "CommunityForensics-LatDiff-78-models",
                "generator_model": f"model-{index % 78}",
            }
        )
    report = audit.validate_gate_rows(rows)
    assert report["fake_model_variants"] == 78
    assert report["training_allowed_rows"] == 0
    rows[0].pop("training_allowed")
    assert audit.validate_gate_rows(rows)["training_allowed_rows"] == 0
    rows[0]["training_allowed"] = True
    with pytest.raises(RuntimeError, match="training_allowed"):
        audit.validate_gate_rows(rows)


def test_clean_metrics_reports_worst_named_model():
    rows = [
        {"label": 0, "generator_model": None},
        {"label": 0, "generator_model": None},
        {"label": 1, "generator_model": "strong"},
        {"label": 1, "generator_model": "weak"},
    ]
    predictions = [
        {"index": 0, "label": 0, "score": 0.1},
        {"index": 1, "label": 0, "score": 0.4},
        {"index": 2, "label": 1, "score": 0.9},
        {"index": 3, "label": 1, "score": 0.2},
    ]
    report = audit.clean_metrics(rows, predictions)
    assert report["overall_auc"] == 0.75
    assert report["by_fake_model"]["strong"] == 1.0
    assert report["by_fake_model"]["weak"] == 0.5
    assert report["worst_fake_model_auc"] == 0.5


def test_identity_separation_checks_v12_source_and_canonical_hashes(tmp_path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    train = [{"image_sha256": "a", "source_image_sha256": "b"}]
    evaluation = [{"image_sha256": "c", "source_image_sha256": "d"}]
    (manifests / "train.jsonl").write_text(json.dumps(train[0]) + "\n")
    (manifests / "eval_frozen.jsonl").write_text(json.dumps(evaluation[0]) + "\n")
    report = audit.validate_identity_separation([{"image_sha256": "e"}], tmp_path)
    assert report["train_identity_overlap"] == 0
    with pytest.raises(RuntimeError, match="overlaps"):
        audit.validate_identity_separation([{"image_sha256": "b"}], tmp_path)
