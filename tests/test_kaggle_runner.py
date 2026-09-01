import json
import sys
from pathlib import Path

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kaggle_train_v3 as runner  # noqa: E402
from kaggle_train_v3 import ManifestDataset, filter_evaluation_rows, select_evaluation_from_predictions  # noqa: E402


def test_manifest_dataset_applies_transform_exactly_once(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), color=(12, 34, 56)).save(image_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"path": "sample.png", "label": 1}) + "\n")
    calls = []

    def transform(image):
        calls.append(image.size)
        return image

    dataset = ManifestDataset(manifest, transform)
    image, label, index = dataset[0]

    assert image.size == (8, 8)
    assert label == 1
    assert index == 0
    assert calls == [(8, 8)]


def test_package_discovery_does_not_walk_unrelated_dataset_trees(tmp_path, monkeypatch):
    unrelated = tmp_path / "unrelated" / "nested"
    unrelated.mkdir(parents=True)
    (unrelated / runner.PACKAGE_NAME).write_bytes(b"must-not-be-found")
    monkeypatch.setattr(runner, "INPUT_ROOT", tmp_path)

    try:
        runner.validate_package()
    except RuntimeError as error:
        assert "expected one extracted dataset" in str(error)
    else:
        raise AssertionError("recursive package discovery unexpectedly occurred")


def test_mounted_root_files_supports_current_and_legacy_kaggle_layouts(
    tmp_path, monkeypatch
):
    current = tmp_path / "datasets" / "owner" / "dataset" / "package.json"
    legacy = tmp_path / "dataset-old" / "package.json"
    current.parent.mkdir(parents=True)
    legacy.parent.mkdir()
    current.write_text("{}")
    legacy.write_text("{}")
    monkeypatch.setattr(runner, "INPUT_ROOT", tmp_path)

    assert runner.mounted_root_files("package.json") == [legacy, current]


def test_content_gate_is_derived_from_selection_predictions():
    predictions = [
        {"image_sha256": "real-a", "label": 0, "score": 0.1, "real_source": "camera"},
        {"image_sha256": "real-b", "label": 0, "score": 0.2, "real_source": "web"},
        {"image_sha256": "fake-a", "label": 1, "score": 0.8, "generator": "gen-a"},
        {"image_sha256": "fake-b", "label": 1, "score": 0.9, "generator": "gen-b"},
    ]
    content_rows = [
        {"image_sha256": "real-b", "label": 0, "real_source": "web"},
        {"image_sha256": "fake-b", "label": 1, "generator": "gen-b"},
    ]

    result, subset = select_evaluation_from_predictions(predictions, content_rows)

    assert result["count"] == 2
    assert result["clean_auc"] == 1.0
    assert [row["image_sha256"] for row in subset] == ["real-b", "fake-b"]


def test_content_gate_rejects_missing_selection_image():
    predictions = [
        {"image_sha256": "real-a", "label": 0, "score": 0.1, "real_source": "camera"},
        {"image_sha256": "fake-a", "label": 1, "score": 0.9, "generator": "gen-a"},
    ]
    content_rows = [
        {"image_sha256": "missing", "label": 0, "real_source": "other"},
        {"image_sha256": "fake-a", "label": 1, "generator": "gen-a"},
    ]

    try:
        select_evaluation_from_predictions(predictions, content_rows)
    except RuntimeError as error:
        assert "absent from selection predictions" in str(error)
    else:
        raise AssertionError("missing content image was not rejected")


def test_content_gate_accepts_same_label_same_score_content_duplicates():
    predictions = [
        {"image_sha256": "real", "label": 0, "score": 0.1},
        {"image_sha256": "duplicate", "label": 1, "score": 0.8, "img_id": "a"},
        {"image_sha256": "duplicate", "label": 1, "score": 0.8, "img_id": "b"},
    ]
    content_rows = [
        {"image_sha256": "real", "label": 0, "real_source": "camera"},
        {"image_sha256": "duplicate", "label": 1, "generator": "gen", "img_id": "a"},
        {"image_sha256": "duplicate", "label": 1, "generator": "gen", "img_id": "b"},
    ]
    result, subset = select_evaluation_from_predictions(predictions, content_rows)
    assert result["count"] == 3
    assert result["clean_auc"] == 1.0
    assert [row["img_id"] for row in subset if row["label"] == 1] == ["a", "b"]


def test_content_gate_rejects_conflicting_duplicate_scores():
    predictions = [
        {"image_sha256": "real", "label": 0, "score": 0.1},
        {"image_sha256": "duplicate", "label": 1, "score": 0.7},
        {"image_sha256": "duplicate", "label": 1, "score": 0.8},
    ]
    try:
        select_evaluation_from_predictions(predictions, [])
    except RuntimeError as error:
        assert "conflicting scores" in str(error)
    else:
        raise AssertionError("conflicting duplicate scores were not rejected")


def test_filter_evaluation_rows_requires_and_removes_hashes():
    rows = [
        {"image_sha256": "keep", "label": 0},
        {"image_sha256": "remove", "label": 0},
    ]
    assert filter_evaluation_rows(rows, {"remove"}) == [rows[0]]
    try:
        filter_evaluation_rows(rows, {"missing"})
    except RuntimeError as error:
        assert "absent" in str(error)
    else:
        raise AssertionError("missing exclusion hash was not rejected")
