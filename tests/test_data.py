from pathlib import Path

from PIL import Image

import json

import pytest

from aigc_detector.data import (
    BinaryFolderDataset,
    BinaryManifestDataset,
    discover_manifest_records,
    source_balanced_weights,
)


def test_fake_is_positive_class(tmp_path: Path):
    for class_name in ("REAL", "FAKE"):
        folder = tmp_path / class_name
        folder.mkdir()
        Image.new("RGB", (8, 8)).save(folder / "sample.png")
    dataset = BinaryFolderDataset(tmp_path)
    labels = {Path(path).parent.name: label for _, label, path in dataset}
    assert labels == {"FAKE": 1.0, "REAL": 0.0}


def test_manifest_preserves_explicit_labels_and_relative_paths(tmp_path: Path):
    real = tmp_path / "camera.jpg"
    fake = tmp_path / "flux.png"
    Image.new("RGB", (8, 8)).save(real)
    Image.new("RGB", (8, 8)).save(fake)
    manifest = tmp_path / "split.jsonl"
    rows = [
        {"path": real.name, "label": 0, "real_source": "camera"},
        {"path": fake.name, "label": 1, "generator": "flux"},
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    dataset = BinaryManifestDataset(manifest)
    assert {Path(path).name: label for _, label, path in dataset} == {
        "camera.jpg": 0.0,
        "flux.png": 1.0,
    }
    records = discover_manifest_records(manifest)
    assert records[0]["real_source"] == "camera"
    assert records[1]["generator"] == "flux"


def test_manifest_rejects_duplicate_paths(tmp_path: Path):
    image = tmp_path / "same.png"
    Image.new("RGB", (8, 8)).save(image)
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text(
        json.dumps({"path": image.name, "label": 0}) + "\n"
        + json.dumps({"path": image.name, "label": 1}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate image path"):
        BinaryManifestDataset(manifest)


def test_source_balanced_weights_equalize_labels_and_named_groups(tmp_path: Path):
    rows = []
    specifications = [
        (0, "real_source", "camera-a"),
        (0, "real_source", "camera-a"),
        (0, "real_source", "camera-b"),
        (1, "generator", "generator-x"),
        (1, "generator", "generator-y"),
        (1, "generator", "generator-y"),
    ]
    for index, (label, key, group) in enumerate(specifications):
        image = tmp_path / f"{index}.png"
        Image.new("RGB", (8, 8)).save(image)
        rows.append({"path": image.name, "label": label, key: group})
    manifest = tmp_path / "balanced.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    dataset = BinaryManifestDataset(manifest)
    weights, report = source_balanced_weights(manifest, dataset.samples)
    by_label = {0: 0.0, 1: 0.0}
    by_group = {}
    records = {record["path"]: record for record in discover_manifest_records(manifest)}
    for (path, label), weight in zip(dataset.samples, weights):
        record = records[path]
        group = record.get("generator", record.get("real_source"))
        by_label[label] += weight
        by_group[(label, group)] = by_group.get((label, group), 0.0) + weight
    assert by_label == pytest.approx({0: 1.0, 1: 1.0})
    assert list(by_group.values()) == pytest.approx([0.5] * 4)
    assert report["groups_per_label"] == {"0": 2, "1": 2}
