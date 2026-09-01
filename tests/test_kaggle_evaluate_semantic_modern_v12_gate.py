import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kaggle_evaluate_semantic_modern_v12_gate as gate  # noqa: E402


def valid_rows():
    rows = []
    generators = sorted(gate.EXPECTED_GENERATORS)
    for prompt in sorted(gate.EXPECTED_PROMPTS):
        for generator in generators:
            pair_id = f"{prompt}-{generator}"
            for label in (0, 1):
                token = f"{prompt}-{generator}-{label}"
                digest = (token.encode().hex() + "0" * 64)[:64]
                source = ("f" + token.encode().hex() + "0" * 64)[:64]
                rows.append(
                    {
                        "label": label,
                        "workflow_purpose": "semantic-matched-modern-audit",
                        "training_allowed": False,
                        "organizer_demo_row": False,
                        "canonicalization": gate.CANONICALIZATION,
                        "canonical_format": "JPEG",
                        "canonical_width": 336,
                        "canonical_height": 336,
                        "image_sha256": digest,
                        "source_image_sha256": source,
                        "semantic_prompt_id": prompt,
                        "paired_generator": generator,
                        "pair_id": pair_id,
                    }
                )
    return rows


def test_frozen_package_and_checkpoint_hashes_are_exact():
    assert gate.GATE_PACKAGE_SHA256 == "52f6749bed16015a1511e6cc6e9e7072d50350b3755148e1c3bea6d645288d69"
    assert gate.GATE_INVENTORY_SHA256 == "86da500fcbbfe76730940bbbf82d789d6773e90086b3f48ab0889adb29ad8496"
    assert gate.CANDIDATES["pe_core"]["checkpoint_sha256"].startswith("f37bd6b4")
    assert gate.CANDIDATES["dinov2_control"]["checkpoint_sha256"].startswith("db07f30c")


def test_working_copy_locator_preserves_hash_checks_and_restores_input(monkeypatch, tmp_path):
    staging = tmp_path / "semantic-gate-package"
    staging.mkdir()
    package = staging / gate.GATE_PACKAGE_NAME
    package.write_bytes(b"frozen gate")
    original_input = tmp_path / "mounted-input"
    observed = {}

    monkeypatch.setattr(gate, "GATE_STAGING_ROOT", staging)
    monkeypatch.setattr(gate.runner, "INPUT_ROOT", original_input)
    monkeypatch.setattr(
        gate.runner,
        "file_sha256",
        lambda path: gate.GATE_PACKAGE_SHA256 if path == package else "unexpected",
    )

    def fake_validate_package():
        observed["input_root"] = gate.runner.INPUT_ROOT
        observed["package_name"] = gate.runner.PACKAGE_NAME
        return tmp_path / "extracted", {"inventory_sha256": gate.GATE_INVENTORY_SHA256}

    monkeypatch.setattr(gate.runner, "validate_package", fake_validate_package)
    root, metadata = gate.validate_gate_package()

    assert observed == {
        "input_root": tmp_path,
        "package_name": gate.GATE_PACKAGE_NAME,
    }
    assert metadata["inventory_sha256"] == gate.GATE_INVENTORY_SHA256
    assert root == gate.GATE_WORK_ROOT
    assert gate.runner.INPUT_ROOT == original_input
    assert gate.runner.PACKAGE_NAME != gate.GATE_PACKAGE_NAME


def test_gate_contract_requires_balanced_prompts_generators_and_pairs():
    report = gate.validate_gate_rows(valid_rows())
    assert report["rows"] == 288
    assert report["pairs"] == 144
    assert report["training_allowed_rows"] == 0


def test_gate_contract_rejects_training_permission():
    rows = valid_rows()
    rows[0]["training_allowed"] = True
    with pytest.raises(RuntimeError, match="audit-use contract"):
        gate.validate_gate_rows(rows)


def test_semantic_metrics_are_pair_and_group_aware():
    rows = valid_rows()
    predictions = [
        {"index": index, "label": row["label"], "score": 0.9 if row["label"] else 0.1}
        for index, row in enumerate(rows)
    ]
    metrics = gate.semantic_metrics(rows, predictions)
    assert metrics["overall_auc"] == 1.0
    assert metrics["paired_accuracy"] == 1.0
    assert metrics["worst_prompt_auc"] == 1.0
    assert metrics["worst_generator_auc_diagnostic_only"] == 1.0


def test_frozen_decision_requires_every_floor():
    metrics = {"overall_auc": 0.71, "paired_accuracy": 0.66, "worst_prompt_auc": 0.56}
    assert gate.gate_decision(metrics, 0.66, 0.56)["passes_all_frozen_floors"]
    metrics["worst_prompt_auc"] = 0.54
    assert not gate.gate_decision(metrics, 0.66, 0.56)["passes_all_frozen_floors"]
