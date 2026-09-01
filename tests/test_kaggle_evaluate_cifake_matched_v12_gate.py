import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kaggle_evaluate_cifake_matched_v12_gate as gate  # noqa: E402


def valid_rows():
    rows = []
    for label in (0, 1):
        for index in range(1000):
            digest = f"{label:01x}{index:063x}"
            source_digest = f"{label + 2:01x}{index:063x}"
            rows.append(
                {
                    "label": label,
                    "dataset": "CIFAKE-matched-source-v12-gate",
                    "source_path_role": "CIFAKE/test",
                    "source_license": "MIT",
                    "license_commercial_use_allowed": True,
                    "training_allowed": False,
                    "evaluation_only": True,
                    "organizer_demo_row": False,
                    "canonicalization": gate.CANONICALIZATION,
                    "canonical_format": "JPEG",
                    "canonical_width": 336,
                    "canonical_height": 336,
                    "image_sha256": digest,
                    "source_image_sha256": source_digest,
                }
            )
    return rows


def test_frozen_gate_constants_and_candidates_are_exact():
    assert gate.GATE_PACKAGE_SHA256 == (
        "b91363fef08bceb3c72f86ca4e5d4fce8b0c0a530d79e56b431dfa8a0087d383"
    )
    assert gate.GATE_INVENTORY_SHA256 == (
        "1b62ff5df23538879a4e922a68648fcee73b73d39fc359b6db71804e69c14f5c"
    )
    assert gate.CANDIDATES["pe_core"]["checkpoint_sha256"].startswith("f37bd6b4")
    assert gate.CANDIDATES["dinov2_control"]["checkpoint_sha256"].startswith(
        "db07f30c"
    )


def test_gate_rows_require_same_dataset_balanced_unique_eval_only_contract():
    report = gate.validate_gate_rows(valid_rows())
    assert report["rows"] == 2000
    assert report["training_allowed_rows"] == 0


def test_gate_rows_reject_training_permission():
    rows = valid_rows()
    rows[0]["training_allowed"] = True
    with pytest.raises(RuntimeError, match="gate-use contract"):
        gate.validate_gate_rows(rows)


@pytest.mark.parametrize(
    ("clean", "worst", "expected", "interpretation"),
    [
        (0.81, 0.61, True, "useful_within_source_evidence_only"),
        (0.79, 0.90, False, "material_source_or_content_dependence_warning"),
        (0.64, 0.90, False, "reject_as_trusted_primary"),
        (0.90, 0.59, False, "useful_within_source_evidence_only"),
    ],
)
def test_decision_uses_frozen_clean_and_transformed_floors(
    clean, worst, expected, interpretation
):
    decision = gate.gate_decision(clean, worst)
    assert decision["passes_frozen_gate"] is expected
    assert decision["clean_interpretation"] == interpretation
