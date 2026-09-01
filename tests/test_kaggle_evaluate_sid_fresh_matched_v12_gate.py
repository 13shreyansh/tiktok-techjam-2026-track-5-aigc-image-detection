import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kaggle_evaluate_sid_fresh_matched_v12_gate as gate  # noqa: E402


def valid_rows():
    rows = []
    for label in (0, 1):
        for index in range(284):
            digest = f"{label:01x}{index:063x}"
            source_digest = f"{label + 2:01x}{index:063x}"
            rows.append(
                {
                    "label": label,
                    "dataset": "SID_Set-fresh-validation-v12-gate",
                    "source_path_role": "SID_Set/validation-00001-of-00034",
                    "source_revision": gate.SOURCE_REVISION,
                    "source_license": "CC-BY-4.0",
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
        "439434c4e59b3dbbd4cbe98b9b94464f9e201a3e60cb4560dd87e11ff31f74b0"
    )
    assert gate.GATE_INVENTORY_SHA256 == (
        "19ca0c433aa4e5cb04f8e36262ff9ea430c382987bc0c5d535d3de42e8f71ca3"
    )
    assert gate.GATE_MANIFEST_SHA256 == (
        "092f981ce515ee2061b9f406dd34e358cf0dc4aac5076f65675095135dcb7a27"
    )
    assert gate.CANDIDATES["pe_core"]["checkpoint_sha256"].startswith("f37bd6b4")
    assert gate.CANDIDATES["dinov2_control"]["checkpoint_sha256"].startswith(
        "db07f30c"
    )


def test_gate_rows_require_balanced_unique_licensed_eval_only_contract():
    report = gate.validate_gate_rows(valid_rows())
    assert report["rows"] == 568
    assert report["training_allowed_rows"] == 0
    assert report["source_license"] == "CC-BY-4.0"


def test_gate_rows_reject_wrong_revision():
    rows = valid_rows()
    rows[0]["source_revision"] = "wrong"
    with pytest.raises(RuntimeError, match="source revision"):
        gate.validate_gate_rows(rows)


@pytest.mark.parametrize(
    ("clean", "worst", "expected", "interpretation"),
    [
        (0.81, 0.61, True, "useful_fresh_high_resolution_same_source_evidence_only"),
        (0.79, 0.90, False, "material_source_or_content_dependence_warning"),
        (0.64, 0.90, False, "reject_as_trusted_primary"),
        (0.90, 0.59, False, "useful_fresh_high_resolution_same_source_evidence_only"),
    ],
)
def test_decision_uses_frozen_clean_and_transformed_floors(
    clean, worst, expected, interpretation
):
    decision = gate.gate_decision(clean, worst)
    assert decision["passes_frozen_gate"] is expected
    assert decision["clean_interpretation"] == interpretation
