import json
from pathlib import Path


def test_v12_manifest_matches_sidecar_and_parameter_limit():
    manifest = json.loads(Path("V12_CHECKPOINT_MANIFEST.json").read_text())
    sidecar = {}
    for line in Path("V12_CHECKPOINTS.sha256").read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        sidecar[filename] = digest
    assert manifest["status"] == "verified_private_not_published"
    assert manifest["selected_runtime_mode"] == "pe_core"
    assert manifest["ensemble_total_parameters"] == 619_004_930
    assert manifest["ensemble_total_parameters"] < manifest["organizer_parameter_limit_exclusive"]
    assert manifest["combined_checkpoint_bytes"] == 2_476_225_646
    assert sidecar == {
        value["filename"]: value["sha256"]
        for value in manifest["checkpoints"].values()
    }
    assert all(value["distribution_url"] is None for value in manifest["checkpoints"].values())
    assert manifest["training_lineage"]["organizer_demo_rows"] == 0
    assert manifest["training_lineage"]["recorded_noncommercial_rows"] == 0
    selected_digest, selected_path = Path("SELECTED_CHECKPOINT.sha256").read_text().split()
    assert selected_path == manifest["checkpoints"]["pe_core"]["filename"]
    assert selected_digest == manifest["checkpoints"]["pe_core"]["sha256"]


def test_v12_runtime_result_preserves_claim_boundary():
    result = json.loads(Path("V12_RUNNABLE_CONTRACT_RESULT.json").read_text())
    assert result["blend"]["return_code"] == 0
    assert result["unchanged_repeat"]["output_byte_exact"] is True
    assert result["pe_core_fallback"]["return_code"] == 0
    assert result["input"]["organizer_demo_rows"] == 0
    assert "not an accuracy" in result["claim_boundary"]


def test_selected_default_result_uses_pe_without_mode_override():
    manifest = json.loads(Path("V12_CHECKPOINT_MANIFEST.json").read_text())
    result = json.loads(Path("V12_SELECTED_DEFAULT_RUN_RESULT.json").read_text())
    selected = manifest["checkpoints"][manifest["selected_runtime_mode"]]

    assert result["status"] == "selected_default_mps_contract_passed"
    assert result["return_code"] == 0
    assert result["mode_environment_override_present"] is False
    assert result["observed_mode"] == manifest["selected_runtime_mode"] == "pe_core"
    assert result["checkpoint_sha256"] == selected["sha256"]
    assert result["total_parameters"] == selected["parameters"]
    assert len(result["scores_in_filename_order"]) == result["images"] == 4
    assert all(0.0 <= score <= 1.0 for score in result["scores_in_filename_order"])
