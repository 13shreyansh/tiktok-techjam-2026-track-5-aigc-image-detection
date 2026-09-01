import json
from pathlib import Path


def test_final_live_rehearsal_matches_selected_checkpoint_and_default_output():
    rehearsal = json.loads(Path("FINAL_LIVE_REHEARSAL_RESULT.json").read_text())
    selected = Path("SELECTED_CHECKPOINT.sha256").read_text().split()[0]
    frozen = json.loads(Path("V12_SELECTED_DEFAULT_RUN_RESULT.json").read_text())

    assert rehearsal["selected_checkpoint_sha256"] == selected
    assert rehearsal["checkpoint_hash_verification"]["return_code"] == 0
    assert rehearsal["managed_sandbox_attempt"]["inference_started"] is False
    run = rehearsal["approved_device_run"]
    assert run["return_code"] == 0
    assert run["mode"] == "pe_core"
    assert run["device"] == "mps"
    assert run["total_parameters"] == 315_776_001
    assert run["output_sha256"] == frozen["output_sha256"]
    assert run["scores_in_filename_order"] == frozen["scores_in_filename_order"]
