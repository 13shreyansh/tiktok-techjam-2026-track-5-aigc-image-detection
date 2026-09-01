import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/evaluate_ntire_v12_arbitration_local.py"


def constants() -> dict:
    tree = ast.parse(SCRIPT.read_text())
    values = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def test_local_recovery_freezes_device_and_batch_contract():
    values = constants()
    assert values["GATE_MANIFEST_SHA256"] == (
        "dfd3f196106544d586a3eb32c22f94d213f0ddd0f642f07d5cfc9e1fb08e2bb6"
    )
    assert values["EXPECTED_ROWS"] == 1024
    assert values["BATCH_SIZE"] == 1
    assert values["WORKERS"] == 0


def test_local_recovery_checks_all_identity_forms_and_demo_terms():
    source = SCRIPT.read_text()
    assert 'row.get("sha256")' in source
    assert 'row.get("image_sha256")' in source
    assert 'row.get("source_image_sha256")' in source
    assert '"demo_only" in lowered' in source
    assert "sha256_file(image) != row[\"image_sha256\"]" in source


def test_local_recovery_plan_preserves_original_decision():
    plan = (ROOT / "NTIRE_V12_LOCAL_RECOVERY_PLAN.json").read_text()
    assert '"status": "frozen_before_any_local_v12_score"' in plan
    assert '"score_recovered_from_kaggle": false' in plan
    assert '"decision_policy": "unchanged from NTIRE_V12_FINAL_ARBITRATION_PLAN.json"' in plan
    recovery = (ROOT / "NTIRE_V12_MPS_BUFFER_RECOVERY.json").read_text()
    assert '"score_produced": false' in recovery
    assert '"batch_size": 1' in recovery
    assert '"data_loader_workers": 0' in recovery
