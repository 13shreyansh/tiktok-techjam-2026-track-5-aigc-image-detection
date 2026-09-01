import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/kaggle_evaluate_ntire_v12_arbitration.py"


def constants() -> dict:
    tree = ast.parse(SCRIPT.read_text())
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, (ast.Constant, ast.Dict, ast.Tuple))
    }


def test_gate_identity_and_size_are_frozen():
    values = constants()
    assert values["GATE_INVENTORY_SHA256"] == (
        "7b9801315bfef184820bf6eef216fd6d11e8dcaf1361387bb6fb9a536db665b6"
    )
    assert values["GATE_MANIFEST_SHA256"] == (
        "cab0d1347bcd78ba2fa8dd09caa0025a181c929ad1ce20fd245af8fd27e8d329"
    )
    assert values["EXPECTED_ROWS"] == 1024
    assert values["EXPECTED_PER_LABEL"] == 512


def test_evaluator_fails_closed_on_overlap_and_demo_terms():
    source = SCRIPT.read_text()
    assert 'raise RuntimeError(f"NTIRE overlaps v12 {role} identities: {overlap}")' in source
    assert '"demo_only" in lowered' in source
    assert 'row.get("training_allowed") is True' in source
    assert "for value in (row.get(\"image_sha256\"), row.get(\"source_image_sha256\"))" in source
    assert "os.replace(temporary, staged)" in source


def test_plan_freezes_default_decision_before_score():
    plan = (ROOT / "NTIRE_V12_FINAL_ARBITRATION_PLAN.json").read_text()
    assert '"status": "frozen_before_any_v12_score_on_this_gate"' in plan
    assert '"pe_official_minus_blend_official_at_least": 0.02' in plan
    assert '"pe_worst_condition_minus_blend_worst_condition_at_least": -0.03' in plan
    assert '"otherwise": "retain the fixed equal blend as default"' in plan
