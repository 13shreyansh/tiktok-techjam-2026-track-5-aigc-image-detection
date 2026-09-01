import ast
from pathlib import Path


SCRIPT = Path("scripts/kaggle_evaluate_v11_quality_route_gate.py")


def assignments() -> dict:
    tree = ast.parse(SCRIPT.read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return values


def test_frozen_package_contract_is_exact():
    values = assignments()
    assert values["PACKAGE_NAME"] == "ntire-v11-quality-route-gate.zip"
    assert values["PACKAGE_BYTES"] == 608_677_127
    assert values["PACKAGE_SHA256"] == (
        "5c68565fbf6a02242af5481e4720dd435b0145ae09306ff4fa80e00c30eeb8c9"
    )
    assert values["EXPECTED_ROWS"] == 1_024


def test_promotion_script_contains_all_frozen_guards():
    source = SCRIPT.read_text()
    required = (
        "clean_auc_drop_at_most_0_002",
        "noise_auc_at_least_0_70",
        "noise_improvement_at_least_0_05",
        "clean_route_rate_at_most_0_05",
        "noise_route_rate_at_least_0_95",
        "noise_mean_score_not_inverted",
        '"organizer_demo_rows": 0',
    )
    for guard in required:
        assert guard in source


def test_gate_is_opened_before_checkpoint_loading():
    source = SCRIPT.read_text()
    assert source.index("manifest, rows, integrity = validate_gate()") < source.index(
        "paths = {name: screen.locate_checkpoint(name)"
    )
