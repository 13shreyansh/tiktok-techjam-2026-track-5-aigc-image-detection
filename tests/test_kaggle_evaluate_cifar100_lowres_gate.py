import ast
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/kaggle_evaluate_cifar100_lowres_gate.py"
PLAN = ROOT / "CIFAR100_LOWRES_GATE_PLAN.json"


def test_gate_plan_is_frozen_diagnostic_only() -> None:
    plan = json.loads(PLAN.read_text())
    assert plan["status"] == "frozen_before_scoring"
    assert plan["authentic_gate"]["rows"] == 1000
    assert plan["authentic_gate"]["training_allowed"] is False
    assert plan["fake_reference"]["new_fake_evidence"] is False
    assert "Never train" in plan["decision_boundary"]
    assert "demo-only" in plan["forbidden_resources"]


def test_kaggle_evaluator_preserves_exact_contract_and_no_learning() -> None:
    source = SCRIPT.read_text()
    tree = ast.parse(source)
    assignments = {
        target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id
        in {
            "EXPECTED_ROWS",
            "IMAGE_SIZE",
            "PHYSICAL_BATCH_SIZE",
            "V6_SHA256",
            "V9_SHA256",
            "V6_WEIGHT",
            "V9_WEIGHT",
        }
    }
    assert assignments["EXPECTED_ROWS"] == 1000
    assert assignments["IMAGE_SIZE"] == 224
    assert assignments["PHYSICAL_BATCH_SIZE"] == 64
    assert assignments["V6_WEIGHT"] == 0.75
    assert assignments["V9_WEIGHT"] == 0.25
    assert assignments["V6_SHA256"].startswith("48ea5077")
    assert assignments["V9_SHA256"].startswith("dd6b26c7")
    assert "repeat(PHYSICAL_BATCH_SIZE - original_count" in source
    assert "torch.float16" in source
    assert "content-equivalent Kaggle extraction" in source
    assert "archive_reverified_in_runtime" in source
    assert 'checkpoint["image_size"]' in source
    for prohibited in ("optimizer", "backward(", ".train("):
        assert prohibited not in source.lower()
