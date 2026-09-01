import ast
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
FRESH = ROOT / "scripts/kaggle_evaluate_v10_lowres_promotion.py"
INTERNAL = ROOT / "scripts/kaggle_evaluate_v10_internal_screen.py"


def constants(path: Path, names: set[str]) -> dict:
    tree = ast.parse(path.read_text())
    return {
        target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in names
    }


def test_fresh_gate_and_fixed_blend_are_checksum_frozen() -> None:
    values = constants(
        FRESH,
        {
            "PACKAGE_BYTES",
            "PACKAGE_SHA256",
            "PACKAGE_INVENTORY_SHA256",
            "EXPECTED_REAL_ROWS",
            "EXPECTED_FAKE_ROWS",
            "EXPECTED_GENERATORS",
            "IMAGE_SIZE",
            "PHYSICAL_BATCH_SIZE",
            "V6_WEIGHT",
            "V10_WEIGHT",
        },
    )
    assert values["PACKAGE_BYTES"] == 146_586_784
    assert values["PACKAGE_SHA256"].startswith("1dd7d460")
    assert values["PACKAGE_INVENTORY_SHA256"].startswith("b2357d44")
    assert values["EXPECTED_REAL_ROWS"] == 1_000
    assert values["EXPECTED_FAKE_ROWS"] == 144
    assert values["EXPECTED_GENERATORS"] == 18
    assert values["IMAGE_SIZE"] == 224
    assert values["PHYSICAL_BATCH_SIZE"] == 64
    assert values["V6_WEIGHT"] == 0.75
    assert values["V10_WEIGHT"] == 0.25


def test_fresh_promotion_requires_exact_candidate_and_prior_screen() -> None:
    source = FRESH.read_text()
    assert 'parser.add_argument("--v6-checkpoint", type=Path, required=True)' in source
    assert 'parser.add_argument("--v10-bytes", type=int, required=True)' in source
    assert 'parser.add_argument("--v10-sha256", required=True)' in source
    assert 'parser.add_argument("--internal-screen", type=Path, required=True)' in source
    assert '"passes_all_frozen_rules": all(checks.values())' in source
    assert '"forbidden_demo_resources_used": False' in source
    for prohibited in ("optimizer", "backward(", ".train("):
        assert prohibited not in source.lower()


def test_internal_screen_runs_before_fresh_gate_and_cannot_learn() -> None:
    source = INTERNAL.read_text()
    values = constants(
        INTERNAL,
        {"IMAGE_SIZE", "PHYSICAL_BATCH_SIZE", "V6_WEIGHT", "V10_WEIGHT"},
    )
    assert values == {
        "IMAGE_SIZE": 224,
        "V6_WEIGHT": 0.75,
        "V10_WEIGHT": 0.25,
        "PHYSICAL_BATCH_SIZE": 64,
    }
    assert '"fresh_promotion_gate_opened": False' in source
    assert '"passes_internal_screen": all(checks.values())' in source
    assert 'parser.add_argument("--v10-sha256", required=True)' in source
    for prohibited in ("optimizer", "backward(", ".train("):
        assert prohibited not in source.lower()


def test_internal_screen_accepts_only_consistent_duplicate_content() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("v10_internal_screen", INTERNAL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    duplicate = {
        "image_sha256": "same",
        "label": 0,
        "v6_score": 0.1,
        "v10_score": 0.2,
        "score": 0.125,
    }
    result = module.subset_by_manifest(
        [{**duplicate, "index": 0}, {**duplicate, "index": 1}],
        [{"image_sha256": "same", "label": 0}],
    )
    assert len(result) == 1
    assert result[0]["index"] == 0
