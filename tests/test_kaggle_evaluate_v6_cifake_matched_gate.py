import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kaggle_evaluate_v6_cifake_matched_gate as evaluator  # noqa: E402


def test_v6_identity_and_model_are_frozen():
    assert evaluator.MODEL_NAME == "vit_pe_core_large_patch14_336"
    assert evaluator.CHECKPOINT_SHA256 == (
        "48ea50773fbd1b7247fff25fde6f985183e29f2eb517b5ac0f6319c1fe38b644"
    )
    assert evaluator.CHECKPOINT_BYTES == 631645967


def test_workshop_matrix_is_exactly_twenty_conditions():
    names = [name for name, _ in evaluator.workshop.conditions()]
    assert len(names) == 20
    assert names[0] == "clean"
    assert names[-1] == "center_crop_80"
    assert "noise_sigma_0.10" in names
