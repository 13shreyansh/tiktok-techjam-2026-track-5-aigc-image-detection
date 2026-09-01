from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "audit_perceptual_overlap.py"
SPEC = importlib.util.spec_from_file_location("audit_perceptual_overlap", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_bk_tree_finds_close_values() -> None:
    tree = MODULE.BKTree()
    tree.add(0b0000, 0)
    tree.add(0b1111, 1)
    assert tree.query(0b0001, 1) == [0]


def test_audit_reports_cross_label_candidate() -> None:
    train = [MODULE.Fingerprint("train.jpg", 0, 0b0000, 0b0000)]
    evaluation = [MODULE.Fingerprint("eval.jpg", 1, 0b0001, 0b0010)]
    result = MODULE.audit(train, evaluation, 1, 1, 10)
    assert result["candidate_pairs"] == 1
    assert result["cross_label_candidate_pairs"] == 1
    assert result["examples"][0]["dhash_distance"] == 1
