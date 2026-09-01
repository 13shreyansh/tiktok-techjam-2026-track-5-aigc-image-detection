import json
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path("scripts").resolve()))
import prepare_semantic_matched_modern_gate as semantic  # noqa: E402


def test_identity_collection_includes_source_and_canonical_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text(
        json.dumps(
            {"sha256": "canonical", "source_image_sha256": "source", "label": 0}
        )
        + "\n"
        + json.dumps({"image_sha256": "raw", "label": 1})
        + "\n"
    )
    assert semantic.collect_excluded_identities([manifest]) == {
        "canonical",
        "source",
        "raw",
    }


def test_global_assignment_is_balanced_and_unique(monkeypatch) -> None:
    monkeypatch.setattr(semantic, "EXPECTED_PROMPTS", 2)
    monkeypatch.setattr(semantic, "EXPECTED_GENERATORS", 2)
    monkeypatch.setattr(semantic, "REAL_ROWS_PER_PROMPT", 2)
    rows = [
        {"prompt_id": 1},
        {"prompt_id": 1},
        {"prompt_id": 2},
        {"prompt_id": 2},
    ]
    fake = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    real = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.7, 0.7]]
    )
    real /= np.linalg.norm(real, axis=1, keepdims=True)
    assigned, similarity = semantic.assign_unique_reals(rows, fake, real)
    selected = [index for values in assigned.values() for index, _ in values]
    assert sorted(assigned) == [1, 2]
    assert all(len(values) == 2 for values in assigned.values())
    assert len(selected) == len(set(selected)) == 4
    assert similarity.shape == (2, 5)


def test_individual_assignment_matches_each_fake_without_reusing_real() -> None:
    fake = np.array([[1.0, 0.0], [0.0, 1.0]])
    real = np.array([[0.9, 0.1], [0.1, 0.9], [0.7, 0.7]])
    real /= np.linalg.norm(real, axis=1, keepdims=True)
    assigned, similarity = semantic.assign_unique_reals_individually(fake, real)
    assert [index for index, _ in assigned] == [0, 1]
    assert len({index for index, _ in assigned}) == 2
    assert similarity.shape == (2, 3)
