import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path("scripts").resolve()))
import select_content_matchable_qwen_prompts as select  # noqa: E402


def test_diverse_selection_is_deterministic_and_cluster_balanced() -> None:
    rng = np.random.default_rng(7)
    prompt_ids = list(range(1, 33))
    features = rng.normal(size=(32, 8)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    matchability = np.linspace(0.1, 0.9, 32)
    first = select.select_diverse_prompts(
        prompt_ids, features, matchability, shortlist_size=24, selected_count=4
    )
    second = select.select_diverse_prompts(
        prompt_ids, features, matchability, shortlist_size=24, selected_count=4
    )
    assert first == second
    selected_ids, clusters, shortlist = first
    assert len(selected_ids) == len(set(selected_ids)) == 4
    assert set(clusters.values()) == {0, 1, 2, 3}
    assert len(shortlist) == 24


def test_diverse_selection_fails_on_short_pool() -> None:
    try:
        select.select_diverse_prompts(
            [1, 2], np.eye(2), np.array([0.1, 0.2]), shortlist_size=2, selected_count=3
        )
    except RuntimeError as error:
        assert "insufficient" in str(error)
    else:
        raise AssertionError("short prompt pool was accepted")
