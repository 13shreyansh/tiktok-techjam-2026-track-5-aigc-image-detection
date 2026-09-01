import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path("scripts").resolve()))
import acquire_qwen_image_bench_audit as acquire  # noqa: E402


def test_explicit_prompt_selection_is_sorted_and_disjoint(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"selected_prompt_ids": [9, 3, 5]}))
    assert acquire.load_prompt_selection(path, 3, {1, 2}) == [3, 5, 9]


def test_explicit_prompt_selection_rejects_prior_prompt(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"selected_prompt_ids": [3, 5]}))
    try:
        acquire.load_prompt_selection(path, 2, {5})
    except RuntimeError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("prior prompt ID was accepted")
