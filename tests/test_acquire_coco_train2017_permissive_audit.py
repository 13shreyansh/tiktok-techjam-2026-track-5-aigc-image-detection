import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path("scripts").resolve()))
import acquire_coco_train2017_permissive_audit as audit  # noqa: E402


def test_exclusion_ids_are_collected_without_label_assumptions(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps({"coco_image_id": 7, "label": 0}) + "\n")
    second.write_text(
        json.dumps({"coco_image_id": 9}) + "\n" + json.dumps({"other": 1}) + "\n"
    )
    assert audit.read_excluded_ids([first, second]) == {7, 9}


def test_fresh_filter_preserves_ranked_order() -> None:
    rows = [{"id": 5}, {"id": 7}, {"id": 9}, {"id": 11}]
    assert audit.filter_fresh_rows(rows, {7}, 2) == [{"id": 5}, {"id": 9}]


def test_fresh_filter_fails_closed_when_pool_is_too_small() -> None:
    try:
        audit.filter_fresh_rows([{"id": 5}], {5}, 1)
    except RuntimeError as error:
        assert "fresh rows" in str(error)
    else:
        raise AssertionError("undersized fresh pool was accepted")
