import io
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path("scripts").resolve()))
import acquire_coco_train2017_permissive as coco  # noqa: E402


def test_streaming_array_parser_reads_only_named_objects() -> None:
    payload = {"info": {"x": 1}, "images": [{"id": 1}, {"id": 2}], "annotations": [{"id": 9}]}
    rows = list(coco.iter_named_array(io.StringIO(json.dumps(payload)), "images"))
    assert rows == [{"id": 1}, {"id": 2}]


def test_only_commercial_compatible_license_ids_are_admitted() -> None:
    assert coco.ALLOWED_LICENSE_IDS == {4, 5, 7, 8}
    assert not ({1, 2, 3, 6} & coco.ALLOWED_LICENSE_IDS)


def test_val2017_locator_is_rejected() -> None:
    row = {
        "id": 7,
        "file_name": "000000000007.jpg",
        "coco_url": "http://images.cocodataset.org/val2017/000000000007.jpg",
        "license": 4,
    }
    try:
        coco.validate_train_row(row, set())
    except ValueError as error:
        assert "val2017" in str(error)
    else:
        raise AssertionError("val2017 row was accepted")


def test_selection_rank_is_deterministic_and_seeded() -> None:
    assert coco.rank(11, 5) == coco.rank(11, 5)
    assert coco.rank(11, 5) != coco.rank(12, 5)
