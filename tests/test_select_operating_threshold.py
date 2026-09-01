from pathlib import Path

from scripts.select_operating_threshold import select_threshold, threshold_metrics


def _example():
    paths = [str(Path(f"/tmp/{name}.jpg")) for name in ("r1", "r2", "f1", "f2")]
    metadata = {
        str(Path(paths[0]).resolve()): {"real_source": "a"},
        str(Path(paths[1]).resolve()): {"real_source": "b"},
        str(Path(paths[2]).resolve()): {"generator": "x"},
        str(Path(paths[3]).resolve()): {"generator": "y"},
    }
    return [0, 0, 1, 1], [0.2, 0.4, 0.6, 0.8], paths, metadata


def test_threshold_metrics_reports_each_source_and_generator() -> None:
    labels, scores, paths, metadata = _example()
    metrics = threshold_metrics(labels, scores, paths, metadata, 0.5)
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["minimum_group_recall"] == 1.0
    assert set(metrics["group_recall"]) == {
        "real:a",
        "real:b",
        "fake:x",
        "fake:y",
    }


def test_select_threshold_maximizes_weakest_group() -> None:
    labels, scores, paths, metadata = _example()
    selected = select_threshold(labels, scores, paths, metadata)
    assert selected["selected"]["minimum_group_recall"] == 1.0
    assert 0.4 < selected["selected"]["threshold"] <= 0.6
    assert selected["default_0.5"]["balanced_accuracy"] == 1.0
