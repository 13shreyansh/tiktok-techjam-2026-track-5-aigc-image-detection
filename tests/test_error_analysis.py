import hashlib
import json
from pathlib import Path

from aigc_detector.error_analysis import (
    load_clean_progress_predictions,
    rank_errors,
    summarize_errors,
)


def test_rank_errors_returns_hardest_examples() -> None:
    paths = [str(Path(f"/tmp/{name}.jpg")) for name in ("r1", "r2", "f1", "f2")]
    metadata = {
        str(Path(paths[0]).resolve()): {"real_source": "real-a"},
        str(Path(paths[1]).resolve()): {"real_source": "real-b"},
        str(Path(paths[2]).resolve()): {"generator": "fake-a"},
        str(Path(paths[3]).resolve()): {"generator": "fake-b"},
    }
    ranked = rank_errors(
        labels=[0, 0, 1, 1],
        scores=[0.9, 0.2, 0.1, 0.8],
        paths=paths,
        metadata=metadata,
        limit=1,
    )
    assert ranked["highest_scoring_reals"][0]["real_source"] == "real-a"
    assert ranked["lowest_scoring_fakes"][0]["generator"] == "fake-a"
    assert ranked["false_positives_at_0.5"][0]["ai_probability"] == 0.9
    assert ranked["false_negatives_at_0.5"][0]["ai_probability"] == 0.1


def test_load_clean_progress_predictions_checks_manifest_signature(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text('{"path":"unused","label":0}\n')
    progress = tmp_path / "evaluation.progress.json"
    progress.write_text(
        json.dumps(
            {
                "signature": {
                    "dataset_source": str(manifest.resolve()),
                    "dataset_source_sha256": hashlib.sha256(
                        manifest.read_bytes()
                    ).hexdigest(),
                    "checkpoint_sha256": "abc",
                },
                "predictions": {
                    "clean": {
                        "labels": [0, 1],
                        "scores": [0.1, 0.9],
                        "paths": ["real.png", "fake.png"],
                    }
                },
            }
        )
    )
    labels, scores, paths, signature = load_clean_progress_predictions(
        progress, manifest
    )
    assert labels == [0.0, 1.0]
    assert scores == [0.1, 0.9]
    assert paths == ["real.png", "fake.png"]
    assert signature["checkpoint_sha256"] == "abc"


def test_summarize_errors_reports_source_and_generator_rates() -> None:
    paths = [str(Path(f"/tmp/{name}.jpg")) for name in ("r1", "r2", "f1", "f2")]
    metadata = {
        str(Path(paths[0]).resolve()): {"real_source": "real-a"},
        str(Path(paths[1]).resolve()): {"real_source": "real-a"},
        str(Path(paths[2]).resolve()): {"generator": "fake-a"},
        str(Path(paths[3]).resolve()): {"generator": "fake-b"},
    }
    summary = summarize_errors(
        labels=[0, 0, 1, 1],
        scores=[0.9, 0.2, 0.1, 0.8],
        paths=paths,
        metadata=metadata,
    )
    assert summary["false_positives"] == 1
    assert summary["false_negatives"] == 1
    assert summary["balanced_accuracy"] == 0.5
    assert summary["by_real_source"]["real-a"]["error_rate"] == 0.5
    assert summary["by_fake_generator"]["fake-a"]["error_rate"] == 1.0
    assert summary["by_fake_generator"]["fake-b"]["error_rate"] == 0.0
