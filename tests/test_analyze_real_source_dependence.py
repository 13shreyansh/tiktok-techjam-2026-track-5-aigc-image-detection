import json

import pytest

from scripts.analyze_real_source_dependence import analyze


def test_source_variance_and_pair_auc_are_computed_from_aligned_paths(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"label": 0, "path": "r1.jpg", "real_source": "easy"},
        {"label": 0, "path": "r2.jpg", "real_source": "easy"},
        {"label": 0, "path": "r3.jpg", "real_source": "hard"},
        {"label": 0, "path": "r4.jpg", "real_source": "hard"},
        {"label": 1, "path": "f1.png"},
        {"label": 1, "path": "f2.png"},
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    paths = [str(tmp_path / row["path"]) for row in rows]
    progress = tmp_path / "progress.json"
    progress.write_text(
        json.dumps(
            {
                "signature": {
                    "checkpoint": "model.pt",
                    "checkpoint_sha256": "abc",
                    "model": "model",
                    "preprocess_mode": "stretch",
                    "codec_normalization": "jpeg_q96",
                    "inference_policy": "reference",
                },
                "predictions": {
                    "clean": {
                        "paths": paths,
                        "scores": [0.1, 0.2, 0.7, 0.8, 0.6, 0.9],
                        "labels": [0, 0, 0, 0, 1, 1],
                    }
                },
            }
        )
    )

    report = analyze(manifest, progress, permutations=99, seed=7)

    assert report["rows"] == {"real": 4, "fake": 2, "real_sources": 2}
    assert report["real_source_score_variance_share"] == pytest.approx(0.9729729730)
    ranked = report["real_sources_ranked_high_to_low_ai_score"]
    assert ranked[0]["real_source"] == "hard"
    assert ranked[0]["auc_against_all_fakes"] == pytest.approx(0.5)
    assert ranked[1]["auc_against_all_fakes"] == 1.0
