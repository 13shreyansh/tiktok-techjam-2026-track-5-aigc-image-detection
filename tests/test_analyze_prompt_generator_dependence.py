import json

import pytest

from scripts.analyze_prompt_generator_dependence import analyze


def test_balanced_grid_separates_prompt_and_generator_effects(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    rows = []
    paths = []
    scores = []
    for prompt in (1, 2):
        for generator, generator_effect in (("a", 0.0), ("b", 0.2)):
            path = tmp_path / f"{prompt}-{generator}.png"
            rows.append(
                {
                    "label": 1,
                    "path": path.name,
                    "prompt_id": prompt,
                    "generator_model": generator,
                }
            )
            paths.append(str(path))
            scores.append(0.2 * prompt + generator_effect)
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    progress = tmp_path / "progress.json"
    real_paths = [str(tmp_path / "real-1.jpg"), str(tmp_path / "real-2.jpg")]
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
                        "paths": real_paths + paths,
                        "scores": [0.1, 0.15] + scores,
                        "labels": [0, 0] + [1] * len(scores),
                    }
                },
            }
        )
    )

    report = analyze(manifest, progress)

    assert report["grid"] == {
        "fake_images": 4,
        "prompts": 2,
        "generators": 2,
        "complete_balanced_grid": True,
    }
    shares = report["sum_of_squares_share"]
    assert shares["prompt"] == pytest.approx(0.5)
    assert shares["generator"] == pytest.approx(0.5)
    assert shares["unexplained_prompt_generator_interaction"] == pytest.approx(0.0)
    assert report["prompts_ranked_low_to_high"][0]["auc_against_all_reals"] == 1.0
