import json
from pathlib import Path

import torch
from PIL import Image

from aigc_detector.evaluation import evaluate_conditions, grouped_metrics


def test_grouped_metrics_exposes_generator_and_real_source_failures(tmp_path: Path):
    paths = [str(tmp_path / name) for name in ("r1", "r2", "f1", "f2")]
    metadata = {
        str(Path(paths[0]).resolve()): {"label": 0, "real_source": "camera-a"},
        str(Path(paths[1]).resolve()): {"label": 0, "real_source": "camera-b"},
        str(Path(paths[2]).resolve()): {"label": 1, "generator": "gan"},
        str(Path(paths[3]).resolve()): {"label": 1, "generator": "diffusion"},
    }
    result = grouped_metrics(
        labels=[0.0, 0.0, 1.0, 1.0],
        scores=[0.1, 0.8, 0.9, 0.2],
        paths=paths,
        metadata=metadata,
    )
    assert result["fake_generators"]["gan"]["true_positive_rate_at_0.5"] == 1.0
    assert result["fake_generators"]["diffusion"]["true_positive_rate_at_0.5"] == 0.0
    assert result["real_sources"]["camera-a"]["true_negative_rate_at_0.5"] == 1.0
    assert result["real_sources"]["camera-b"]["true_negative_rate_at_0.5"] == 0.0
    assert result["worst_fake_generator_auc"] == 0.5
    assert result["worst_real_source_auc"] == 0.5
    assert result["generator_real_source_pairs"]["gan"]["camera-a"]["auc"] == 1.0
    assert result["generator_real_source_pairs"]["diffusion"]["camera-b"]["auc"] == 0.0
    assert result["worst_generator_real_source_pair_auc"] == 0.0


def test_evaluation_resume_reuses_saved_condition_predictions(tmp_path: Path):
    real_path = tmp_path / "real.png"
    fake_path = tmp_path / "fake.png"
    Image.new("RGB", (8, 8), color="black").save(real_path)
    Image.new("RGB", (8, 8), color="white").save(fake_path)
    manifest = tmp_path / "eval.jsonl"
    manifest.write_text(
        "".join(
            [
                json.dumps(
                    {"path": str(real_path), "label": 0, "real_source": "camera"}
                )
                + "\n",
                json.dumps(
                    {"path": str(fake_path), "label": 1, "generator": "heldout"}
                )
                + "\n",
            ]
        )
    )
    callbacks: list[str] = []
    result = evaluate_conditions(
        model=torch.nn.Identity(),
        dataset_root=manifest,
        device=torch.device("cpu"),
        image_size=8,
        batch_size=2,
        workers=0,
        max_per_class=None,
        seed=20260829,
        robust=False,
        completed_predictions={
            "clean": {
                "labels": [0.0, 1.0],
                "scores": [0.1, 0.9],
                "paths": [str(real_path), str(fake_path)],
            }
        },
        prediction_callback=lambda name, _: callbacks.append(name),
    )
    assert result["clean_auc"] == 1.0
    assert result["conditions"]["clean"]["groups"][
        "worst_generator_real_source_pair_auc"
    ] == 1.0
    assert callbacks == []
