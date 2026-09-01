import json

import pytest

from scripts import score_fixed_v12_blend as blend


def write_matrix(root, candidate, scores):
    root.mkdir()
    signature = {
        "candidate": candidate,
        "checkpoint_sha256": blend.EXPECTED[candidate],
        "conditions": list(blend.CONDITIONS),
    }
    (root / "progress.json").write_text(
        json.dumps({"completed": True, "signature": signature})
    )
    for condition in blend.CONDITIONS:
        rows = [
            {"index": index, "label": label, "image_sha256": f"sha-{index}", "score": score}
            for index, (label, score) in enumerate(scores)
        ]
        (root / f"{condition}_predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )


def test_equal_blend_scores_identical_rows(tmp_path):
    pe = tmp_path / "pe"
    dino = tmp_path / "dino"
    write_matrix(pe, "pe_core", [(0, 0.2), (0, 0.3), (1, 0.7), (1, 0.8)])
    write_matrix(dino, "dinov2_control", [(0, 0.1), (0, 0.4), (1, 0.6), (1, 0.9)])
    report = blend.score(pe, dino)
    assert report["clean_auc"] == 1.0
    assert report["pooled_robust_auc"] == 1.0
    assert report["official_style_score"] == 1.0
    assert report["rows_per_condition"] == 4


def test_equal_blend_rejects_identity_mismatch(tmp_path):
    pe = tmp_path / "pe"
    dino = tmp_path / "dino"
    write_matrix(pe, "pe_core", [(0, 0.2), (1, 0.8)])
    write_matrix(dino, "dinov2_control", [(0, 0.2), (1, 0.8)])
    path = dino / "clean_predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["image_sha256"] = "different"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(RuntimeError, match="identity mismatch"):
        blend.score(pe, dino)
