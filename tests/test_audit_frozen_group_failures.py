import json

import pytest

from scripts.audit_frozen_group_failures import audit_gate


def test_audit_gate_ranks_pairs_and_noise_drops(tmp_path):
    clean = [
        {"image_sha256": "r1", "label": 0, "real_source": "real-a", "v6_score": 0.2, "score": 0.1},
        {"image_sha256": "r2", "label": 0, "real_source": "real-b", "v6_score": 0.4, "score": 0.3},
        {"image_sha256": "f1", "label": 1, "generator": "gen-a", "v6_score": 0.8, "score": 0.9},
        {"image_sha256": "f2", "label": 1, "generator_model": "gen-b", "v6_score": 0.6, "score": 0.7},
    ]
    noise = [
        {**clean[0], "v6_score": 0.3, "score": 0.4},
        {**clean[1], "v6_score": 0.5, "score": 0.6},
        {**clean[2], "v6_score": 0.7, "score": 0.8},
        {**clean[3], "v6_score": 0.4, "score": 0.5},
    ]
    clean_path = tmp_path / "clean.jsonl"
    noise_path = tmp_path / "noise.jsonl"
    clean_path.write_text("".join(json.dumps(row) + "\n" for row in clean))
    noise_path.write_text("".join(json.dumps(row) + "\n" for row in noise))

    result = audit_gate(clean_path, noise_path)

    assert result["rows"] == 4
    assert result["clean"]["overall"]["blend_auc"] == pytest.approx(1.0)
    assert result["noise_sigma_0.10"]["overall"]["blend_auc"] == pytest.approx(0.75)
    assert len(result["worst_clean_blend_pairs"]) == 4


def test_audit_gate_rejects_mismatched_rows(tmp_path):
    clean_path = tmp_path / "clean.jsonl"
    noise_path = tmp_path / "noise.jsonl"
    clean_path.write_text(json.dumps({"image_sha256": "a", "label": 0}) + "\n")
    noise_path.write_text(json.dumps({"image_sha256": "b", "label": 0}) + "\n")

    with pytest.raises(ValueError, match="identical and ordered"):
        audit_gate(clean_path, noise_path)
