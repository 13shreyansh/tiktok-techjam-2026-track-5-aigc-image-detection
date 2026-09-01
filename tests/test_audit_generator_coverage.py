import json

from scripts.audit_generator_coverage import audit


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_exact_generator_and_raw_family_coverage_are_separate(tmp_path):
    train = tmp_path / "train.jsonl"
    gate = tmp_path / "gate.jsonl"
    write_rows(
        train,
        [
            {"label": 1, "generator": "old", "family": "latent-diffusion"},
            {"label": 0, "real_source": "real"},
        ],
    )
    write_rows(
        gate,
        [
            {"label": 1, "generator_model": "new", "family": "LatDiff"},
            {"label": 0, "real_source": "real"},
        ],
    )

    report = audit({"v6": train}, {"external": gate})
    comparison = report["comparisons"]["v6"]["external"]

    assert comparison["exact_generator_names_absent_from_training"] == ["new"]
    assert comparison["exact_generator_name_holdout_fraction"] == 1.0
    assert comparison["raw_family_labels_absent_from_training"] == ["LatDiff"]
    assert "canonical" in report["interpretation"][1]
