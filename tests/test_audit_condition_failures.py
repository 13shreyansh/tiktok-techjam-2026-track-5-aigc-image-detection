import json

import pytest

from scripts.audit_condition_failures import parse_input, summarize_evaluation


def test_summarize_evaluation_exposes_hidden_pair(tmp_path):
    def condition(auc, pair_auc):
        return {
            "auc": auc,
            "groups": {
                "fake_generators": {"fake-a": {"auc_against_all_reals": pair_auc}},
                "real_sources": {"real-a": {"auc_against_all_fakes": pair_auc}},
                "generator_real_source_pairs": {
                    "fake-a": {"real-a": {"auc": pair_auc}}
                },
            },
        }

    source = tmp_path / "eval.json"
    source.write_text(
        json.dumps(
                {
                    "checkpoint": "model.pt",
                    "model": "control-model",
                    "dataset_source": "manifest.jsonl",
                "evaluation": {
                    "official": {"official_score": 0.91},
                    "conditions": {
                        "clean": condition(0.95, 0.92),
                        "noise": condition(0.90, 0.44),
                    },
                },
            }
        )
    )

    result = summarize_evaluation("test", source)

    assert result["worst_overall_condition"]["condition"] == "noise"
    assert result["model"] == "control-model"
    assert result["worst_subgroup_pair_condition"]["worst_pair"] == {
        "fake_generator": "fake-a",
        "real_source": "real-a",
        "auc": 0.44,
    }
    assert result["conditions_ranked_by_auc"][0]["auc_delta_from_clean"] == pytest.approx(-0.05)


def test_parse_input_allows_equal_sign_in_path():
    name, path = parse_input("gate=folder=variant/eval.json")
    assert name == "gate"
    assert str(path) == "folder=variant/eval.json"
