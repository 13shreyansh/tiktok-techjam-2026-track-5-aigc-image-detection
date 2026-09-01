import json
from pathlib import Path

from scripts.compare_selection_reports import compare


def write_local_report(path: Path, clean: float, worst_pair: float) -> None:
    path.write_text(
        json.dumps(
            {
                "arguments": {"model": path.stem},
                "history": [
                    {
                        "evaluation": {
                            "conditions": {
                                "clean": {
                                    "auc": clean,
                                    "groups": {
                                        "worst_fake_generator_auc": 0.8,
                                        "worst_real_source_auc": 0.75,
                                        "worst_generator_real_source_pair_auc": worst_pair,
                                    },
                                }
                            }
                        }
                    }
                ],
            }
        )
    )


def test_comparison_vetoes_below_chance_subgroup(tmp_path: Path):
    collapsed = tmp_path / "collapsed.json"
    broad = tmp_path / "broad.json"
    write_local_report(collapsed, clean=0.99, worst_pair=0.49)
    write_local_report(broad, clean=0.85, worst_pair=0.65)
    payload = compare([collapsed, broad])
    assert payload["provisional_leader"] == str(broad)
    assert payload["candidates"][0]["below_chance_subgroup_veto"] is False
    assert payload["candidates"][1]["below_chance_subgroup_veto"] is True


def test_comparison_has_no_leader_when_every_candidate_is_vetoed(tmp_path: Path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_local_report(first, clean=0.99, worst_pair=0.49)
    write_local_report(second, clean=0.90, worst_pair=0.40)
    assert compare([first, second])["provisional_leader"] is None
