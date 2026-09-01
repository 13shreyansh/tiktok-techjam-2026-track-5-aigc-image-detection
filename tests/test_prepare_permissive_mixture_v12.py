from pathlib import Path
import sys

sys.path.insert(0, str(Path("scripts").resolve()))
import prepare_permissive_mixture_v12 as v12  # noqa: E402


def test_v12_forbids_known_noncommercial_and_demo_sources() -> None:
    for name in ("afhq", "celebahq", "ffhq", "demo_only", "val2017"):
        assert name in v12.FORBIDDEN_PATH_TERMS


def test_v12_training_fake_counts_match_predeclared_total() -> None:
    fixed = 1500 + 287 + 800 + 800 + 576
    assert fixed + sum(v12.WILDFAKE_TRAIN_COUNTS.values()) == 6787


def test_v12_filters_forbidden_paths_before_deterministic_selection(tmp_path: Path) -> None:
    clean = tmp_path / "clean.png"
    forbidden = tmp_path / "afhq" / "forbidden.png"
    rows = [
        {"label": 1, "generator": "styleGAN", "_source": forbidden},
        {"label": 1, "generator": "styleGAN", "_source": clean},
    ]
    assert v12.v6_group(rows, 1, "generator", "styleGAN", 1, 7)[0]["_source"] == clean


def test_v12_evaluation_fake_counts_match_predeclared_total() -> None:
    assert sum(v12.EVAL_COUNTS.values()) + 144 == 1000


def test_v12_rank_is_deterministic() -> None:
    assert v12.stable_rank(4, "x", "y") == v12.stable_rank(4, "x", "y")
    assert v12.stable_rank(4, "x", "y") != v12.stable_rank(5, "x", "y")


def test_coco_row_receives_explicit_commercial_licence_metadata() -> None:
    row = v12.normalize_coco_row(
        {
            "path": "/tmp/train2017/example.jpg",
            "source_license_id": 4,
            "source_license_name": "Attribution License",
        }
    )
    assert row["dataset"] == "COCO-train2017-commercial-compatible"
    assert row["license_commercial_use_allowed"] is True
    assert "Attribution License" in row["source_license"]
