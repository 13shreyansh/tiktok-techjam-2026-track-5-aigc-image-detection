import sys
from pathlib import Path


sys.path.insert(0, str(Path("scripts").resolve()))

import prepare_demo_rehearsal as demo  # noqa: E402


def test_demo_rank_is_deterministic_and_role_specific() -> None:
    digest = "a" * 64
    assert demo.rank("real", digest) == demo.rank("real", digest)
    assert demo.rank("real", digest) != demo.rank("fake", digest)


def test_demo_source_excludes_organizer_demo_and_model_selection() -> None:
    text = Path("scripts/prepare_demo_rehearsal.py").read_text()
    assert '"demo_only" in str(source).lower()' in text
    assert '"model_selection_or_calibration": False' in text
    assert "lowest role-specific SHA-256 rank before inference" in text
