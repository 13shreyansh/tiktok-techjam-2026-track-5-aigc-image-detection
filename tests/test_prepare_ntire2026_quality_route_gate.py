from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_ntire2026_quality_route_gate as gate  # noqa: E402


def test_rank_is_deterministic_and_v11_namespaced() -> None:
    assert gate.rank("example.jpg") == gate.rank("example.jpg")
    assert gate.rank("example.jpg") != gate.rank("different.jpg")
    assert gate.rank("example.jpg") != __import__("hashlib").sha256(
        f"{gate.SEED}:example.jpg".encode()
    ).hexdigest()


def test_read_excluded_source_names(tmp_path: Path) -> None:
    manifest = tmp_path / "old.jsonl"
    manifest.write_text(
        '{"source_filename":"a.jpg"}\n{"source_filename":"b.jpg"}\n'
    )
    assert gate.read_excluded_source_names(manifest) == {"a.jpg", "b.jpg"}

