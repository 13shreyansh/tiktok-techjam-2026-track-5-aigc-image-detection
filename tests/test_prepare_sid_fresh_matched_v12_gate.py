import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_sid_fresh_matched_v12_gate as gate  # noqa: E402


def test_fresh_sid_source_identity_is_pinned():
    assert gate.PER_LABEL == 284
    assert gate.SOURCE_SHARD_BYTES == 505_844_042
    assert gate.SOURCE_SHARD_SHA256 == (
        "1447bbd98adf7eda68fca5615560c6b1de34c8e30157ff6b34ebd1e015a18042"
    )
    assert gate.SOURCE_MANIFEST_SHA256 == (
        "5f0815ac6ffac25bfd7724747a53d7536ea464d600dab9a3071480e724313c7f"
    )


def test_prior_sid_hashes_excludes_recorded_identity_but_not_fresh_source(tmp_path):
    source = tmp_path / "fresh.jsonl"
    source.write_text(json.dumps({"dataset": "SID_Set", "image_sha256": "a" * 64}) + "\n")
    used = tmp_path / "used.jsonl"
    used.write_text(
        json.dumps(
            {
                "dataset": "SID_Set",
                "source_image_sha256": "b" * 64,
                "image_sha256": "c" * 64,
                "sha256": "d" * 64,
            }
        )
        + "\n"
    )

    hashes, manifests = gate.prior_sid_hashes(tmp_path, source)

    assert hashes == {"b" * 64, "c" * 64, "d" * 64}
    assert manifests == [str(used.resolve())]
