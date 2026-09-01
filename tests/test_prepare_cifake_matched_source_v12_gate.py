import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_cifake_matched_source_v12_gate import prior_cifake_hashes  # noqa: E402


def test_prior_cifake_hashes_uses_all_recorded_identity_fields(tmp_path):
    manifest = tmp_path / "used.jsonl"
    rows = [
        {
            "dataset": "CIFAKE",
            "source_image_sha256": "a" * 64,
            "image_sha256": "b" * 64,
            "sha256": "c" * 64,
        },
        {"dataset": "other", "sha256": "d" * 64},
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))

    hashes, manifests = prior_cifake_hashes(tmp_path)

    assert hashes == {"a" * 64, "b" * 64, "c" * 64}
    assert manifests == [str(manifest.resolve())]
