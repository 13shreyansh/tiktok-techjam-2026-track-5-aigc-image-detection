import json
from pathlib import Path


def test_checkpoint_sidecar_matches_manifest() -> None:
    manifest = json.loads(Path("ENSEMBLE_CHECKPOINT_MANIFEST.json").read_text())
    observed = {}
    for line in Path("CHECKPOINTS.sha256").read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        observed[filename] = digest
    assert observed == {
        "models/model.pt": manifest["checkpoints"]["v6"]["sha256"],
        "models/model_v9.pt": manifest["checkpoints"]["v9"]["sha256"],
    }
