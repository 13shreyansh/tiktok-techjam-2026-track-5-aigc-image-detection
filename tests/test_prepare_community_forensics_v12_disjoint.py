import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_community_forensics_v12_disjoint as prepare  # noqa: E402


def test_derive_excludes_both_v12_source_and_canonical_identities():
    gate = [
        {"image_sha256": "keep", "label": 1, "generator_model": "model-a"},
        {"image_sha256": "source-overlap", "label": 0, "real_source": "sid"},
        {"image_sha256": "canonical-overlap", "label": 0, "real_source": "cifake"},
    ]
    v12 = [
        {"image_sha256": "canonical-overlap", "source_image_sha256": "source-overlap"}
    ]
    retained, report = prepare.derive(gate, v12)
    assert retained == [gate[0]]
    assert report["excluded_rows"] == 2
    assert report["excluded_labels"] == {0: 2}
    assert report["v12_identity_overlap_after_filter"] == 0
