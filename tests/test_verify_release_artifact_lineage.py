import json

from scripts.verify_release_artifact_lineage import audit


def test_audit_keeps_compact_ablation_with_v9(tmp_path):
    v6 = tmp_path / "v6.pt"
    v9 = tmp_path / "missing-v9.pt"
    v6.write_bytes(b"v6")
    import hashlib

    v6_hash = hashlib.sha256(b"v6").hexdigest()
    v9_hash = "9" * 64
    compact_hash = "c" * 64
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "combined_checkpoint_bytes": 5,
                "checkpoints": {
                    "v6": {"bytes": 2, "sha256": v6_hash, "distribution_url": None},
                    "v9": {
                        "bytes": 3,
                        "sha256": v9_hash,
                        "distribution_url": None,
                        "rejected_fp16_storage_ablation": {
                            "sha256": compact_hash,
                            "exact_clean_qwen_screen_passed": False,
                        },
                    },
                },
            }
        )
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps({"v6_checkpoint_sha256": v6_hash, "v9_checkpoint_sha256": v9_hash})
    )
    compact = tmp_path / "compact.json"
    compact.write_text(
        json.dumps(
            {
                "source_checkpoint_sha256": v9_hash,
                "compact_checkpoint_sha256": compact_hash,
                "passes_exact_clean_qwen_screen": False,
            }
        )
    )

    report = audit(manifest, [evidence], compact, v6, v9)

    assert report["checks_passed"] is True
    assert report["local_artifacts"]["v6"]["matches_manifest"] is True
    assert report["local_artifacts"]["v9"]["present"] is False
    assert report["distribution_ready"] is False
    assert "exact v9 checkpoint is absent locally" in report["distribution_blockers"]
