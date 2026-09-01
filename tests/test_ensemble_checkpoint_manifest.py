import json
from pathlib import Path


def test_ensemble_checkpoint_manifest_is_exact_and_truthful():
    manifest = json.loads(Path("ENSEMBLE_CHECKPOINT_MANIFEST.json").read_text())
    assert manifest["status"] == "verified_locally_and_on_kaggle_but_not_published"
    assert manifest["ensemble_total_parameters"] == 631_552_002
    assert manifest["ensemble_total_parameters"] < manifest["organizer_parameter_limit_exclusive"]
    assert manifest["combined_checkpoint_bytes"] == 1_894_848_234
    assert manifest["checkpoints"]["v6"]["sha256"] == "48ea50773fbd1b7247fff25fde6f985183e29f2eb517b5ac0f6319c1fe38b644"
    assert manifest["checkpoints"]["v9"]["sha256"] == "dd6b26c7849489447c7e96823f5b5e87c31623ca8ae0d28a5b162bb2dcb65075"
    assert manifest["checkpoints"]["v6"]["distribution_url"] is None
    assert manifest["checkpoints"]["v9"]["distribution_url"] is None
    assert "rejected_fp16_storage_ablation" not in manifest["checkpoints"]["v6"]
    rejected_v9 = manifest["checkpoints"]["v9"]["rejected_fp16_storage_ablation"]
    assert rejected_v9["sha256"] == "85094e995c17cca25c1e5367a580d88f5ceb927045fc978e52d4ba1b1c845c45"
    assert rejected_v9["exact_clean_qwen_screen_passed"] is False
    preserved = manifest["checkpoints"]["v9"]["private_preservation"]
    assert preserved["state"] == "successful_private_kaggle_version_output"
    assert preserved["account_locator_redacted_from_release_tree"] is True
    assert "script_version_id" not in preserved
    assert "private_version_url" not in preserved
    assert preserved["pre_save_sha256_verified"] is True
    assert preserved["post_save_download_and_sha256_verified"] is True
    assert preserved["post_save_verified_filename"] == "000_model_v9_exact_dd6b26c.pt"
    assert preserved["post_save_verified_bytes"] == manifest["checkpoints"]["v9"]["bytes"]
    assert preserved["post_save_verified_sha256"] == manifest["checkpoints"]["v9"]["sha256"]
    assert manifest["checkpoints"]["v9"]["local_repository_state"] == "present_but_gitignored"
    assert preserved["publicly_accessible"] is False


def test_ensemble_checkpoint_manifest_matches_frozen_arithmetic():
    policy = json.loads(Path("ENSEMBLE_CHECKPOINT_MANIFEST.json").read_text())["inference_contract"]
    assert policy == {
        "v6_probability_weight": 0.75,
        "v9_probability_weight": 0.25,
        "physical_batch_size": 64,
        "cuda_autocast_dtype": "float16",
        "blend_location_dtype": "cuda_float16",
        "score_conversion": "float32_after_blend",
    }
