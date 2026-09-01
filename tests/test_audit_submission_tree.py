from pathlib import Path


SCRIPT = Path("scripts/audit_submission_tree.py")


def source() -> str:
    return SCRIPT.read_text()


def test_submission_tree_audit_requires_judge_facing_artifacts() -> None:
    text = source()
    for name in (
        "README.md",
        "MODEL_CARD.md",
        "ROBUSTNESS_AND_ERROR_ANALYSIS.md",
        "THIRD_PARTY_NOTICES.md",
        "demo/index.html",
        "SELECTED_CHECKPOINT.sha256",
        "V12_CHECKPOINT_MANIFEST.json",
        "V12_RUNNABLE_CONTRACT_RESULT.json",
        "V12_SELECTED_DEFAULT_RUN_RESULT.json",
        "NTIRE_V12_FINAL_ARBITRATION_RESULT.json",
        "V12_ERROR_ANALYSIS_RESULT.json",
        "MODEL_WEIGHTS_LICENSE.md",
        "run_v12.sh",
    ):
        assert name in text


def test_submission_tree_audit_rejects_weights_and_private_locators() -> None:
    text = source()
    for suffix in (".pt", ".pth", ".safetensors", ".zip"):
        assert suffix in text
    for kind in (
        "absolute_user_home",
        "private_kaggle_notebook_url",
        "account_scoped_kaggle_input",
        "private_key_material",
        "unresolved_release_placeholder",
    ):
        assert kind in text


def test_submission_tree_audit_checks_parameter_limit_and_urls() -> None:
    text = source()
    assert 'manifest["selected_runtime_mode"]' in text
    assert 'selected["parameters"]' in text
    assert 'manifest["organizer_parameter_limit_exclusive"]' in text
    assert 'selected.get("distribution_url")' in text


def test_submission_tree_audit_supports_history_free_export() -> None:
    text = source()
    assert '"git_tracked_tree"' in text
    assert '"history_free_filesystem_tree"' in text
    assert 'root.rglob("*")' in text


def test_submission_tree_audit_preserves_explicit_safety_counts() -> None:
    text = source()
    for field in (
        "forbidden_tracked_weights_archives_or_large_files",
        "flagged_private_or_personal_locators_in_current_tree",
        "flagged_private_key_material",
        "unresolved_release_placeholders",
        "history_boundary",
    ):
        assert field in text
