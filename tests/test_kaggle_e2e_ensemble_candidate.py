from pathlib import Path


SCRIPT = Path("scripts/kaggle_e2e_ensemble_candidate.py")


def test_e2e_candidate_freezes_runner_and_open_gate():
    text = SCRIPT.read_text()
    assert 'RUNNER_SHA256 = "' in text
    assert "RUNNER_SHA256_AFTER_NULL_FIX" not in text
    assert "SOURCE_ARCHIVE_SHA256_AFTER_NULL_FIX" not in text
    assert 'EXPECTED_ROWS = 576' in text
    assert 'sealed.MANIFESTS["qwen_prompt_holdout"]' in text


def test_e2e_candidate_uses_isolated_python_and_exact_score_gate():
    text = SCRIPT.read_text()
    assert '"/usr/bin/python3"' in text
    assert '"aigc_detector.predict_ensemble"' in text
    assert '"passes": max(drifts) == 0.0' in text
    assert 'finite_probabilities_in_unit_interval' in text


def test_e2e_candidate_refuses_artifact_reuse():
    text = SCRIPT.read_text()
    assert 'if INPUT_DIRECTORY.exists() or OUTPUT.exists() or REPORT.exists()' in text
    assert 'refusing to reuse an existing candidate E2E artifact' in text


def test_e2e_candidate_does_not_name_forbidden_demo_resources():
    assert 'demo' not in SCRIPT.read_text().lower()
