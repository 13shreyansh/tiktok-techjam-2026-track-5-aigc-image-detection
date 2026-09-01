from pathlib import Path


def test_unseen_prompt_script_freezes_lineage_and_contract():
    text = Path("scripts/kaggle_evaluate_qwen_unseen_prompt_holdout.py").read_text()
    assert "d2493deb153b020cf169c7e3f57d15e4dd697038" in text
    assert "985d0842c9f38a4771cb247cf48753edf6b9564f9d41eb2b1fdc7bf0af85e0c7" in text
    assert "c7f565e333aa09954243d0c41e90fb447c02045a8d57df513ff711b1e7c1caaa" in text
    assert "PHYSICAL_BATCH_SIZE = 64" in text
    assert "jpeg_q96_stretch_full_frame" in text
    assert "promotion.V6_SHA256" in text
    assert "promotion.V9_SHA256" in text
    assert "demo" not in text.lower()
