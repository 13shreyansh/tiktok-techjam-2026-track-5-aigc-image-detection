from pathlib import Path


def test_demo_page_uses_verified_final_evidence_and_accessible_controls():
    text = Path("demo/index.html").read_text()

    for value in ("0.9872", "0.9837", "0.9308", "315,776,001", "161", "4"):
        assert value in text
    assert "run_v12.sh" in text
    assert "hidden-score estimate" in text
    assert "prefers-reduced-motion" in text
    assert "aria-label" in text
    assert "gradient text" not in text.lower()
    assert text.count("<section ") == 6
    for phrase in (
        "Our first great score was bad news",
        "removed ways to cheat",
        "made the world broader",
        "attractive fixes kept failing",
        "locked the final exam",
        "What survived",
    ):
        assert phrase in text


def test_product_context_is_judge_facing_brand_web():
    text = Path("PRODUCT.md").read_text()
    assert "## Register\n\nbrand" in text
    assert "## Platform\n\nweb" in text
    assert "Evidence before assertion" in text
