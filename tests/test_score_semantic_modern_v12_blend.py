import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kaggle_evaluate_semantic_modern_v12_gate as gate  # noqa: E402
import score_fixed_v12_blend as fixed  # noqa: E402
import score_semantic_modern_v12_blend as blend  # noqa: E402


def test_equal_blend_scores_frozen_semantic_groups(tmp_path: Path):
    rows = []
    for prompt in sorted(gate.EXPECTED_PROMPTS):
        for generator in sorted(gate.EXPECTED_GENERATORS):
            pair = f"{prompt}-{generator}"
            for label in (0, 1):
                token = f"{prompt}-{generator}-{label}"
                rows.append(
                    {
                        "label": label,
                        "workflow_purpose": "semantic-matched-modern-audit",
                        "training_allowed": False,
                        "organizer_demo_row": False,
                        "canonicalization": gate.CANONICALIZATION,
                        "canonical_format": "JPEG",
                        "canonical_width": 336,
                        "canonical_height": 336,
                        "image_sha256": (token.encode().hex() + "0" * 64)[:64],
                        "source_image_sha256": ("f" + token.encode().hex() + "0" * 64)[:64],
                        "semantic_prompt_id": prompt,
                        "paired_generator": generator,
                        "pair_id": pair,
                    }
                )
    manifest = tmp_path / "gate.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    roots = {}
    for candidate, offset in (("pe_core", 0.0), ("dinov2_control", 0.02)):
        root = tmp_path / candidate
        root.mkdir()
        progress = {
            "completed": True,
            "signature": {
                "candidate": candidate,
                "checkpoint_sha256": fixed.EXPECTED[candidate],
                "conditions": list(fixed.CONDITIONS),
            },
        }
        (root / "progress.json").write_text(json.dumps(progress))
        for condition in fixed.CONDITIONS:
            predictions = []
            for index, row in enumerate(rows):
                predictions.append(
                    {
                        "index": index,
                        "label": row["label"],
                        "image_sha256": row["image_sha256"],
                        "score": (0.8 if row["label"] else 0.2) + offset,
                    }
                )
            (root / f"{condition}_predictions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in predictions)
            )
        roots[candidate] = root
    result = blend.score(roots["pe_core"], roots["dinov2_control"], manifest)
    assert result["clean_auc"] == 1.0
    assert result["clean_semantic_metrics"]["paired_accuracy"] == 1.0
    assert result["frozen_gate_decision"]["passes_all_frozen_floors"]
