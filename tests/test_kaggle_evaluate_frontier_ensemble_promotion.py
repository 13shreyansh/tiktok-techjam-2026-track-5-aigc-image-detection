import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/kaggle_evaluate_frontier_ensemble_promotion.py"


def source() -> str:
    return SCRIPT.read_text()


def test_preselected_weight_and_checkpoint_hashes_are_frozen():
    text = source()
    assert 'V6_WEIGHT = 0.75' in text
    assert 'V9_WEIGHT = 0.25' in text
    assert '48ea50773fbd1b7247fff25fde6f985183e29f2eb517b5ac0f6319c1fe38b644' in text
    assert 'dd6b26c7849489447c7e96823f5b5e87c31623ca8ae0d28a5b162bb2dcb65075' in text


def test_pair_evaluation_uses_identical_tensor_batch_and_no_router():
    text = source()
    assert 'images.to("cuda:0"' in text
    assert 'images.to("cuda:1"' in text
    assert 'v6_model(images_v6)' in text
    assert 'v9_model(images_v9)' in text
    assert 'V6_WEIGHT * v6_scores_cpu + V9_WEIGHT * v9_scores_cpu' in text
    assert 'threshold' not in text.lower()
    assert 'router' not in text.lower()


def test_pair_evaluation_uses_verified_full_t4_batch():
    text = source()
    assert '"physical_batch_size": 128' in text
    assert 'batch_size=INFERENCE_POLICY["physical_batch_size"]' in text


def test_resume_signature_includes_numerical_inference_policy():
    text = source()
    assert '"autocast_dtype": "float16"' in text
    assert '"score_conversion": "per_model_float32_cpu"' in text
    assert '"blend_location_dtype": "cpu_float32"' in text
    assert '"inference_policy": INFERENCE_POLICY' in text


def test_all_four_predeclared_gates_are_present():
    text = source()
    for name in (
        'qwen_prompt_holdout',
        'ntire_shard5_full_audit',
        'internal_3071',
        'community_forensics_624',
    ):
        assert name in text


def test_non_frontier_pair_guards_use_the_summary_keys():
    text = source()
    assert '"worst_pair_auc": -0.01' not in text
    assert 'delta["clean_worst_pair_auc"] >= -0.01' in text
    assert 'delta["pooled_worst_pair_auc"] >= -0.01' in text


def test_demo_only_resources_are_not_named():
    lowered = source().lower()
    assert 'dall-e advanced' not in lowered
    assert 'coco val2017' not in lowered
