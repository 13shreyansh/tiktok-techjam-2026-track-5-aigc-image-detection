import sys
import hashlib
import json
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kaggle_evaluate_openimages_source_rotation_v12 as audit  # noqa: E402
import kaggle_evaluate_semantic_modern_v12_gate as base  # noqa: E402


def valid_rows():
    rows = []
    for prompt in sorted(base.EXPECTED_PROMPTS):
        for generator in sorted(base.EXPECTED_GENERATORS):
            pair_id = f"{prompt}-{generator}"
            for label in (0, 1):
                token = f"{prompt}-{generator}-{label}"
                rows.append(
                    {
                        "label": label,
                        "workflow_purpose": "semantic-matched-modern-audit",
                        "training_allowed": False,
                        "organizer_demo_row": False,
                        "canonicalization": base.CANONICALIZATION,
                        "canonical_format": "JPEG",
                        "canonical_width": 336,
                        "canonical_height": 336,
                        "image_sha256": (token.encode().hex() + "0" * 64)[:64],
                        "source_image_sha256": ("f" + token.encode().hex() + "0" * 64)[:64],
                        "semantic_prompt_id": prompt,
                        "paired_generator": generator,
                        "pair_id": pair_id,
                        "dataset": "Open-Images-V7-validation-CVDF" if label == 0 else "Qwen-Image-Bench",
                        "real_source": "Open-Images-V7-validation-CVDF" if label == 0 else None,
                    }
                )
    return rows


def test_frozen_source_rotation_hashes_are_exact():
    assert audit.PACKAGE_SHA256 == "c9862550a1476e60b13e9c262e751d3d75e8c582a71d60496c064185b63e4906"
    assert audit.INVENTORY_SHA256 == "ed12e13391e84770aa3296eb1db1e13d97e74fba1d65fca480bd77cac2250382"
    assert audit.MANIFEST_SHA256 == "a696d1a781dacaa66183fb96e6a4078ebdf3dd429dedb657524ddf846b3667d6"


def test_openimages_contract_accepts_only_openimages_reals():
    report = audit.validate_openimages_rows(valid_rows())
    assert report["real_source"] == "Open-Images-V7-validation-CVDF"


def test_openimages_contract_rejects_wrong_real_collection():
    rows = valid_rows()
    rows[0]["dataset"] = "COCO-train2017"
    with pytest.raises(RuntimeError, match="real-source mismatch"):
        audit.validate_openimages_rows(rows)


def test_mounted_gate_recomputes_content_inventory(monkeypatch, tmp_path):
    root = tmp_path / "owner" / audit.GATE_MOUNT_SLUG / "gate"
    image = root / "images" / "00"
    image.mkdir(parents=True)
    content = b"frozen image"
    digest = hashlib.sha256(content).hexdigest()
    path = image / f"{digest}.jpg"
    path.write_bytes(content)
    manifest = root / "manifests" / "eval_semantic_matched.jsonl"
    manifest.parent.mkdir()
    manifest.write_text("{}\n")
    inventory = hashlib.sha256(
        f"{digest}\t{len(content)}\timages/00/{digest}.jpg\n".encode()
    ).hexdigest()
    (root / "package.json").write_text(
        json.dumps({"inventory_sha256": inventory, "unique_images": 1})
    )
    monkeypatch.setattr(audit, "INVENTORY_SHA256", inventory)
    monkeypatch.setattr(audit, "MANIFEST_SHA256", hashlib.sha256(b"{}\n").hexdigest())
    monkeypatch.setattr(audit, "GATE_MOUNT_SEARCH_ROOT", tmp_path)
    observed_root, report = audit.validate_mounted_gate()
    assert observed_root == root
    assert report["mount_validation"] == "file_inventory"
