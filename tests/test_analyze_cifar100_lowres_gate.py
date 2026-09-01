from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_cifar100_lowres_gate as audit


def rows(label: int, count: int, score: float, fine_classes: int = 100) -> list[dict]:
    return [
        {
            "label": label,
            "image_sha256": f"{label}-{index}",
            "fine_label": index % fine_classes,
            "v6_score": score,
            "score": score,
        }
        for index in range(count)
    ]


def test_condition_metrics_detects_complete_inversion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit, "BOOTSTRAP_REPLICATES", 10)
    real = rows(0, 1000, 0.9)
    fake = rows(1, 288, 0.1, fine_classes=1)
    result = audit.condition_metrics(real, fake)
    assert result["blend_auc"] == 0.0
    assert result["mean_score_inversion"] is True
    assert result["illustrative_fraction_real_at_or_above_0.5"]["blend"] == 1.0
    decision = audit.interpretation(result, result)
    assert "inversion confirmed" in decision["predeclared_noise_band"]
    assert decision["repair_priority"] is True


def test_auc_uses_fake_as_positive_label() -> None:
    assert audit.auc(np.asarray([0.1, 0.2]), np.asarray([0.8, 0.9])) == 1.0
    assert audit.auc(np.asarray([0.8, 0.9]), np.asarray([0.1, 0.2])) == 0.0


def test_duplicate_real_hashes_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit, "BOOTSTRAP_REPLICATES", 2)
    real = rows(0, 1000, 0.1)
    real[1]["image_sha256"] = real[0]["image_sha256"]
    fake = rows(1, 288, 0.9, fine_classes=1)
    with pytest.raises(ValueError, match="duplicate hashes"):
        audit.condition_metrics(real, fake)
