from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kaggle_screen_v11_quality_route as screen  # noqa: E402


def test_metrics_and_screen_decision() -> None:
    rows = [
        {"label": 0, "general_score": 0.1, "score": 0.2},
        {"label": 0, "general_score": 0.2, "score": 0.3},
        {"label": 1, "general_score": 0.8, "score": 0.7},
        {"label": 1, "general_score": 0.9, "score": 0.8},
    ]
    assert screen.metrics(rows, "score")["auc"] == 1.0
    reports = {
        "gate": {
            "clean": {
                "general": {"auc": 0.9},
                "routed": {"auc": 0.899},
                "route_rate": 0.01,
            },
            "noise_sigma_0.10": {
                "general": {"auc": 0.6},
                "routed": {"auc": 0.7},
                "route_rate": 0.99,
            },
        }
    }
    assert screen.screen_decision(reports)["passes"] is True


def test_screen_rejects_noise_regression() -> None:
    reports = {
        "gate": {
            "clean": {
                "general": {"auc": 0.9},
                "routed": {"auc": 0.9},
                "route_rate": 0.0,
            },
            "noise_sigma_0.10": {
                "general": {"auc": 0.8},
                "routed": {"auc": 0.79},
                "route_rate": 1.0,
            },
        }
    }
    assert screen.screen_decision(reports)["passes"] is False

