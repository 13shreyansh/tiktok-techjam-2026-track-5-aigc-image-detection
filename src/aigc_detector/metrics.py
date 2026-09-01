from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class OfficialMetrics:
    clean_auc: float
    robust_auc: float
    official_score: float
    clean_count: int
    robust_count: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _as_arrays(labels: Iterable[float], scores: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(list(labels), dtype=np.int64)
    p = np.asarray(list(scores), dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or y.size != p.size or y.size == 0:
        raise ValueError("labels and scores must be non-empty one-dimensional arrays of equal length")
    if not np.isfinite(p).all():
        raise ValueError("scores must be finite")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("both authentic=0 and AI-generated=1 labels are required")
    return y, p


def auc(labels: Iterable[float], scores: Iterable[float]) -> float:
    y, p = _as_arrays(labels, scores)
    return float(roc_auc_score(y, p))


def official_metrics(
    clean_labels: Iterable[float],
    clean_scores: Iterable[float],
    robust_labels: Iterable[float],
    robust_scores: Iterable[float],
) -> OfficialMetrics:
    clean_y, clean_p = _as_arrays(clean_labels, clean_scores)
    robust_y, robust_p = _as_arrays(robust_labels, robust_scores)
    clean_auc = float(roc_auc_score(clean_y, clean_p))
    robust_auc = float(roc_auc_score(robust_y, robust_p))
    return OfficialMetrics(
        clean_auc=clean_auc,
        robust_auc=robust_auc,
        official_score=0.5 * clean_auc + 0.5 * robust_auc,
        clean_count=int(clean_y.size),
        robust_count=int(robust_y.size),
    )
