#!/usr/bin/env python3
"""Measure how well non-semantic file metadata alone predicts real versus AI."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC = ["log_width", "log_height", "log_area", "log_bytes", "aspect", "is_square"]
CATEGORICAL = ["format", "mode", "suffix"]


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def feature_rows(manifest: Path) -> tuple[list[dict], list[int]]:
    features, labels = [], []
    for row in rows(manifest):
        path = Path(row["path"])
        if not path.is_absolute():
            path = (manifest.parent / path).resolve()
        with Image.open(path) as image:
            width, height = image.size
            image_format = str(image.format or "unknown")
            mode = str(image.mode)
        features.append(
            {
                "log_width": math.log1p(width),
                "log_height": math.log1p(height),
                "log_area": math.log1p(width * height),
                "log_bytes": math.log1p(path.stat().st_size),
                "aspect": width / height,
                "is_square": float(width == height),
                "format": image_format,
                "mode": mode,
                "suffix": path.suffix.lower(),
            }
        )
        labels.append(int(row["label"]))
    return features, labels


def matrix(records: list[dict]) -> np.ndarray:
    return np.asarray([[row[key] for key in NUMERIC + CATEGORICAL] for row in records], dtype=object)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train_x, train_y = feature_rows(args.train)
    eval_x, eval_y = feature_rows(args.eval)
    numeric_indices = list(range(len(NUMERIC)))
    categorical_indices = list(range(len(NUMERIC), len(NUMERIC) + len(CATEGORICAL)))
    model = Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            Pipeline(
                                [
                                    ("impute", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            numeric_indices,
                        ),
                        (
                            "categorical",
                            OneHotEncoder(handle_unknown="ignore"),
                            categorical_indices,
                        ),
                    ]
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(matrix(train_x), train_y)
    train_scores = model.predict_proba(matrix(train_x))[:, 1]
    eval_scores = model.predict_proba(matrix(eval_x))[:, 1]
    square_rule = np.asarray([row["is_square"] for row in eval_x], dtype=float)
    png_rule = np.asarray([row["format"] == "PNG" for row in eval_x], dtype=float)
    report = {
        "warning": (
            "A high score proves that the dataset leaks labels through container or "
            "shape metadata. It does not prove that the image model uses those cues."
        ),
        "features": NUMERIC + CATEGORICAL,
        "train_rows": len(train_y),
        "eval_rows": len(eval_y),
        "train_auc": float(roc_auc_score(train_y, train_scores)),
        "eval_auc": float(roc_auc_score(eval_y, eval_scores)),
        "eval_square_only_auc": float(roc_auc_score(eval_y, square_rule)),
        "eval_png_only_auc": float(roc_auc_score(eval_y, png_rule)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
