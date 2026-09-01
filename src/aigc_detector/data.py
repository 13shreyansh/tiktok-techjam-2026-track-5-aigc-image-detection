from __future__ import annotations

import random
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def discover_binary_images(root: Path) -> list[tuple[Path, int]]:
    """Discover REAL/FAKE folders and map AI-generated (FAKE) to positive label 1."""
    samples: list[tuple[Path, int]] = []
    aliases = {"real": 0, "authentic": 0, "fake": 1, "ai": 1, "generated": 1}
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        key = class_dir.name.lower()
        if key not in aliases:
            continue
        label = aliases[key]
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                samples.append((path, label))
    labels = {label for _, label in samples}
    if labels != {0, 1}:
        raise ValueError(f"expected both real and fake class folders under {root}; found labels={labels}")
    return samples


def balanced_limit(samples: list[tuple[Path, int]], max_per_class: Optional[int], seed: int) -> list[tuple[Path, int]]:
    if max_per_class is None:
        return samples
    grouped: dict[int, list[tuple[Path, int]]] = defaultdict(list)
    for sample in samples:
        grouped[sample[1]].append(sample)
    rng = random.Random(seed)
    chosen: list[tuple[Path, int]] = []
    for label in (0, 1):
        pool = grouped[label]
        rng.shuffle(pool)
        chosen.extend(pool[:max_per_class])
    rng.shuffle(chosen)
    return chosen


class BinaryFolderDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        max_per_class: Optional[int] = None,
        seed: int = 20260829,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples = balanced_limit(discover_binary_images(self.root), max_per_class, seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, float(label), str(path)


def discover_manifest_records(manifest_path: Path) -> list[dict]:
    """Load validated manifest rows and resolve each image path."""
    records: list[dict] = []
    seen: set[Path] = set()
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on {manifest_path}:{line_number}") from error
            if not isinstance(row, dict) or "path" not in row or "label" not in row:
                raise ValueError(f"manifest row {line_number} requires path and label")
            label = row["label"]
            if label not in (0, 1):
                raise ValueError(f"manifest row {line_number} has invalid label {label!r}")
            path = Path(str(row["path"]))
            if not path.is_absolute():
                path = manifest_path.parent / path
            path = path.resolve()
            if path in seen:
                raise ValueError(f"duplicate image path on manifest row {line_number}: {path}")
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise ValueError(f"manifest row {line_number} is not a supported image: {path}")
            seen.add(path)
            records.append({**row, "path": path, "label": int(label)})
    labels = {record["label"] for record in records}
    if labels != {0, 1}:
        raise ValueError(f"expected both labels in {manifest_path}; found labels={labels}")
    return records


def discover_manifest_images(manifest_path: Path) -> list[tuple[Path, int]]:
    """Load a provenance-preserving JSONL manifest for binary training."""
    samples = [
        (record["path"], record["label"])
        for record in discover_manifest_records(manifest_path)
    ]
    return samples


def source_balanced_weights(
    manifest_path: str | Path,
    samples: list[tuple[Path, int]],
) -> tuple[list[float], dict]:
    """Weight labels equally and sources equally within each label.

    This prevents a large dataset branch from dominating an epoch. It does not
    claim to remove source fingerprints; source-held-out evaluation remains the
    deciding gate.
    """
    records = discover_manifest_records(Path(manifest_path))
    sampled_paths = {path.resolve() for path, _ in samples}
    group_by_path: dict[Path, tuple[int, str]] = {}
    counts: Counter[tuple[int, str]] = Counter()
    for record in records:
        if record["path"] not in sampled_paths:
            continue
        label = int(record["label"])
        key = "generator" if label == 1 else "real_source"
        group = str(record.get(key, "unknown"))
        group_key = (label, group)
        group_by_path[record["path"]] = group_key
        counts[group_key] += 1
    groups_per_label = Counter(label for label, _ in counts)
    weights = []
    for path, label in samples:
        group_key = group_by_path[path.resolve()]
        group_count = counts[group_key]
        weights.append(1.0 / (groups_per_label[label] * group_count))
    report = {
        "policy": "equal labels; equal named sources within each label",
        "groups_per_label": {
            str(label): groups_per_label[label] for label in sorted(groups_per_label)
        },
        "group_counts": {
            f'{"fake" if label else "real"}:{group}': count
            for (label, group), count in sorted(counts.items())
        },
    }
    return weights, report


class BinaryManifestDataset(BinaryFolderDataset):
    """Binary dataset backed by JSONL rows with path, label and optional provenance."""

    def __init__(
        self,
        manifest_path: str | Path,
        transform: Optional[Callable] = None,
        max_per_class: Optional[int] = None,
        seed: int = 20260829,
    ) -> None:
        self.root = Path(manifest_path)
        self.transform = transform
        self.samples = balanced_limit(
            discover_manifest_images(self.root), max_per_class, seed
        )


def binary_dataset(
    source: str | Path,
    transform: Optional[Callable] = None,
    max_per_class: Optional[int] = None,
    seed: int = 20260829,
) -> BinaryFolderDataset:
    source_path = Path(source)
    dataset_type = BinaryManifestDataset if source_path.suffix.lower() == ".jsonl" else BinaryFolderDataset
    return dataset_type(source_path, transform=transform, max_per_class=max_per_class, seed=seed)
