#!/usr/bin/env python3
"""Screen two manifests for resized or recompressed image overlap.

This is a conservative audit, not an automatic deduplicator.  It reports
candidate pairs whose difference hash and average hash are both close.  Exact
byte overlap remains the stronger invariant and is checked separately by the
manifest builders.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


@dataclass(frozen=True)
class Fingerprint:
    path: str
    label: int
    dhash: int
    ahash: int


class BKTree:
    """Small metric tree for 64-bit Hamming-distance lookup."""

    def __init__(self) -> None:
        self.root: tuple[int, list[int], dict[int, object]] | None = None

    @staticmethod
    def distance(left: int, right: int) -> int:
        return bin(left ^ right).count("1")

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = (value, [index], {})
            return
        node = self.root
        while True:
            node_value, indices, children = node
            distance = self.distance(value, node_value)
            if distance == 0:
                indices.append(index)
                return
            child = children.get(distance)
            if child is None:
                children[distance] = (value, [index], {})
                return
            node = child  # type: ignore[assignment]

    def query(self, value: int, radius: int) -> list[int]:
        if self.root is None:
            return []
        matches: list[int] = []
        pending = [self.root]
        while pending:
            node_value, indices, children = pending.pop()
            distance = self.distance(value, node_value)
            if distance <= radius:
                matches.extend(indices)
            low, high = distance - radius, distance + radius
            pending.extend(
                child
                for edge, child in children.items()
                if low <= edge <= high
            )
        return matches


def load_rows(manifest: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def bits_to_int(bits: list[bool]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def fingerprint(manifest: Path, row: dict) -> Fingerprint:
    image_path = (manifest.parent / row["path"]).resolve()
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        dhash_image = image.resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(dhash_image.getdata())
        dhash_bits = [
            pixels[y * 9 + x] > pixels[y * 9 + x + 1]
            for y in range(8)
            for x in range(8)
        ]
        ahash_image = image.resize((8, 8), Image.Resampling.LANCZOS)
        ahash_pixels = list(ahash_image.getdata())
        average = sum(ahash_pixels) / len(ahash_pixels)
        ahash_bits = [pixel >= average for pixel in ahash_pixels]
    return Fingerprint(
        path=str(image_path),
        label=int(row["label"]),
        dhash=bits_to_int(dhash_bits),
        ahash=bits_to_int(ahash_bits),
    )


def fingerprints(manifest: Path, rows: list[dict], workers: int) -> list[Fingerprint]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda row: fingerprint(manifest, row), rows))


def audit(
    train: list[Fingerprint],
    evaluation: list[Fingerprint],
    dhash_radius: int,
    ahash_radius: int,
    max_examples: int,
) -> dict:
    tree = BKTree()
    for index, item in enumerate(train):
        tree.add(item.dhash, index)
    count = 0
    cross_label_count = 0
    examples: list[dict] = []
    for candidate in evaluation:
        for index in tree.query(candidate.dhash, dhash_radius):
            reference = train[index]
            ahash_distance = BKTree.distance(candidate.ahash, reference.ahash)
            if ahash_distance > ahash_radius:
                continue
            count += 1
            cross_label_count += int(candidate.label != reference.label)
            if len(examples) < max_examples:
                examples.append(
                    {
                        "train_path": reference.path,
                        "train_label": reference.label,
                        "eval_path": candidate.path,
                        "eval_label": candidate.label,
                        "dhash_distance": BKTree.distance(
                            candidate.dhash, reference.dhash
                        ),
                        "ahash_distance": ahash_distance,
                    }
                )
    return {
        "train_images": len(train),
        "evaluation_images": len(evaluation),
        "dhash_radius": dhash_radius,
        "ahash_radius": ahash_radius,
        "candidate_pairs": count,
        "cross_label_candidate_pairs": cross_label_count,
        "examples": examples,
        "interpretation": (
            "Candidate pairs require manual review; perceptual hashes can collide. "
            "A cross-label candidate is especially serious and must be resolved "
            "before using the evaluation gate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dhash-radius", type=int, default=4)
    parser.add_argument("--ahash-radius", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=100)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for radius in (args.dhash_radius, args.ahash_radius):
        if not 0 <= radius <= 64:
            parser.error("hash radii must be between 0 and 64")

    train_rows = load_rows(args.train_manifest)
    eval_rows = load_rows(args.eval_manifest)
    train_fingerprints = fingerprints(args.train_manifest, train_rows, args.workers)
    eval_fingerprints = fingerprints(args.eval_manifest, eval_rows, args.workers)
    result = audit(
        train_fingerprints,
        eval_fingerprints,
        args.dhash_radius,
        args.ahash_radius,
        args.max_examples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
