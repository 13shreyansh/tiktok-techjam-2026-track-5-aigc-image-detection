#!/usr/bin/env python3
"""Build an audit-only, content-matched modern-generator gate.

For each unused Qwen Image Bench image, a separate pinned full CLIP model
computes a visual embedding. A globally unique set of fresh COCO train2017
photographs is assigned to those fake-image embeddings. Candidate detector
scores are never read. Both labels then receive the exact v12 canonicalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import open_clip
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset

try:
    from materialize_canonical_v12 import POLICY, canonical_bytes, file_sha256
except ModuleNotFoundError:  # package-style import used by local tests
    from scripts.materialize_canonical_v12 import POLICY, canonical_bytes, file_sha256


MODEL_NAME = "RN50-quickgelu"
MODEL_REPOSITORY = "timm/resnet50_clip.openai"
MODEL_REVISION = "ec3d92cf63a5f9d591f0d611b736895966c73076"
MODEL_FILENAME = "open_clip_model.safetensors"
MODEL_WEIGHT_SHA256 = "da0baa37fb2211eee5729ab69ed587eb10d48f5b7076ceeed01552fc3d4cb4ee"
MODEL_WEIGHT_BYTES = 408291932
SEED = 20260901
EXPECTED_GENERATORS = 18
EXPECTED_PROMPTS = 8
REAL_ROWS_PER_PROMPT = 18


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def resolve_row_path(row: dict, manifest: Path) -> Path:
    path = Path(row["path"])
    return path if path.is_absolute() else (manifest.parent / path).resolve()


def identity_values(row: dict) -> set[str]:
    return {
        str(row[key])
        for key in ("sha256", "image_sha256", "source_image_sha256")
        if row.get(key)
    }


def collect_excluded_identities(manifests: list[Path]) -> set[str]:
    identities: set[str] = set()
    for manifest in manifests:
        for row in read_rows(manifest):
            identities.update(identity_values(row))
    return identities


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as opened:
            image = opened.convert("RGB")
        return self.transform(image)


def extract_features(
    model: torch.nn.Module,
    paths: list[Path],
    transform,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(
        ImageDataset(paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    features = []
    with torch.inference_mode():
        for batch in loader:
            values = model.encode_image(batch.to(device), normalize=True).float()
            features.append(values.cpu())
    return torch.cat(features).numpy()


def assign_unique_reals(
    fake_rows: list[dict], fake_features: np.ndarray, real_features: np.ndarray
) -> tuple[dict[int, list[tuple[int, float]]], np.ndarray]:
    prompt_indices: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(fake_rows):
        prompt_indices[int(row["prompt_id"])].append(index)
    prompt_ids = sorted(prompt_indices)
    if len(prompt_ids) != EXPECTED_PROMPTS:
        raise RuntimeError(f"expected {EXPECTED_PROMPTS} prompts: {prompt_ids}")
    centroids = []
    for prompt_id in prompt_ids:
        indices = prompt_indices[prompt_id]
        if len(indices) != EXPECTED_GENERATORS:
            raise RuntimeError(
                f"prompt {prompt_id}: expected {EXPECTED_GENERATORS} generators"
            )
        centroid = fake_features[indices].mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        centroids.append(centroid)
    centroid_array = np.stack(centroids)
    similarity = centroid_array @ real_features.T
    slot_prompt_positions = np.repeat(
        np.arange(len(prompt_ids)), REAL_ROWS_PER_PROMPT
    )
    slot_scores = similarity[slot_prompt_positions]
    slot_rows, real_columns = linear_sum_assignment(-slot_scores)
    if len(set(real_columns.tolist())) != len(real_columns):
        raise RuntimeError("semantic assignment reused a real image")
    assigned: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for slot_row, real_index in zip(slot_rows.tolist(), real_columns.tolist()):
        prompt_position = int(slot_prompt_positions[slot_row])
        prompt_id = prompt_ids[prompt_position]
        assigned[prompt_id].append(
            (int(real_index), float(similarity[prompt_position, real_index]))
        )
    for prompt_id in prompt_ids:
        assigned[prompt_id].sort(key=lambda item: (-item[1], item[0]))
        if len(assigned[prompt_id]) != REAL_ROWS_PER_PROMPT:
            raise RuntimeError(f"prompt {prompt_id}: incomplete assignment")
    return dict(assigned), similarity


def assign_unique_reals_individually(
    fake_features: np.ndarray, real_features: np.ndarray
) -> tuple[list[tuple[int, float]], np.ndarray]:
    """Globally pair each actual fake image with one unique closest real image."""
    similarity = fake_features @ real_features.T
    fake_indices, real_indices = linear_sum_assignment(-similarity)
    if fake_indices.tolist() != list(range(len(fake_features))):
        raise RuntimeError("individual semantic assignment is incomplete")
    if len(set(real_indices.tolist())) != len(real_indices):
        raise RuntimeError("individual semantic assignment reused a real image")
    assigned = [
        (int(real_index), float(similarity[fake_index, real_index]))
        for fake_index, real_index in zip(fake_indices.tolist(), real_indices.tolist())
    ]
    return assigned, similarity


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return file_sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake-manifest", type=Path, required=True)
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--matching-mode",
        choices=("prompt-centroid", "individual-image"),
        default="prompt-centroid",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    started = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    fake_rows = read_rows(args.fake_manifest)
    real_rows = read_rows(args.real_manifest)
    if len(fake_rows) != EXPECTED_GENERATORS * EXPECTED_PROMPTS:
        raise RuntimeError(f"unexpected fake row count: {len(fake_rows)}")
    if len(real_rows) < EXPECTED_GENERATORS * EXPECTED_PROMPTS:
        raise RuntimeError(f"real pool too small: {len(real_rows)}")
    if Counter(int(row["label"]) for row in fake_rows) != {1: len(fake_rows)}:
        raise RuntimeError("fake manifest label contract failed")
    if Counter(int(row["label"]) for row in real_rows) != {0: len(real_rows)}:
        raise RuntimeError("real manifest label contract failed")

    fake_paths = [resolve_row_path(row, args.fake_manifest) for row in fake_rows]
    real_paths = [resolve_row_path(row, args.real_manifest) for row in real_rows]
    for path in fake_paths + real_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    source_hashes = [file_sha256(path) for path in fake_paths + real_paths]
    if len(set(source_hashes)) != len(source_hashes):
        raise RuntimeError("source identity collision inside semantic gate inputs")
    excluded = collect_excluded_identities(args.exclude_manifest)
    overlap = set(source_hashes) & excluded
    if overlap:
        raise RuntimeError(f"semantic inputs overlap prior identities: {len(overlap)}")

    weight_path = Path(
        hf_hub_download(
            repo_id=MODEL_REPOSITORY,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
        )
    )
    if weight_path.stat().st_size != MODEL_WEIGHT_BYTES:
        raise RuntimeError("semantic matcher weight size mismatch")
    if file_sha256(weight_path) != MODEL_WEIGHT_SHA256:
        raise RuntimeError("semantic matcher weight checksum mismatch")

    if args.device == "auto":
        selected_device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        selected_device = args.device
    device = torch.device(selected_device)
    model, _, transform = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=str(weight_path),
        device=device,
    )
    model.eval()
    fake_features = extract_features(
        model, fake_paths, transform, device, args.batch_size
    )
    real_features = extract_features(
        model, real_paths, transform, device, args.batch_size
    )
    if args.matching_mode == "individual-image":
        individual_assignments, all_similarity = assign_unique_reals_individually(
            fake_features, real_features
        )
        assignments = None
    else:
        assignments, all_similarity = assign_unique_reals(
            fake_rows, fake_features, real_features
        )
        individual_assignments = None
    del model

    args.output.mkdir(parents=True)
    image_root = args.output / "images"
    image_root.mkdir()
    output_rows: list[dict] = []
    derivative_hashes: set[str] = set()
    match_scores = []
    by_prompt_scores: dict[str, list[float]] = defaultdict(list)
    fake_by_prompt: dict[int, list[tuple[dict, Path, str]]] = defaultdict(list)
    for row, path, digest in zip(fake_rows, fake_paths, source_hashes[: len(fake_rows)]):
        fake_by_prompt[int(row["prompt_id"])].append((row, path, digest))

    paired_items = []
    if args.matching_mode == "individual-image":
        if individual_assignments is None:
            raise AssertionError("individual assignments missing")
        for fake_index, (real_index, score) in enumerate(individual_assignments):
            fake_row = fake_rows[fake_index]
            paired_items.append(
                (
                    int(fake_row["prompt_id"]),
                    str(fake_row["generator_model"]),
                    fake_row,
                    fake_paths[fake_index],
                    source_hashes[fake_index],
                    real_index,
                    score,
                )
            )
    else:
        if assignments is None:
            raise AssertionError("centroid assignments missing")
        for prompt_id in sorted(fake_by_prompt):
            fakes = sorted(
                fake_by_prompt[prompt_id],
                key=lambda item: str(item[0]["generator_model"]),
            )
            for pair_index, ((fake_row, fake_path, fake_digest), (real_index, score)) in enumerate(
                zip(fakes, assignments[prompt_id])
            ):
                paired_items.append(
                    (
                        prompt_id,
                        f"{pair_index:02d}",
                        fake_row,
                        fake_path,
                        fake_digest,
                        real_index,
                        score,
                    )
                )

    for prompt_id, pair_suffix, fake_row, fake_path, fake_digest, real_index, score in sorted(
        paired_items, key=lambda item: (item[0], item[1])
    ):
            real_row = real_rows[real_index]
            real_path = real_paths[real_index]
            real_digest = source_hashes[len(fake_rows) + real_index]
            safe_suffix = hashlib.sha256(pair_suffix.encode()).hexdigest()[:8]
            pair_id = f"p{prompt_id:04d}-{safe_suffix}"
            match_scores.append(score)
            by_prompt_scores[str(prompt_id)].append(score)
            for label, row, source_path, source_digest in (
                (0, real_row, real_path, real_digest),
                (1, fake_row, fake_path, fake_digest),
            ):
                data, original_size = canonical_bytes(source_path)
                derivative_digest = hashlib.sha256(data).hexdigest()
                if derivative_digest in derivative_hashes:
                    raise RuntimeError(f"canonical image collision: {derivative_digest}")
                derivative_hashes.add(derivative_digest)
                destination = image_root / derivative_digest[:2] / f"{derivative_digest}.jpg"
                destination.parent.mkdir(exist_ok=True)
                destination.write_bytes(data)
                output_rows.append(
                    {
                        **row,
                        "path": str(destination.resolve()),
                        "label": label,
                        "source_image_sha256": source_digest,
                        "sha256": derivative_digest,
                        "pair_id": pair_id,
                        "semantic_prompt_id": prompt_id,
                        "paired_generator": str(fake_row["generator_model"]),
                        "semantic_match_cosine": score,
                        "semantic_matcher": MODEL_NAME,
                        "canonicalization": POLICY,
                        "canonical_width": 336,
                        "canonical_height": 336,
                        "canonical_format": "JPEG",
                        "original_width": original_size[0],
                        "original_height": original_size[1],
                        "organizer_demo_row": False,
                        "training_allowed": False,
                        "workflow_purpose": "semantic-matched-modern-audit",
                    }
                )

    output_rows.sort(key=lambda row: (row["pair_id"], int(row["label"])))
    if Counter(int(row["label"]) for row in output_rows) != {0: 144, 1: 144}:
        raise RuntimeError("balanced gate invariant failed")
    if len({row["pair_id"] for row in output_rows}) != 144:
        raise RuntimeError("pair identity invariant failed")
    manifest = args.output / "eval_semantic_matched.jsonl"
    manifest_sha256 = write_jsonl(manifest, output_rows)
    report = {
        "status": "built_not_scored",
        "purpose": "semantic-matched-modern-audit",
        "rows": len(output_rows),
        "labels": {"real": 144, "fake": 144},
        "pairs": 144,
        "prompts": sorted(fake_by_prompt),
        "generators": sorted({str(row["generator_model"]) for row in fake_rows}),
        "real_pool_rows": len(real_rows),
        "source_identity_overlap": 0,
        "unique_source_images": len(source_hashes),
        "unique_canonical_images": len(derivative_hashes),
        "organizer_demo_rows": 0,
        "training_allowed_rows": 0,
        "semantic_match": {
            "method": (
                "global unique assignment from each actual fake-image CLIP embedding"
                if args.matching_mode == "individual-image"
                else "global unique assignment to per-prompt fake-image CLIP centroids"
            ),
            "matching_mode": args.matching_mode,
            "model": MODEL_NAME,
            "implementation": f"open_clip_torch=={open_clip.__version__}",
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "weight_filename": MODEL_FILENAME,
            "weight_bytes": MODEL_WEIGHT_BYTES,
            "weight_sha256": MODEL_WEIGHT_SHA256,
            "model_license": "MIT",
            "model_license_url": "https://github.com/openai/CLIP/blob/main/LICENSE",
            "seed": SEED,
            "deterministic_algorithms": True,
            "fake_feature_sha256": hashlib.sha256(fake_features.tobytes()).hexdigest(),
            "real_feature_sha256": hashlib.sha256(real_features.tobytes()).hexdigest(),
            "selected_cosine_min": min(match_scores),
            "selected_cosine_mean": float(np.mean(match_scores)),
            "selected_cosine_max": max(match_scores),
            "pool_cosine_mean": float(np.mean(all_similarity)),
            "by_prompt": {
                prompt: {
                    "min": min(values),
                    "mean": float(np.mean(values)),
                    "max": max(values),
                }
                for prompt, values in sorted(by_prompt_scores.items())
            },
        },
        "canonicalization": POLICY,
        "manifest_sha256": manifest_sha256,
        "derivative_inventory_sha256": hashlib.sha256(
            "".join(sorted(derivative_hashes)).encode()
        ).hexdigest(),
        "elapsed_seconds": time.time() - started,
        "device": selected_device,
        "hardware": platform.platform(),
        "torch": torch.__version__,
        "open_clip": open_clip.__version__,
        "ru_maxrss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "boundary": (
            "The gate was selected without candidate detector scores and is audit-only. "
            "Independent CLIP matching reduces but cannot prove removal of all content bias."
        ),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
