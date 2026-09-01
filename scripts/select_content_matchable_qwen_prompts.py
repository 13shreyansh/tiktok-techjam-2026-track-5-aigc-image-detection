#!/usr/bin/env python3
"""Select unused Qwen prompts that fresh real photographs can represent.

The selection uses only prompt text and an audit-only fresh COCO real pool. A
pinned full CLIP model ranks prompts by the mean of their 18 best real-image
matches. K-means over the best 96 prompts then preserves scene diversity. No
candidate detector, fake image or detector score is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset


SOURCE_BYTES = 185276698
SOURCE_SHA256 = "3def4dbb901bc75a13b755864e288a6e8f9a50010c1310129f57a7a84f44c005"
SOURCE_URL = (
    "https://huggingface.co/datasets/Qwen/Qwen-Image-Bench/resolve/"
    "d2493deb153b020cf169c7e3f57d15e4dd697038/"
    "qwen_image_bench_hf_v0518.jsonl?download=true"
)
MODEL_NAME = "RN50-quickgelu"
MODEL_REPOSITORY = "timm/resnet50_clip.openai"
MODEL_REVISION = "ec3d92cf63a5f9d591f0d611b736895966c73076"
MODEL_FILENAME = "open_clip_model.safetensors"
MODEL_WEIGHT_BYTES = 408291932
MODEL_WEIGHT_SHA256 = "da0baa37fb2211eee5729ab69ed587eb10d48f5b7076ceeed01552fc3d4cb4ee"
SEED = 20260901
TOP_REAL_MATCHES = 18
SHORTLIST = 96
SELECTED = 8


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as opened:
            return self.transform(opened.convert("RGB"))


def select_diverse_prompts(
    prompt_ids: list[int],
    text_features: np.ndarray,
    matchability: np.ndarray,
    shortlist_size: int = SHORTLIST,
    selected_count: int = SELECTED,
) -> tuple[list[int], dict[int, int], list[int]]:
    if len(prompt_ids) != len(text_features) or len(prompt_ids) != len(matchability):
        raise RuntimeError("prompt feature length mismatch")
    if len(prompt_ids) < shortlist_size or shortlist_size < selected_count:
        raise RuntimeError("insufficient prompt candidates")
    order = np.argsort(-matchability, kind="stable")
    shortlist_indices = order[:shortlist_size]
    clusters = KMeans(
        n_clusters=selected_count,
        random_state=SEED,
        n_init=20,
    ).fit_predict(text_features[shortlist_indices])
    selected_indices = []
    cluster_by_prompt: dict[int, int] = {}
    for cluster in range(selected_count):
        members = shortlist_indices[np.where(clusters == cluster)[0]]
        if not len(members):
            raise RuntimeError(f"empty prompt cluster: {cluster}")
        best = sorted(
            members.tolist(), key=lambda index: (-matchability[index], prompt_ids[index])
        )[0]
        selected_indices.append(best)
        cluster_by_prompt[prompt_ids[best]] = cluster
    selected_ids = sorted(prompt_ids[index] for index in selected_indices)
    return selected_ids, cluster_by_prompt, shortlist_indices.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.source_jsonl.stat().st_size != SOURCE_BYTES:
        raise RuntimeError("Qwen prompt source size mismatch")
    if file_sha256(args.source_jsonl) != SOURCE_SHA256:
        raise RuntimeError("Qwen prompt source checksum mismatch")

    started = time.time()
    excluded_prompt_ids = {
        int(row["prompt_id"])
        for manifest in args.exclude_manifest
        for row in read_rows(manifest)
    }
    prompts = []
    with args.source_jsonl.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            prompt_id = int(row["ID"])
            if prompt_id not in excluded_prompt_ids:
                prompts.append(
                    {
                        "prompt_id": prompt_id,
                        "prompt_en": str(row["prompt_en"]),
                        "prompt_cn": str(row["prompt_cn"]),
                    }
                )
    prompts.sort(key=lambda row: row["prompt_id"])

    real_rows = read_rows(args.real_manifest)
    if any(int(row["label"]) != 0 for row in real_rows):
        raise RuntimeError("real pool label contract failed")
    real_paths = [Path(row["path"]) for row in real_rows]
    if any(not path.is_file() for path in real_paths):
        raise RuntimeError("real pool image missing")

    weight_path = Path(
        hf_hub_download(
            repo_id=MODEL_REPOSITORY,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
        )
    )
    if weight_path.stat().st_size != MODEL_WEIGHT_BYTES:
        raise RuntimeError("CLIP weight size mismatch")
    if file_sha256(weight_path) != MODEL_WEIGHT_SHA256:
        raise RuntimeError("CLIP weight checksum mismatch")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=str(weight_path), device="cpu"
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()

    image_features = []
    with torch.inference_mode():
        for batch in DataLoader(
            ImageDataset(real_paths, preprocess),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        ):
            image_features.append(model.encode_image(batch, normalize=True).float())
        text_features = []
        for start in range(0, len(prompts), 64):
            tokens = tokenizer(
                [row["prompt_en"] for row in prompts[start : start + 64]]
            )
            text_features.append(model.encode_text(tokens, normalize=True).float())
    image_array = torch.cat(image_features).numpy()
    text_array = torch.cat(text_features).numpy()
    similarity = text_array @ image_array.T
    top_values = np.partition(
        similarity, -TOP_REAL_MATCHES, axis=1
    )[:, -TOP_REAL_MATCHES:]
    matchability = np.mean(top_values, axis=1)
    prompt_ids = [row["prompt_id"] for row in prompts]
    selected_ids, cluster_by_prompt, shortlist_indices = select_diverse_prompts(
        prompt_ids, text_array, matchability
    )
    prompt_by_id = {row["prompt_id"]: row for row in prompts}
    rank_by_index = {
        index: rank
        for rank, index in enumerate(
            np.argsort(-matchability, kind="stable").tolist(), start=1
        )
    }
    index_by_id = {prompt_id: index for index, prompt_id in enumerate(prompt_ids)}
    selected_rows = []
    for prompt_id in selected_ids:
        index = index_by_id[prompt_id]
        selected_rows.append(
            {
                **prompt_by_id[prompt_id],
                "cluster": cluster_by_prompt[prompt_id],
                "matchability_rank": rank_by_index[index],
                "mean_top18_cosine": float(matchability[index]),
                "minimum_top18_cosine": float(np.min(top_values[index])),
                "maximum_top18_cosine": float(np.max(top_values[index])),
            }
        )
    shortlist_ids = [prompt_ids[index] for index in shortlist_indices]
    report = {
        "status": "selected_without_candidate_scores",
        "selected_prompt_ids": selected_ids,
        "selected_prompts": selected_rows,
        "eligible_prompt_count": len(prompts),
        "excluded_prompt_ids": sorted(excluded_prompt_ids),
        "selection": {
            "matchability": "mean CLIP cosine to 18 best fresh COCO real images",
            "shortlist_size": SHORTLIST,
            "shortlist_prompt_ids": shortlist_ids,
            "diversity": "k-means over normalized CLIP text embeddings",
            "clusters": SELECTED,
            "seed": SEED,
        },
        "real_pool": {
            "rows": len(real_rows),
            "manifest_sha256": file_sha256(args.real_manifest),
            "training_allowed": False,
            "organizer_demo_rows": 0,
        },
        "prompt_source": {
            "url": SOURCE_URL,
            "bytes": SOURCE_BYTES,
            "sha256": SOURCE_SHA256,
        },
        "semantic_matcher": {
            "model": MODEL_NAME,
            "implementation": f"open_clip_torch=={open_clip.__version__}",
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "filename": MODEL_FILENAME,
            "weight_bytes": MODEL_WEIGHT_BYTES,
            "weight_sha256": MODEL_WEIGHT_SHA256,
            "license": "MIT",
            "license_url": "https://github.com/openai/CLIP/blob/main/LICENSE",
        },
        "elapsed_seconds": time.time() - started,
        "device": "cpu",
        "hardware": platform.platform(),
        "torch": torch.__version__,
        "ru_maxrss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "boundary": (
            "Prompt selection reads only prompt text and fresh real images. "
            "It does not read fake images, candidate detectors or detector scores."
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
