#!/usr/bin/env python3
"""Acquire a checksum-pinned synthetic-only Community Forensics shard subset.

The source Parquet file is large, so it remains in the ignored Hugging Face
cache.  Only a small, deterministic, decode-verified selection is retained.
Real rows are unconditionally rejected; by default we also retain only fake
images prompted from LAION records, avoiding organizer-prohibited COCO pixels
and even avoiding COCO-derived prompts as an extra conservative boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError


REPO_ID = "OwensLab/CommunityForensics-Small"
REVISION = "6c539a534c07917307c381f5af4053c6091b5278"
DEFAULT_FILE = "data/HFCF_small_0.parquet"
DEFAULT_BYTES = 1_231_926_042
DEFAULT_SHA256 = "0ce98c9b4f66eca160939982fe7aac84253af7d135485e5bc83ca8425cfe220c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_slug(value: str) -> str:
    readable = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")[:48]
    suffix = hashlib.sha256(value.encode()).hexdigest()[:12]
    return f"{readable or 'model'}-{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filename", default=DEFAULT_FILE)
    parser.add_argument("--expected-bytes", type=int, default=DEFAULT_BYTES)
    parser.add_argument("--expected-sha256", default=DEFAULT_SHA256)
    parser.add_argument("--per-model", type=int, default=4)
    parser.add_argument("--prompt-source", default="LAION")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/community_forensics_small_shard0_fakes"),
    )
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_root}")

    parquet_path = Path(
        hf_hub_download(
            REPO_ID,
            args.filename,
            repo_type="dataset",
            revision=REVISION,
        )
    )
    observed_bytes = parquet_path.stat().st_size
    observed_sha256 = sha256(parquet_path)
    if observed_bytes != args.expected_bytes or observed_sha256 != args.expected_sha256:
        raise SystemExit(
            f"source mismatch: bytes={observed_bytes}, sha256={observed_sha256}"
        )

    args.output_root.mkdir(parents=True)
    selected_counts: Counter[str] = Counter()
    source_models: set[str] = set()
    skipped = Counter()
    rows = []
    columns = [
        "image_name",
        "format",
        "image_data",
        "model_name",
        "nsfw_flag",
        "real_source",
        "subset",
        "label",
        "architecture",
    ]
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(batch_size=64, columns=columns):
        for record in batch.to_pylist():
            if int(record["label"]) != 1:
                skipped["non_fake"] += 1
                continue
            model_name = str(record["model_name"])
            source_models.add(model_name)
            if bool(record["nsfw_flag"]):
                skipped["nsfw"] += 1
                continue
            if str(record["real_source"]).casefold() != args.prompt_source.casefold():
                skipped["other_prompt_source"] += 1
                continue
            if selected_counts[model_name] >= args.per_model:
                skipped["per_model_cap"] += 1
                continue
            data = bytes(record["image_data"])
            try:
                with Image.open(io.BytesIO(data)) as image:
                    image.verify()
                    decoded_format = str(image.format or record["format"]).lower()
            except (UnidentifiedImageError, OSError, SyntaxError) as error:
                skipped[f"decode:{type(error).__name__}"] += 1
                continue
            digest = hashlib.sha256(data).hexdigest()
            suffix = {"jpeg": ".jpg", "jpg": ".jpg", "png": ".png", "webp": ".webp"}.get(
                decoded_format, ".img"
            )
            model_slug = safe_slug(model_name)
            destination = args.output_root / "images" / model_slug / f"{digest}{suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            selected_counts[model_name] += 1
            rows.append(
                {
                    "path": str(destination.relative_to(args.output_root)),
                    "label": 1,
                    "fake_source": "CommunityForensics-Small",
                    "generator": "CommunityForensics-Systematic-LatDiff-breadth",
                    "generator_model": model_name,
                    "family": str(record["architecture"]),
                    "prompt_source": str(record["real_source"]),
                    "source_subset": str(record["subset"]),
                    "source_image_name": str(record["image_name"]),
                    "image_sha256": digest,
                }
            )

    missing = sorted(model for model in source_models if selected_counts[model] < args.per_model)
    if missing:
        raise SystemExit(
            f"{len(missing)} source models yielded fewer than {args.per_model} eligible images"
        )
    rows.sort(key=lambda row: (row["generator_model"], row["image_sha256"]))
    manifest = args.output_root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    canonical_models = "\n".join(sorted(source_models)) + "\n"
    report = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "license": "CC-BY-NC-SA-4.0; per-generator licences remain applicable",
        "source_file": args.filename,
        "source_bytes": observed_bytes,
        "source_sha256": observed_sha256,
        "source_rows": parquet.metadata.num_rows,
        "source_models": len(source_models),
        "source_model_inventory_sha256": hashlib.sha256(canonical_models.encode()).hexdigest(),
        "selection": {
            "label": 1,
            "nsfw_flag": False,
            "prompt_source": args.prompt_source,
            "per_model": args.per_model,
            "selected_rows": len(rows),
            "selected_models": len(selected_counts),
        },
        "skipped": dict(sorted(skipped.items())),
        "forbidden_organizer_demo_pixels_present": False,
        "warning": (
            "This adds many latent-diffusion model variants, not many independent "
            "generator architectures; keep architecture-held-out gates decisive."
        ),
    }
    (args.output_root / "acquisition.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
