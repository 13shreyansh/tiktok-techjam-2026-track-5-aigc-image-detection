"""Acquire a prompt-matched subset from the official Qwen Image Bench.

The source contains 1,000 prompts rendered by 18 recent image generators.  The
same deterministic prompt IDs are selected for every generator, so generator
comparisons do not silently change prompt content.  Audit and candidate-
training prompt IDs can be kept disjoint; organizer demo resources are never
read.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image


REPOSITORY = "Qwen/Qwen-Image-Bench"
REVISION = "d2493deb153b020cf169c7e3f57d15e4dd697038"
API_URL = (
    f"https://huggingface.co/api/datasets/{REPOSITORY}/revision/{REVISION}?blobs=true"
)
SOURCE_URL = f"https://huggingface.co/datasets/{REPOSITORY}"
MODELS = (
    "FLUX.2-pro",
    "FLUX.2_max",
    "GLM-Image",
    "GPT-Image-1",
    "GPT-Image-1.5",
    "HunyuanImage-3.0",
    "Imagen-4.0",
    "Imagen-4.0-Ultra",
    "Qwen-Image",
    "Qwen-Image-2.0-pro",
    "Qwen-Image-2512",
    "Seedream-4.0",
    "Seedream-4.5",
    "Seedream-5.0",
    "gpt-image-2",
    "kling_v2_1",
    "nano-banana-2.0",
    "nano-banana-pro",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "track5-audit/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def inspect_image(path: Path) -> dict:
    with Image.open(path) as image:
        image.load()
        if image.width < 1 or image.height < 1:
            raise RuntimeError(f"invalid image geometry: {path}")
        return {
            "format": image.format,
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
        }


def load_prompt_selection(
    path: Path, expected_count: int, excluded_prompt_ids: set[int]
) -> list[int]:
    report = json.loads(path.read_text(encoding="utf-8"))
    prompt_ids = sorted(int(value) for value in report["selected_prompt_ids"])
    if len(prompt_ids) != expected_count or len(set(prompt_ids)) != expected_count:
        raise RuntimeError(
            f"prompt selection count mismatch: {len(prompt_ids)} != {expected_count}"
        )
    overlap = set(prompt_ids) & excluded_prompt_ids
    if overlap:
        raise RuntimeError(f"selected prompt IDs overlap exclusions: {sorted(overlap)}")
    if any(prompt_id < 1 or prompt_id > 1000 for prompt_id in prompt_ids):
        raise RuntimeError("selected prompt ID outside 1..1000")
    return prompt_ids


def download(entry: dict, output_root: Path) -> dict:
    relative = Path(entry["rfilename"])
    destination = output_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = int(entry["size"])
    if destination.is_file() and destination.stat().st_size == expected_bytes:
        observed_sha256 = sha256(destination)
        return {
            **entry,
            "path": str(destination.resolve()),
            "sha256": observed_sha256,
            **inspect_image(destination),
        }

    encoded = urllib.parse.quote(entry["rfilename"], safe="/")
    url = (
        f"https://huggingface.co/datasets/{REPOSITORY}/resolve/"
        f"{REVISION}/{encoded}?download=true"
    )
    temporary = destination.with_name(destination.name + ".partial")
    for attempt in range(1, 5):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "track5-audit/1"}
            )
            with urllib.request.urlopen(request, timeout=180) as response, temporary.open(
                "wb"
            ) as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            if temporary.stat().st_size != expected_bytes:
                raise RuntimeError(
                    f"size mismatch for {entry['rfilename']}: "
                    f"{temporary.stat().st_size} != {expected_bytes}"
                )
            temporary.replace(destination)
            observed_sha256 = sha256(destination)
            return {
                **entry,
                "path": str(destination.resolve()),
                "sha256": observed_sha256,
                **inspect_image(destination),
            }
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/qwen_image_bench_audit"),
    )
    parser.add_argument("--per-model", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--prompt-selection", type=Path)
    parser.add_argument(
        "--purpose", choices=("audit", "train-candidate"), default="audit"
    )
    args = parser.parse_args()
    if not 1 <= args.per_model <= 1000:
        raise SystemExit("--per-model must be between 1 and 1000")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    started = time.time()
    metadata = fetch_json(API_URL)
    if metadata.get("sha") != REVISION:
        raise RuntimeError(f"dataset revision changed: {metadata.get('sha')}")
    if metadata.get("private") or metadata.get("gated"):
        raise RuntimeError("expected an ungated public dataset")
    if metadata.get("cardData", {}).get("license") != "apache-2.0":
        raise RuntimeError("expected the pinned Apache-2.0 dataset declaration")

    excluded_prompt_ids: set[int] = set()
    for excluded_manifest in args.exclude_manifest:
        excluded_prompt_ids.update(
            int(json.loads(line)["prompt_id"])
            for line in excluded_manifest.read_text().splitlines()
            if line
        )
    if args.prompt_selection:
        prompt_ids = load_prompt_selection(
            args.prompt_selection, args.per_model, excluded_prompt_ids
        )
        prompt_selection_source = {
            "path": str(args.prompt_selection),
            "sha256": sha256(args.prompt_selection),
        }
    else:
        candidate_prompt_ids = [
            prompt_id
            for prompt_id in range(1, 1001)
            if prompt_id not in excluded_prompt_ids
        ]
        if len(candidate_prompt_ids) < args.per_model:
            raise RuntimeError("too few prompt IDs remain after exclusions")
        prompt_ids = sorted(
            random.Random(args.seed).sample(candidate_prompt_ids, args.per_model)
        )
        prompt_selection_source = None
    siblings = [
        sibling
        for sibling in metadata.get("siblings", [])
        if str(sibling.get("rfilename", "")).startswith("images/")
    ]
    by_model_prompt: dict[tuple[str, int], list[dict]] = {}
    for sibling in siblings:
        parts = str(sibling["rfilename"]).split("/")
        if len(parts) != 3 or parts[1] not in MODELS:
            continue
        prompt_id = int(parts[2].split("_", 1)[0])
        by_model_prompt.setdefault((parts[1], prompt_id), []).append(sibling)

    selected = []
    for model in MODELS:
        for prompt_id in prompt_ids:
            candidates = by_model_prompt.get((model, prompt_id), [])
            if len(candidates) != 1:
                raise RuntimeError(
                    f"expected one image for {model} prompt {prompt_id}: {candidates}"
                )
            sibling = candidates[0]
            selected.append(
                {
                    "rfilename": sibling["rfilename"],
                    "size": int(sibling.get("size") or sibling.get("lfs", {}).get("size")),
                    "blob_id": sibling.get("blobId")
                    or sibling.get("lfs", {}).get("oid"),
                    "generator_model": model,
                    "prompt_id": prompt_id,
                }
            )

    source_inventory = json.dumps(selected, sort_keys=True, separators=(",", ":"))
    source_inventory_sha256 = hashlib.sha256(source_inventory.encode()).hexdigest()
    args.output_root.mkdir(parents=True, exist_ok=True)
    completed: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download, entry, args.output_root): entry
            for entry in selected
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            completed.append(future.result())
            if index % 25 == 0 or index == len(selected):
                print(
                    json.dumps({"downloaded_or_verified": index, "total": len(selected)}),
                    flush=True,
                )

    completed.sort(key=lambda item: (item["generator_model"], item["prompt_id"]))
    manifest_path = args.output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as stream:
        for item in completed:
            local_path = Path(item["path"])
            stream.write(
                json.dumps(
                    {
                        "path": str(local_path.relative_to(args.output_root.resolve())),
                        "label": 1,
                        "generator": item["generator_model"],
                        "generator_model": item["generator_model"],
                        "family": "frontier-2026-image-generation",
                        "prompt_id": item["prompt_id"],
                        "image_sha256": item["sha256"],
                        "source_path": item["rfilename"],
                        "source_blob_id": item["blob_id"],
                        "workflow_purpose": args.purpose,
                        "original_format": item["format"],
                        "original_mode": item["mode"],
                        "original_width": item["width"],
                        "original_height": item["height"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    report = {
        "source_url": SOURCE_URL,
        "api_url": API_URL,
        "revision": REVISION,
        "license": "Apache-2.0 declared by pinned dataset card",
        "workflow_purpose": args.purpose,
        "audit_only": args.purpose == "audit",
        "training_allowed_by_this_workflow": args.purpose == "train-candidate",
        "seed": args.seed,
        "prompt_ids": prompt_ids,
        "prompt_selection_source": prompt_selection_source,
        "excluded_prompt_ids": sorted(excluded_prompt_ids),
        "models": list(MODELS),
        "per_model": args.per_model,
        "images": len(completed),
        "bytes": sum(int(item["size"]) for item in completed),
        "source_inventory_sha256": source_inventory_sha256,
        "manifest_sha256": sha256(manifest_path),
        "elapsed_seconds": time.time() - started,
    }
    report_path = args.output_root / "acquisition.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
