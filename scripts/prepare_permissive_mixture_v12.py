#!/usr/bin/env python3
"""Build the non-commercial-source-free v12 train and frozen evaluation manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


SEED = 20260831
FORBIDDEN_PATH_TERMS = {
    "afhq",
    "celebahq",
    "demo_only",
    "ffhq",
    "laion",
    "lsun-church",
    "val2017",
}
WILDFAKE_TRAIN_COUNTS = {
    "ADM": 650,
    "VQDM": 500,
    "BigGAN": 400,
    "styleGAN": 250,
    "starGAN": 400,
    "DDPM": 300,
    "SDv1.5-DPMSolver": 324,
}
EVAL_COUNTS = {
    "PixArt-Sigma": 250,
    "Imagen": 250,
    "GigaGAN": 178,
    "GALIP": 178,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(seed: int, namespace: str, identity: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{identity}".encode()).hexdigest()


def read_manifest(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        source = Path(row["path"])
        if not source.is_absolute():
            source = (path.parent / source).resolve()
        row["_source"] = source
        rows.append(row)
    return rows


def image_paths(root: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    return sorted(path.resolve() for path in root.rglob("*") if path.suffix.lower() in suffixes)


def choose(rows: list[dict], count: int, seed: int, namespace: str) -> list[dict]:
    if len(rows) < count:
        raise ValueError(f"{namespace}: found {len(rows)}, need {count}")
    ranked = sorted(
        rows,
        key=lambda row: stable_rank(seed, namespace, str(row["_source"])),
    )
    return ranked[:count]


def licensed(row: dict, *, dataset: str, license_id: str, license_url: str) -> dict:
    result = {key: value for key, value in row.items() if key != "_source"}
    result["path"] = str(row["_source"])
    result["dataset"] = dataset
    result["source_license"] = license_id
    result["source_license_url"] = license_url
    result["license_commercial_use_allowed"] = True
    result["training_allowed"] = True
    result["organizer_demo_row"] = False
    result["_source"] = row["_source"]
    return result


def normalize_coco_row(row: dict) -> dict:
    result = dict(row)
    result["label"] = 0
    result["real_source"] = "COCO-train2017-commercial-compatible"
    result["family"] = "diverse-real-photography"
    result["dataset"] = "COCO-train2017-commercial-compatible"
    result["source_license"] = (
        f'COCO-license-{result["source_license_id"]}:'
        f'{result["source_license_name"]}'
    )
    result["license_commercial_use_allowed"] = True
    result["training_allowed"] = True
    result["organizer_demo_row"] = False
    result["_source"] = Path(result["path"])
    return result


def cifake_rows(root: Path, label: int, count: int, seed: int) -> list[dict]:
    role = "FAKE" if label else "REAL"
    candidates = [{"_source": path, "label": label} for path in image_paths(root / role)]
    selected = choose(candidates, count, seed, f"cifake-{role.lower()}")
    output = []
    for row in selected:
        if label:
            row.update(
                fake_source="CIFAKE",
                generator="CIFAKE-Stable-Diffusion",
                family="latent-diffusion",
            )
        else:
            row.update(real_source="CIFAKE-CIFAR10", family="low-resolution-photo")
        output.append(
            licensed(
                row,
                dataset="CIFAKE",
                license_id="MIT",
                license_url="https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images/blob/e112a942abaecd02b6b1f6f646c807d56be8fb62/README.md#license",
            )
        )
    return output


def v6_group(
    rows: list[dict], label: int, key: str, value: str, count: int, seed: int
) -> list[dict]:
    eligible = []
    for row in rows:
        if int(row["label"]) != label or row.get(key) != value:
            continue
        lowered = str(row["_source"]).lower()
        if any(term in lowered for term in FORBIDDEN_PATH_TERMS):
            continue
        eligible.append(row)
    return choose(eligible, count, seed, f"v6-{key}-{value}")


def validate_and_materialize(rows: list[dict], role: str) -> tuple[list[dict], str]:
    output = []
    content_seen: dict[str, str] = {}
    for row in rows:
        path = Path(row.pop("_source")).resolve()
        lowered = str(path).lower()
        matches = sorted(term for term in FORBIDDEN_PATH_TERMS if term in lowered)
        if matches:
            raise ValueError(f"{role}: forbidden source terms {matches}: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = row.get("image_sha256") or row.get("sha256") or file_sha256(path)
        if digest in content_seen:
            raise ValueError(
                f"{role}: duplicate content {digest}: {content_seen[digest]} and {path}"
            )
        content_seen[digest] = str(path)
        row["path"] = str(path)
        row["sha256"] = digest
        row["workflow_purpose"] = role
        output.append(row)
    output.sort(key=lambda row: stable_rank(SEED, role, row["sha256"]))
    inventory = hashlib.sha256(
        "".join(f'{row["label"]}:{row["sha256"]}\n' for row in output).encode()
    ).hexdigest()
    return output, inventory


def write_manifest(path: Path, rows: list[dict]) -> str:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return file_sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", type=Path, default=Path("datasets/coco_train2017_permissive_v12/manifest.jsonl"))
    parser.add_argument("--cifake-train", type=Path, default=Path("datasets/cifake/train"))
    parser.add_argument("--cifake-test", type=Path, default=Path("datasets/cifake/test"))
    parser.add_argument("--v6-train", type=Path, default=Path("datasets/family_mixture_v6/train.jsonl"))
    parser.add_argument("--v6-eval", type=Path, default=Path("datasets/family_mixture_v6/eval_selection_perceptual_clean.jsonl"))
    parser.add_argument("--qwen-train", type=Path, default=Path("datasets/qwen_image_bench_train_candidate/manifest.jsonl"))
    parser.add_argument("--qwen-eval", type=Path, default=Path("datasets/qwen_image_bench_holdout_v2/manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("datasets/permissive_mixture_v12"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    coco = read_manifest(args.coco)
    if len(coco) != 6000:
        raise ValueError(f"expected 6000 frozen COCO rows, got {len(coco)}")
    coco_train = [normalize_coco_row(row) for row in coco[:5000]]
    coco_eval = [normalize_coco_row(row) for row in coco[5000:5600]]

    v6_train = read_manifest(args.v6_train)
    train_reals = [dict(row) for row in coco_train]
    train_reals += cifake_rows(args.cifake_train, 0, 1500, args.seed)
    sid_real = v6_group(v6_train, 0, "real_source", "SID_Set", 287, args.seed)
    train_reals += [
        licensed(
            row,
            dataset="SID_Set",
            license_id="CC-BY-4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
        )
        for row in sid_real
    ]

    train_fakes = cifake_rows(args.cifake_train, 1, 1500, args.seed)
    sid_fake = v6_group(v6_train, 1, "generator", "SID_Set-unspecified", 287, args.seed)
    train_fakes += [
        licensed(
            row,
            dataset="SID_Set",
            license_id="CC-BY-4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
        )
        for row in sid_fake
    ]
    for generator in ("FLUX.1-schnell", "Stable-Diffusion-3-Medium"):
        selected = v6_group(v6_train, 1, "generator", generator, 800, args.seed)
        train_fakes += [
            licensed(
                row,
                dataset="DiTFake",
                license_id="Apache-2.0",
                license_url="https://www.apache.org/licenses/LICENSE-2.0",
            )
            for row in selected
        ]
    qwen_train = read_manifest(args.qwen_train)
    train_fakes += [
        licensed(
            row,
            dataset="Qwen-Image-Bench",
            license_id="Apache-2.0",
            license_url="https://www.apache.org/licenses/LICENSE-2.0",
        )
        for row in qwen_train
    ]
    for generator, count in WILDFAKE_TRAIN_COUNTS.items():
        selected = v6_group(v6_train, 1, "generator", generator, count, args.seed)
        train_fakes += [
            licensed(
                row,
                dataset="WildFake",
                license_id="Apache-2.0-dataset-record",
                license_url="https://www.apache.org/licenses/LICENSE-2.0",
            )
            for row in selected
        ]

    if len(train_reals) != 6787 or len(train_fakes) != 6787:
        raise ValueError(
            f"unexpected train balance: real={len(train_reals)}, fake={len(train_fakes)}"
        )

    v6_eval = read_manifest(args.v6_eval)
    eval_reals = [dict(row) for row in coco_eval]
    eval_reals += cifake_rows(args.cifake_test, 0, 250, args.seed + 1)
    sid_eval_real = v6_group(v6_eval, 0, "real_source", "SID_Set", 150, args.seed)
    eval_reals += [
        licensed(
            row,
            dataset="SID_Set",
            license_id="CC-BY-4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
        )
        for row in sid_eval_real
    ]
    if len(eval_reals) != 1000:
        raise ValueError(f"unexpected eval real rows: {len(eval_reals)}")
    eval_fakes = []
    for generator, count in EVAL_COUNTS.items():
        selected = v6_group(v6_eval, 1, "generator", generator, count, args.seed)
        dataset = "DiTFake" if generator == "PixArt-Sigma" else "WildFake"
        licence = "Apache-2.0" if dataset == "DiTFake" else "Apache-2.0-dataset-record"
        eval_fakes += [
            licensed(
                row,
                dataset=dataset,
                license_id=licence,
                license_url="https://www.apache.org/licenses/LICENSE-2.0",
            )
            for row in selected
        ]
    qwen_eval = choose(read_manifest(args.qwen_eval), 144, args.seed, "qwen-eval")
    eval_fakes += [
        licensed(
            row,
            dataset="Qwen-Image-Bench",
            license_id="Apache-2.0",
            license_url="https://www.apache.org/licenses/LICENSE-2.0",
        )
        for row in qwen_eval
    ]
    if len(eval_fakes) != 1000:
        raise ValueError(f"unexpected eval fake rows: {len(eval_fakes)}")
    for row in eval_reals + eval_fakes:
        row["training_allowed"] = False
        row["evaluation_only"] = True

    train_rows, train_inventory = validate_and_materialize(
        train_reals + train_fakes, "v12-train"
    )
    eval_rows, eval_inventory = validate_and_materialize(
        eval_reals + eval_fakes, "v12-eval-frozen"
    )
    train_hashes = {row["sha256"] for row in train_rows}
    eval_hashes = {row["sha256"] for row in eval_rows}
    overlap = train_hashes & eval_hashes
    if overlap:
        raise ValueError(f"train/eval content overlap: {len(overlap)}")

    args.output.mkdir(parents=True)
    train_path = args.output / "train.jsonl"
    eval_path = args.output / "eval_frozen.jsonl"
    train_manifest_sha = write_manifest(train_path, train_rows)
    eval_manifest_sha = write_manifest(eval_path, eval_rows)
    report = {
        "seed": args.seed,
        "rule": "no dataset marked non-commercial; no organizer demo-only row",
        "train_rows": len(train_rows),
        "train_label_counts": dict(sorted(Counter(row["label"] for row in train_rows).items())),
        "train_real_sources": dict(sorted(Counter(row.get("real_source") for row in train_rows if row["label"] == 0).items())),
        "train_fake_generators": dict(sorted(Counter(row.get("generator") for row in train_rows if row["label"] == 1).items())),
        "train_licenses": dict(sorted(Counter(row["source_license"] for row in train_rows).items())),
        "eval_rows": len(eval_rows),
        "eval_label_counts": dict(sorted(Counter(row["label"] for row in eval_rows).items())),
        "eval_fake_generators": dict(sorted(Counter(row.get("generator") for row in eval_rows if row["label"] == 1).items())),
        "train_manifest_sha256": train_manifest_sha,
        "eval_manifest_sha256": eval_manifest_sha,
        "train_content_inventory_sha256": train_inventory,
        "eval_content_inventory_sha256": eval_inventory,
        "train_eval_content_overlap": 0,
        "organizer_demo_rows": 0,
        "noncommercial_rows": 0,
        "training_allowed": True,
    }
    (args.output / "selection.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
