#!/usr/bin/env python3
"""Build a source-aligned DDPM-train/DDIM-test split from verified WildFake files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import zipfile
from pathlib import Path


SOURCES = {
    "train_fake": {
        "archive": "datasets/wildfake/archives/DDPM.zip",
        "csv": "datasets/wildfake/metadata/ddpm.csv",
        "archive_bytes": 8_141_353_209,
        "archive_sha256": "665e0a32c0231a3a8ef3cf0fb2379431756cf504ed806653d28f3ac8fd6e5be7",
        "csv_bytes": 8_712_759,
        "csv_sha256": "4200679c4f74d8bc39a36d458c28c1930e94dceb5189efc926dad27e44ed97f2",
        "prefix": "./Diffusion_based/",
        "label": "1",
    },
    "test_fake": {
        "archive": "datasets/wildfake/archives/DDIM.zip",
        "csv": "datasets/wildfake/metadata/ddim.csv",
        "archive_bytes": 6_054_264_809,
        "archive_sha256": "fa509e0ae546d91b2edd6dad91a1efe0ae3bd5c50d31609cb1db56a31d9f6e9c",
        "csv_bytes": 7_498_439,
        "csv_sha256": "6e8f3cadca7551e57f009f0f63ca5967ffd25d6639f42a5b1b081142da74cd11",
        "prefix": "./Diffusion_based/",
        "label": "1",
    },
    "real": {
        "archive": "datasets/wildfake/archives/imagenet.zip",
        "csv": "datasets/wildfake/metadata/real_imagenet.csv",
        "archive_bytes": 1_378_959_009,
        "archive_sha256": "f80fe448c6f1dbfb9e8fd143c168d19697b1e38ee00e2d4cd040337592f515a7",
        "csv_bytes": 8_563_441,
        "csv_sha256": "31d887ea8bcf5b0aa9d6dd9a4024a98ba7283abe7f0f5c8b9464d7d63134df02",
        "prefix": "./Real/",
        "label": "0",
    },
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_verified_rows(spec: dict[str, object]) -> list[dict[str, str]]:
    archive, index = Path(str(spec["archive"])), Path(str(spec["csv"]))
    for path, size_key, hash_key in (
        (archive, "archive_bytes", "archive_sha256"),
        (index, "csv_bytes", "csv_sha256"),
    ):
        if path.stat().st_size != int(spec[size_key]):
            raise ValueError(f"unexpected size for {path}")
        observed = digest(path)
        if observed != spec[hash_key]:
            raise ValueError(f"unexpected SHA-256 for {path}: {observed}")
    with index.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["IsFake"] == spec["label"]]


def extract(
    rows: list[dict[str, str]],
    spec: dict[str, object],
    destination: Path,
) -> list[dict[str, object]]:
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(Path(str(spec["archive"]))) as source_zip:
        members = set(source_zip.namelist())
        for position, row in enumerate(rows):
            member = row["Image_path"].replace(str(spec["prefix"]), "", 1)
            if member not in members:
                raise ValueError(f"indexed archive member is missing: {member}")
            output = destination / f"{position:06d}{Path(member).suffix.lower()}"
            with source_zip.open(member) as source, output.open("wb") as target:
                shutil.copyfileobj(source, target)
            records.append({"output": str(output), "source_member": member, "row": int(row["Num"])})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("datasets/wildfake_ddpm_train_ddim_test"))
    parser.add_argument("--train-per-class", type=int, default=4096)
    parser.add_argument("--test-per-class", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit(f"refusing to mix with existing output: {args.output_root}")
    if min(args.train_per_class, args.test_per_class) < 1:
        raise SystemExit("sample counts must be positive")

    real_rows = load_verified_rows(SOURCES["real"])
    train_fake_rows = load_verified_rows(SOURCES["train_fake"])
    test_fake_rows = load_verified_rows(SOURCES["test_fake"])
    random.Random(args.seed).shuffle(real_rows)
    random.Random(args.seed + 1).shuffle(train_fake_rows)
    random.Random(args.seed + 2).shuffle(test_fake_rows)
    needed_real = args.train_per_class + args.test_per_class
    if len(real_rows) < needed_real:
        raise ValueError("not enough non-overlapping real images")

    selections = {
        "train/REAL": (real_rows[args.test_per_class:needed_real], SOURCES["real"]),
        "train/FAKE": (train_fake_rows[: args.train_per_class], SOURCES["train_fake"]),
        "test/REAL": (real_rows[: args.test_per_class], SOURCES["real"]),
        "test/FAKE": (test_fake_rows[: args.test_per_class], SOURCES["test_fake"]),
    }
    manifest = {
        "seed": args.seed,
        "design": "DDPM fake train, DDIM fake test, disjoint ImageNet real train/test",
        "train_generator": "DDPM",
        "test_generator": "DDIM",
        "splits": {},
    }
    for relative, (rows, spec) in selections.items():
        manifest["splits"][relative] = {
            "count": len(rows),
            "archive_sha256": spec["archive_sha256"],
            "index_sha256": spec["csv_sha256"],
            "files": extract(rows, spec, args.output_root / relative),
        }
    manifest_path = args.output_root / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(args.output_root), "splits": {k: len(v[0]) for k, v in selections.items()}}, indent=2))


if __name__ == "__main__":
    main()
