#!/usr/bin/env python3
"""Export a training checkpoint as a smaller inference-only FP16 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_checkpoint(
    source: Path,
    destination: Path,
    expected_source_sha256: str | None = None,
) -> dict:
    source_sha256 = file_sha256(source)
    if (
        expected_source_sha256 is not None
        and source_sha256 != expected_source_sha256.lower()
    ):
        raise ValueError(
            "source checkpoint SHA-256 mismatch: "
            f"{source_sha256} != {expected_source_sha256.lower()}"
        )
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    required = {"state_dict", "model_name", "image_size"}
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"checkpoint is missing required keys: {missing}")
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("checkpoint state_dict is empty or invalid")

    converted = {}
    floating_parameters = 0
    total_parameters = 0
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state_dict entry is not a tensor: {name}")
        value = tensor.detach().cpu().contiguous()
        total_parameters += value.numel()
        if value.is_floating_point():
            floating_parameters += value.numel()
            value = value.to(torch.float16)
        converted[name] = value

    exported = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"state_dict", "optimizer_state_dict", "scaler_state_dict"}
    }
    exported.update(
        {
            "format_version": 2,
            "state_dict": converted,
            "storage_dtype": "float16",
            "source_checkpoint_sha256": source_sha256,
            "total_state_dict_parameters": total_parameters,
            "floating_state_dict_parameters": floating_parameters,
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(exported, temporary)
    temporary.replace(destination)
    report = {
        "source": str(source),
        "destination": str(destination),
        "source_bytes": source.stat().st_size,
        "destination_bytes": destination.stat().st_size,
        "source_sha256": exported["source_checkpoint_sha256"],
        "destination_sha256": file_sha256(destination),
        "total_state_dict_parameters": total_parameters,
        "floating_state_dict_parameters": floating_parameters,
        "storage_dtype": "float16",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args()
    report = export_checkpoint(
        args.source,
        args.destination,
        expected_source_sha256=args.expected_source_sha256,
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
