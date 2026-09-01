from pathlib import Path

import pytest
import torch

from scripts.export_inference_checkpoint import export_checkpoint


def test_export_checkpoint_converts_floating_state_and_preserves_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pt"
    destination = tmp_path / "destination.pt"
    torch.save(
        {
            "format_version": 1,
            "model_name": "test-model",
            "image_size": 16,
            "state_dict": {
                "weight": torch.tensor([[1.25, -2.5]], dtype=torch.float32),
                "counter": torch.tensor([3], dtype=torch.int64),
            },
            "optimizer_state_dict": {"must": "be removed"},
        },
        source,
    )

    report = export_checkpoint(source, destination)
    exported = torch.load(destination, map_location="cpu", weights_only=True)

    assert exported["format_version"] == 2
    assert exported["model_name"] == "test-model"
    assert exported["state_dict"]["weight"].dtype == torch.float16
    assert exported["state_dict"]["counter"].dtype == torch.int64
    assert "optimizer_state_dict" not in exported
    assert report["total_state_dict_parameters"] == 3
    assert report["floating_state_dict_parameters"] == 2
    assert len(report["destination_sha256"]) == 64


def test_export_checkpoint_rejects_unexpected_source_hash(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    destination = tmp_path / "destination.pt"
    torch.save(
        {
            "model_name": "test-model",
            "image_size": 16,
            "state_dict": {"weight": torch.ones(1)},
        },
        source,
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        export_checkpoint(
            source,
            destination,
            expected_source_sha256="0" * 64,
        )
    assert not destination.exists()
