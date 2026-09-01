import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


SCRIPT = Path("scripts/materialize_canonical_v12.py")


def save(path: Path, size: tuple[int, int], fmt: str) -> None:
    Image.new("RGB", size, color=(20, 80, 150)).save(path, format=fmt)


def manifest(path: Path, image: Path, label: int, training: bool) -> None:
    path.write_text(
        json.dumps(
            {
                "path": str(image),
                "label": label,
                "organizer_demo_row": False,
                "training_allowed": training,
            }
        )
        + "\n"
    )


def test_canonicalization_erases_source_shape_and_container(tmp_path: Path) -> None:
    real = tmp_path / "real.png"
    fake = tmp_path / "fake.webp"
    save(real, (80, 40), "PNG")
    save(fake, (45, 90), "WEBP")
    train = tmp_path / "train.jsonl"
    evaluate = tmp_path / "eval.jsonl"
    manifest(train, real, 0, True)
    manifest(evaluate, fake, 1, False)
    output = tmp_path / "canonical"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--train",
            str(train),
            "--eval",
            str(evaluate),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [
        json.loads((output / "train.jsonl").read_text()),
        json.loads((output / "eval_frozen.jsonl").read_text()),
    ]
    for row in rows:
        with Image.open(row["path"]) as image:
            assert image.format == "JPEG"
            assert image.size == (336, 336)
        assert row["canonicalization"].endswith("jpeg_q96_subsampling0")
    report = json.loads((output / "canonicalization.json").read_text())
    assert report["train_eval_overlap"] == 0
    assert report["organizer_demo_rows"] == 0
