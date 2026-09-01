import json
import subprocess
import sys
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "package_kaggle_subset.py"


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_package_deduplicates_shared_images_and_rewrites_paths(tmp_path: Path) -> None:
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"not-a-real-image-but-content-addressable")
    duplicate = tmp_path / "duplicate.jpg"
    duplicate.write_bytes(image.read_bytes())
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_manifest(first, [{"path": "sample.jpg", "label": 0}])
    write_manifest(second, [{"path": "duplicate.jpg", "label": 1}])
    output = tmp_path / "package.zip"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(first),
            "--manifest",
            str(second),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.with_suffix(".zip.provenance.json").read_text())
    assert report["format_version"] == 2
    assert report["source_paths"] == 2
    assert report["unique_images"] == 1
    assert report["package_sha256"]
    assert [entry["source_manifest_name"] for entry in report["manifests"]] == [
        "first.jsonl",
        "second.jsonl",
    ]
    assert all("source_manifest" not in entry for entry in report["manifests"])
    with zipfile.ZipFile(output) as archive:
        image_names = [name for name in archive.namelist() if name.startswith("images/")]
        assert len(image_names) == 1
        for manifest_name in ("manifests/first.jsonl", "manifests/second.jsonl"):
            row = json.loads(archive.read(manifest_name))
            assert row["path"].startswith("../images/")
            assert row["image_sha256"]
    assert "scanned 2/2 source paths; packaged 1 unique images" in completed.stdout


def test_package_refuses_demo_only_path(tmp_path: Path) -> None:
    forbidden = tmp_path / "demo_only"
    forbidden.mkdir()
    (forbidden / "sample.jpg").write_bytes(b"forbidden")
    manifest = tmp_path / "rows.jsonl"
    write_manifest(manifest, [{"path": "demo_only/sample.jpg", "label": 0}])

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--output", str(tmp_path / "x.zip")],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "forbidden demo-only source" in completed.stderr
