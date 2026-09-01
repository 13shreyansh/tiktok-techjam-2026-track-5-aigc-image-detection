from pathlib import Path

import json

from scripts.prepare_cifake_lowres_repair_block import (
    read_exclusions,
    select_candidates,
    sha256_file,
)


def test_select_candidates_is_hash_ordered_and_excludes_paths(tmp_path: Path):
    from PIL import Image

    paths = []
    for index in range(4):
        path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (32, 32), (index, index, index)).save(path)
        paths.append(path)
    selected = select_candidates(paths, {paths[0].resolve()}, set(), 2)
    assert len(selected) == 2
    assert selected == sorted(selected)
    assert paths[0].resolve() not in {path for _, path in selected}


def test_read_exclusions_hashes_rows_without_declared_hash(tmp_path: Path):
    from PIL import Image

    image = tmp_path / "source.jpg"
    Image.new("RGB", (32, 32), (17, 18, 19)).save(image)
    manifest = tmp_path / "exclude.jsonl"
    manifest.write_text(json.dumps({"path": image.name}) + "\n")

    excluded_paths, excluded_hashes = read_exclusions([manifest])

    assert excluded_paths == {image.resolve()}
    assert excluded_hashes == {sha256_file(image)}
