#!/usr/bin/env python3
"""Acquire deterministic WildFake members without downloading a whole ZIP.

ModelScope's immutable WildFake ZIP objects support HTTP byte ranges. This
utility reads the remote central directory, deterministically selects named
generator groups, downloads only those members, and records every ZIP CRC and
member path. The expected whole-archive SHA-256 remains provenance evidence;
it cannot be recomputed from a partial download.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import zipfile
from pathlib import Path, PurePosixPath

import requests
from remotezip import RemoteZip
from PIL import Image, UnidentifiedImageError


def parse_group(value: str) -> tuple[str, str, str]:
    parts = value.split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("group must be GENERATOR:FAMILY:PREFIX")
    return parts[0], parts[1], parts[2].rstrip("/") + "/"


def remote_size(url: str) -> tuple[int, dict[str, str]]:
    response = requests.get(
        url,
        headers={"Range": "bytes=0-0", "User-Agent": "track5-wildfake-subset/1"},
        allow_redirects=True,
        timeout=60,
    )
    response.raise_for_status()
    if response.status_code != 206:
        raise SystemExit(f"remote server ignored byte range: HTTP {response.status_code}")
    match = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
    if not match:
        raise SystemExit("missing or invalid Content-Range header")
    evidence = {
        key: response.headers[key]
        for key in ("ETag", "Last-Modified", "x-linked-etag")
        if key in response.headers
    }
    response.close()
    return int(match.group(1)), evidence


def group_seed(seed: int, generator: str, prefix: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{generator}\0{prefix}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def stratified_windows(
    candidates: list[zipfile.ZipInfo], count: int, blocks: int, seed: int
) -> list[zipfile.ZipInfo]:
    """Select spread-out contiguous windows to minimize HTTP round trips."""
    if count > len(candidates):
        raise ValueError("requested more members than are available")
    if count == len(candidates):
        return list(candidates)
    blocks = min(blocks, count)
    base, remainder = divmod(count, blocks)
    selected = []
    rng = random.Random(seed)
    for index in range(blocks):
        quota = base + (1 if index < remainder else 0)
        lower = index * len(candidates) // blocks
        upper = (index + 1) * len(candidates) // blocks
        start = rng.randint(lower, upper - quota)
        selected.extend(candidates[start : start + quota])
    return selected


def fetch_range(
    session: requests.Session,
    url: str,
    start: int,
    end: int,
    total: int,
    destination,
) -> int:
    response = session.get(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "track5-wildfake-subset/1",
        },
        allow_redirects=True,
        stream=True,
        timeout=(60, 180),
    )
    response.raise_for_status()
    expected_range = f"bytes {start}-{end}/{total}"
    if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
        raise SystemExit(
            f"invalid range response for {start}-{end}: "
            f"HTTP {response.status_code}, {response.headers.get('Content-Range')}"
        )
    destination.seek(start)
    written = 0
    for chunk in response.iter_content(1024 * 1024):
        if chunk:
            destination.write(chunk)
            written += len(chunk)
    response.close()
    expected_bytes = end - start + 1
    if written != expected_bytes:
        raise SystemExit(f"short range response: {written} != {expected_bytes}")
    return written


def safe_relative(member: str, prefix: str) -> Path:
    relative = PurePosixPath(member).relative_to(PurePosixPath(prefix))
    if relative.is_absolute() or ".." in relative.parts or not relative.name:
        raise SystemExit(f"unsafe archive member: {member}")
    return Path(*relative.parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--archive-sha256")
    parser.add_argument("--archive-bytes", required=True, type=int)
    parser.add_argument("--label", type=int, choices=(0, 1), default=1)
    parser.add_argument("--source-name", default="WildFake")
    parser.add_argument("--group", required=True, action="append", type=parse_group)
    parser.add_argument("--per-group", type=int, default=1024)
    parser.add_argument(
        "--reserve-per-group",
        type=int,
        default=16,
        help="deterministic extra candidates used when source members are not decodable",
    )
    parser.add_argument("--blocks-per-group", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        type=Path,
        default=[],
        help="exclude archive_member values already retained by an earlier shard",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_root}")
    observed_bytes, headers = remote_size(args.url)
    if observed_bytes != args.archive_bytes:
        raise SystemExit(
            f"remote size mismatch: observed {observed_bytes}, expected {args.archive_bytes}"
        )

    excluded_members: set[str] = set()
    for exclude_manifest in args.exclude_manifest:
        with exclude_manifest.open(encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                member = row.get("archive_member")
                if not member:
                    raise SystemExit(
                        f"missing archive_member in exclusion manifest: {exclude_manifest}"
                    )
                excluded_members.add(str(member))

    args.output_root.mkdir(parents=True)
    manifest_path = args.output_root / "manifest.jsonl"
    report_groups = []
    rows = []
    try:
        with RemoteZip(args.url) as archive:
            all_infos = sorted(archive.infolist(), key=lambda info: info.header_offset)
            files = [info for info in all_infos if not info.is_dir()]
            start_dir = archive.start_dir
            selected_by_group = []
            for generator, family, prefix in args.group:
                candidates = sorted(
                    (
                        info
                        for info in files
                        if info.filename.startswith(prefix)
                        and info.filename not in excluded_members
                    ),
                    key=lambda info: info.header_offset,
                )
                requested = args.per_group + args.reserve_per_group
                if len(candidates) < requested:
                    raise SystemExit(
                        f"{generator} has {len(candidates)} members, fewer than {requested}"
                    )
                selected = stratified_windows(
                    candidates,
                    requested,
                    args.blocks_per_group,
                    group_seed(args.seed, generator, prefix),
                )
                selected_by_group.append((generator, family, prefix, candidates, selected))

        selected_infos = {
            (info.filename, info.header_offset): info
            for _, _, _, _, selected in selected_by_group
            for info in selected
        }
        global_positions = {
            (info.filename, info.header_offset): index for index, info in enumerate(all_infos)
        }
        ranges = []
        for info in sorted(selected_infos.values(), key=lambda item: item.header_offset):
            position = global_positions[(info.filename, info.header_offset)]
            end = (
                all_infos[position + 1].header_offset
                if position + 1 < len(all_infos)
                else start_dir
            )
            if ranges and info.header_offset == ranges[-1][1]:
                ranges[-1] = (ranges[-1][0], end)
            else:
                ranges.append((info.header_offset, end))

        sparse_path = args.output_root / ".partial_sparse_archive.zip"
        fetched_bytes = 0
        with requests.Session() as session, sparse_path.open("wb") as sparse:
            sparse.seek(args.archive_bytes - 1)
            sparse.write(b"\0")
            fetched_bytes += fetch_range(
                session,
                args.url,
                start_dir,
                args.archive_bytes - 1,
                args.archive_bytes,
                sparse,
            )
            for start, end in ranges:
                fetched_bytes += fetch_range(
                    session,
                    args.url,
                    start,
                    end - 1,
                    args.archive_bytes,
                    sparse,
                )

        with zipfile.ZipFile(sparse_path) as archive:
            for generator, family, prefix, candidates, selected in selected_by_group:
                selection_hash = hashlib.sha256()
                group_bytes = 0
                retained = 0
                rejected = []
                for info in selected:
                    if retained == args.per_group:
                        break
                    relative = safe_relative(info.filename, prefix)
                    data = archive.read(info.filename)
                    if len(data) != info.file_size:
                        raise SystemExit(f"member size mismatch: {info.filename}")
                    try:
                        with Image.open(io.BytesIO(data)) as image:
                            image.verify()
                    except (UnidentifiedImageError, OSError, SyntaxError) as error:
                        rejected.append(
                            {
                                "archive_member": info.filename,
                                "archive_crc32": f"{info.CRC:08x}",
                                "archive_bytes": info.file_size,
                                "reason": f"{type(error).__name__}: {error}",
                            }
                        )
                        continue
                    destination = args.output_root / "images" / generator / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
                    group_bytes += len(data)
                    retained += 1
                    selection_hash.update(
                        f"{info.filename}\t{info.CRC:08x}\t{info.file_size}\n".encode()
                    )
                    provenance = (
                        {"fake_source": args.source_name, "generator": generator}
                        if args.label == 1
                        else {"real_source": generator}
                    )
                    rows.append(
                        {
                            "path": str(destination.relative_to(args.output_root)),
                            "label": args.label,
                            **provenance,
                            "family": family,
                            "archive_member": info.filename,
                            "archive_crc32": f"{info.CRC:08x}",
                            "archive_revision": args.revision,
                        }
                    )
                if retained != args.per_group:
                    raise SystemExit(
                        f"{generator}: only {retained} decodable members from "
                        f"{len(selected)} candidates"
                    )
                report_groups.append(
                    {
                        "generator": generator,
                        "family": family,
                        "prefix": prefix,
                        "available_members": len(candidates),
                        "selected_members": retained,
                        "candidate_members_downloaded": len(selected),
                        "rejected_members": rejected,
                        "selected_bytes": group_bytes,
                        "selection_inventory_sha256": selection_hash.hexdigest(),
                    }
                )
        sparse_path.unlink()
        rows.sort(
            key=lambda row: (
                row.get("generator", row.get("real_source", "unknown")),
                row["archive_member"],
            )
        )
        manifest_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        report = {
            "source_url": args.url,
            "revision": args.revision,
            "whole_archive_expected_sha256": args.archive_sha256,
            "whole_archive_expected_bytes": args.archive_bytes,
            "whole_archive_downloaded": False,
            "remote_headers": headers,
            "seed": args.seed,
            "excluded_manifests": [str(path) for path in args.exclude_manifest],
            "excluded_archive_members": len(excluded_members),
            "blocks_per_group": args.blocks_per_group,
            "remote_data_ranges": len(ranges) + 1,
            "remote_bytes_fetched": fetched_bytes,
            "groups": report_groups,
            "manifest_rows": len(rows),
        }
        (args.output_root / "acquisition.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
    except BaseException:
        # An incomplete directory is deliberately left visible for diagnosis;
        # the overwrite guard prevents it from being mistaken for a valid run.
        raise


if __name__ == "__main__":
    main()
