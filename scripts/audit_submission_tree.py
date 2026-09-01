#!/usr/bin/env python3
"""Audit the final public submission tree without mutating it."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


REQUIRED = {
    "README.md",
    "LICENSE",
    "OFFICIAL_REQUIREMENTS.md",
    "MODEL_CARD.md",
    "ROBUSTNESS_AND_ERROR_ANALYSIS.md",
    "THIRD_PARTY_NOTICES.md",
    "demo/index.html",
    "run_v12.sh",
    "SELECTED_CHECKPOINT.sha256",
    "V12_CHECKPOINT_MANIFEST.json",
    "V12_RUNNABLE_CONTRACT_RESULT.json",
    "V12_MPS_RUNNABLE_CONTRACT_RESULT.json",
    "V12_SELECTED_DEFAULT_RUN_RESULT.json",
    "NTIRE_V12_FINAL_ARBITRATION_RESULT.json",
    "V12_ERROR_ANALYSIS_RESULT.json",
    "MODEL_WEIGHTS_LICENSE.md",
    "pyproject.toml",
    "requirements-runtime.txt",
    "requirements-dev.txt",
}
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".onnx",
    ".parquet",
    ".pt",
    ".pth",
    ".safetensors",
    ".tar",
    ".tgz",
    ".zip",
}
PRIVATE_PATTERNS = {
    "absolute_user_home": re.compile("/" + r"Users/[^/\s]+/"),
    "codex_attachment": re.compile(r"\.codex/" + "attachments/"),
    "private_kaggle_notebook_url": re.compile(r"https://www\.kaggle\.com/code/"),
    "private_kaggle_version_id": re.compile(r"scriptVersionId(?:=|`?\s*[:])"),
    "account_scoped_kaggle_input": re.compile(
        "/kaggle/input/" + r"datasets/[^/\s]+/"
    ),
    "private_key_material": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "unresolved_release_placeholder": re.compile(
        r"\[" + "VERIFY_AND_INSERT" + r"\]"
    ),
}


def source_files(root: Path) -> tuple[list[Path], str]:
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        return (
            [root / item.decode() for item in result.stdout.split(b"\0") if item],
            "git_tracked_tree",
        )
    return (
        sorted(path for path in root.rglob("*") if path.is_file()),
        "history_free_filesystem_tree",
    )


def audit(root: Path) -> dict:
    paths, inventory_mode = source_files(root)
    relative = {str(path.relative_to(root)) for path in paths}
    blockers: list[dict] = []

    for missing in sorted(REQUIRED - relative):
        blockers.append({"kind": "missing_required_file", "path": missing})

    for path in paths:
        rel = str(path.relative_to(root))
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            blockers.append({"kind": "forbidden_tracked_artifact", "path": rel})
        if path.stat().st_size > 10 * 1024 * 1024:
            blockers.append(
                {
                    "kind": "tracked_file_over_10_mib",
                    "path": rel,
                    "bytes": path.stat().st_size,
                }
            )
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for name, pattern in PRIVATE_PATTERNS.items():
                if pattern.search(line):
                    blockers.append(
                        {
                            "kind": name,
                            "path": rel,
                            "line": line_number,
                        }
                    )

    executable = {}
    for name in ("run_v12.sh",):
        path = root / name
        executable[name] = path.exists() and os.access(path, os.X_OK)
        if not executable[name]:
            blockers.append({"kind": "run_script_not_executable", "path": name})

    manifest = json.loads((root / "V12_CHECKPOINT_MANIFEST.json").read_text())
    selected_mode = str(manifest["selected_runtime_mode"])
    if selected_mode not in manifest["checkpoints"]:
        blockers.append({"kind": "invalid_selected_runtime_mode", "value": selected_mode})
        selected = {"parameters": manifest["organizer_parameter_limit_exclusive"]}
    else:
        selected = manifest["checkpoints"][selected_mode]
    parameters = int(selected["parameters"])
    limit = int(manifest["organizer_parameter_limit_exclusive"])
    under_limit = parameters < limit
    if not under_limit:
        blockers.append(
            {
                "kind": "parameter_limit_violation",
                "observed": parameters,
                "exclusive_limit": limit,
            }
        )
    distribution_urls_present = bool(selected.get("distribution_url"))
    if not distribution_urls_present:
        blockers.append({"kind": "missing_public_checkpoint_urls"})

    blocker_kinds = [record["kind"] for record in blockers]
    forbidden_count = sum(
        kind in {"forbidden_tracked_artifact", "tracked_file_over_10_mib"}
        for kind in blocker_kinds
    )
    private_locator_count = sum(
        kind
        in {
            "absolute_user_home",
            "codex_attachment",
            "private_kaggle_notebook_url",
            "private_kaggle_version_id",
            "account_scoped_kaggle_input",
        }
        for kind in blocker_kinds
    )
    private_key_count = blocker_kinds.count("private_key_material")
    placeholder_count = blocker_kinds.count("unresolved_release_placeholder")
    source_safety_blockers = {
        "missing_required_file",
        "forbidden_tracked_artifact",
        "tracked_file_over_10_mib",
        "absolute_user_home",
        "codex_attachment",
        "private_kaggle_notebook_url",
        "private_kaggle_version_id",
        "account_scoped_kaggle_input",
        "private_key_material",
        "run_script_not_executable",
        "parameter_limit_violation",
        "invalid_selected_runtime_mode",
    }
    source_tree_safe = not any(kind in source_safety_blockers for kind in blocker_kinds)
    if source_tree_safe and blockers:
        status = "source_tree_safe_but_distribution_blocked"
    elif source_tree_safe:
        status = "source_tree_ready"
    else:
        status = "source_tree_blocked"

    return {
        "status": status,
        "purpose": "Final public source-tree and deliverable audit.",
        "inventory_mode": inventory_mode,
        "tracked_files": len(paths),
        "required_files_present": not bool(REQUIRED - relative),
        "run_scripts_executable": executable,
        "selected_runtime_mode": selected_mode,
        "selected_total_parameters": parameters,
        "organizer_parameter_limit_exclusive": limit,
        "under_parameter_limit": under_limit,
        "forbidden_tracked_weights_archives_or_large_files": forbidden_count,
        "flagged_private_or_personal_locators_in_current_tree": private_locator_count,
        "flagged_private_key_material": private_key_count,
        "unresolved_release_placeholders": placeholder_count,
        "distribution_urls_present": distribution_urls_present,
        "blockers": blockers,
        "passes": not blockers,
        "reason": (
            "The source tree is mechanically ready, but external public links "
            "and immutable checkpoint distribution are not yet verified."
            if source_tree_safe and blockers
            else "No current-tree blocker was found."
            if source_tree_safe
            else "At least one current-tree safety or completeness blocker remains."
        ),
        "history_boundary": (
            "The public main branch is the reviewed history-free source release."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.root.resolve())
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if not report["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
