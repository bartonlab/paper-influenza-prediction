#!/usr/bin/env python3
"""Create and verify the pre-migration plotting archive.

The archive is deliberately local and ignored by Git.  It preserves the current
working copies (including untracked files) rather than reconstructing them from
Git, because this repository's working tree contains substantial unpublished
plotting data and code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable


ARCHIVE_NAME = "pre_plot_reorg_2026-08-26"

# These paths comprise the plotting workspace and its known root-level inputs.
# Generated cache folders and manuscript drafts are intentionally out of scope.
DEFAULT_SOURCES = (
    Path("data"),
    Path("figures"),
    Path("figures.ipynb"),
    Path("figures.py"),
    Path("mplot.py"),
    Path("notebooks"),
    Path("src"),
    Path(".ipynb_checkpoints"),
    Path("data-processing-bkup.ipynb"),
    Path("H1N1_HA.defattr"),
    Path("H1N1_sasa.csv"),
    Path("H3N2_HA.defattr"),
    Path("H3N2_sasa.csv"),
    Path("h3-h1-selection.pdf"),
    Path("h3n2_distances_to_the_future.csv"),
    Path("not-in-db.dat"),
    Path("selection_coeff_align_pos_H3-H1.csv"),
    Path(".gitignore"),
    Path("README.md"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_git(repo: Path, *args: str) -> bytes:
    command = ["git", *args]
    result = subprocess.run(
        command,
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        joined = " ".join(command)
        raise RuntimeError(f"{joined} failed with exit code {result.returncode}")
    return result.stdout


def iter_regular_files(root: Path) -> Iterable[Path]:
    for directory, names, files in os.walk(root):
        names.sort()
        files.sort()
        base = Path(directory)
        for name in files:
            path = base / name
            if path.is_file():
                yield path


def copy_source(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # macOS cp -c creates an APFS clone when possible, avoiding an unnecessary
    # second physical 2-GB allocation while preserving independent file paths.
    command = ["cp", "-ac", str(source), str(destination)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Could not archive {source}: {detail}")


def capture_git_state(repo: Path, git_state: Path) -> None:
    git_state.mkdir(parents=True, exist_ok=False)
    records = {
        "head.txt": ("rev-parse", "HEAD"),
        "branch.txt": ("branch", "--show-current"),
        "status.txt": ("status", "--porcelain=v1", "--untracked-files=all"),
        "diff_worktree.patch": ("diff", "--binary", "--", "."),
        "diff_index.patch": ("diff", "--cached", "--binary", "--", "."),
        "tracked_files.txt": ("ls-files",),
    }
    for filename, args in records.items():
        (git_state / filename).write_bytes(run_git(repo, *args))


def inspect_csv(path: Path) -> tuple[int | None, list[str] | None, str | None]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            rows = sum(1 for _ in reader)
        return rows, header, None
    except (UnicodeDecodeError, csv.Error, OSError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def write_manifests(repo: Path, archive_root: Path, sources: tuple[Path, ...]) -> None:
    original_root = archive_root / "original"
    manifest_path = archive_root / "file_manifest.csv"
    mapping_path = archive_root / "path_mapping.csv"
    sums_path = archive_root / "SHA256SUMS"
    schema_path = archive_root / "schema_inventory.csv"

    rows: list[dict[str, object]] = []
    for archived in iter_regular_files(original_root):
        relative = archived.relative_to(original_root)
        source = repo / relative
        archive_hash = sha256(archived)
        source_hash = sha256(source) if source.is_file() else ""
        rows.append(
            {
                "original_path": relative.as_posix(),
                "archive_path": archived.relative_to(repo).as_posix(),
                "bytes": archived.stat().st_size,
                "source_sha256": source_hash,
                "archive_sha256": archive_hash,
                "verified_equal": source_hash == archive_hash,
            }
        )

    fieldnames = [
        "original_path",
        "archive_path",
        "bytes",
        "source_sha256",
        "archive_sha256",
        "verified_equal",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "original_path",
                "archive_path",
                "canonical_path",
                "relationship",
                "migration_status",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["original_path"],
                    row["archive_path"],
                    "",
                    "byte_preserving_snapshot",
                    "archived_snapshot",
                ]
            )

    with sums_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            relative_archive = Path(str(row["archive_path"])).relative_to(archive_root.relative_to(repo))
            handle.write(f"{row['archive_sha256']}  {relative_archive.as_posix()}\n")

    # CSV schemas are useful for the active plotting data.  Private GISAID CSVs
    # are excluded from row counting; their bytes and hashes remain in the main
    # manifest and their contents remain untouched.
    with schema_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["original_path", "data_rows", "columns_json", "error"])
        for row in rows:
            relative = Path(str(row["original_path"]))
            if relative.suffix.lower() != ".csv" or relative.parts[:2] == ("data", "GISAID"):
                continue
            count, header, error = inspect_csv(original_root / relative)
            writer.writerow(
                [
                    relative.as_posix(),
                    "" if count is None else count,
                    "" if header is None else json.dumps(header, ensure_ascii=False),
                    "" if error is None else error,
                ]
            )

    missing = [path.as_posix() for path in sources if not (repo / path).exists()]
    summary = {
        "archive_name": archive_root.name,
        "created_for_migration_date": "2026-08-26",
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_files_verified_equal": all(bool(row["verified_equal"]) for row in rows),
        "source_roots": [path.as_posix() for path in sources],
        "missing_optional_sources": missing,
    }
    (archive_root / "archive_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if not summary["all_files_verified_equal"]:
        raise RuntimeError("At least one archived file differs from its source")


def create_archive(repo: Path, archive_root: Path) -> None:
    if archive_root.exists():
        raise FileExistsError(
            f"Archive already exists: {archive_root}. Refusing to overwrite it."
        )

    archive_root.mkdir(parents=True)
    original_root = archive_root / "original"
    original_root.mkdir()
    capture_git_state(repo, archive_root / "git_state")

    present_sources = tuple(path for path in DEFAULT_SOURCES if (repo / path).exists())
    for relative in present_sources:
        copy_source(repo / relative, original_root / relative)

    write_manifests(repo, archive_root, DEFAULT_SOURCES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to this script's parent repository)",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help=f"Archive destination (defaults to archive/{ARCHIVE_NAME})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    archive_root = (
        args.archive_root.resolve()
        if args.archive_root is not None
        else repo / "archive" / ARCHIVE_NAME
    )
    try:
        archive_root.relative_to(repo)
    except ValueError:
        print("Archive root must be inside the repository", file=sys.stderr)
        return 2

    create_archive(repo, archive_root)
    print(archive_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
