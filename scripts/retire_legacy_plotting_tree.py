#!/usr/bin/env python3
"""Retire archive-verified legacy plotting data and generated outputs.

The command is intentionally conservative.  It defaults to a read-only dry
run, audits every file proposed for removal against the verified pre-migration
archive, and refuses the *entire* operation if any counterpart or checksum is
missing or different.  Passing ``--execute`` is the only way to remove files.

The retained layout is deliberately small:

* everything below ``data/plotting/``;
* the purpose directories below ``figures/``; and
* ``figures/README.md``.

The audited root-level plotting inputs and scratch outputs listed in
``ROOT_LEGACY_PATHS`` are also retired after the same archive verification.

The archive and canonical plotting data are never removal targets.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Iterable, Mapping


ARCHIVE_NAME = "pre_plot_reorg_2026-08-26"
PURPOSE_DIRECTORIES = (
    "paper",
    "supplement",
    "slides",
    "poster",
    "grant",
    "diagnostic",
)
FIGURES_README = "README.md"
IGNORED_METADATA_NAMES = {".DS_Store"}
ROOT_LEGACY_PATHS = (
    "H1N1_HA.defattr",
    "H1N1_sasa.csv",
    "H3N2_HA.defattr",
    "H3N2_sasa.csv",
    "h3-h1-selection.pdf",
    "h3n2_distances_to_the_future.csv",
    "not-in-db.dat",
    "selection_coeff_align_pos_H3-H1.csv",
)
FIGURES_README_TEXT = """# Generated figures

Figure outputs are grouped by their intended use:

- `paper/`
- `supplement/`
- `slides/`
- `poster/`
- `grant/`
- `diagnostic/`

These are generated artifacts. Their recipes, inputs, style profiles, and
canonical output paths are recorded in `../figure_manifest.yaml`.

The pre-reorganization flat outputs are preserved under
`../archive/pre_plot_reorg_2026-08-26/original/figures/` for visual comparison
and recovery.
"""


class SafetyError(RuntimeError):
    """Raised when retirement cannot be proven safe."""


@dataclass(frozen=True)
class ManifestRecord:
    relative_path: str
    byte_count: int
    source_sha256: str
    archive_sha256: str
    verified_equal: bool


@dataclass(frozen=True)
class RemovalEntry:
    """One verified filesystem object included in a retirement plan."""

    relative_path: str
    kind: str
    byte_count: int
    sha256: str
    mode: int
    device: int
    inode: int
    mtime_ns: int

    @property
    def depth(self) -> int:
        return len(PurePosixPath(self.relative_path).parts)


@dataclass(frozen=True)
class RetirementPlan:
    repo: Path
    archive_root: Path
    removal_roots: tuple[str, ...]
    entries: tuple[RemovalEntry, ...]
    directories_to_create: tuple[str, ...]
    create_figures_readme: bool

    @property
    def file_count(self) -> int:
        return sum(entry.kind == "file" for entry in self.entries)

    @property
    def directory_count(self) -> int:
        return sum(entry.kind == "directory" for entry in self.entries)

    @property
    def byte_count(self) -> int:
        return sum(
            entry.byte_count for entry in self.entries if entry.kind == "file"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _paths_overlap(first: Path, second: Path) -> bool:
    return _is_within(first, second) or _is_within(second, first)


def _normal_manifest_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise SafetyError(f"Archive manifest contains an unsafe path: {value!r}")
    return path.as_posix()


def _parse_verified(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def load_verified_manifest(archive_root: Path) -> dict[str, ManifestRecord]:
    """Load an archive manifest only after checking its verification summary."""

    summary_path = archive_root / "archive_summary.json"
    manifest_path = archive_root / "file_manifest.csv"
    original_root = archive_root / "original"
    if not archive_root.is_dir() or archive_root.is_symlink():
        raise SafetyError(f"Archive root is absent or is not a plain directory: {archive_root}")
    if not original_root.is_dir() or original_root.is_symlink():
        raise SafetyError(f"Archive source tree is absent or unsafe: {original_root}")
    if not summary_path.is_file() or summary_path.is_symlink():
        raise SafetyError(f"Archive verification summary is absent: {summary_path}")
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise SafetyError(f"Archive file manifest is absent: {manifest_path}")

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetyError(f"Could not read archive verification summary: {exc}") from exc
    if summary.get("all_files_verified_equal") is not True:
        raise SafetyError("Archive summary does not certify that all files were equal")

    required = {
        "original_path",
        "bytes",
        "source_sha256",
        "archive_sha256",
        "verified_equal",
    }
    records: dict[str, ManifestRecord] = {}
    try:
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not required.issubset(reader.fieldnames or []):
                missing = sorted(required.difference(reader.fieldnames or []))
                raise SafetyError(
                    f"Archive manifest is missing required columns: {', '.join(missing)}"
                )
            for line_number, row in enumerate(reader, start=2):
                relative = _normal_manifest_path(row["original_path"])
                if relative in records:
                    raise SafetyError(f"Duplicate archive manifest path: {relative}")
                try:
                    byte_count = int(row["bytes"])
                except (TypeError, ValueError) as exc:
                    raise SafetyError(
                        f"Invalid byte count at archive manifest line {line_number}"
                    ) from exc
                if byte_count < 0:
                    raise SafetyError(
                        f"Negative byte count at archive manifest line {line_number}"
                    )
                source_digest = row["source_sha256"].strip().lower()
                archive_digest = row["archive_sha256"].strip().lower()
                for label, digest in (
                    ("source", source_digest),
                    ("archive", archive_digest),
                ):
                    if len(digest) != 64 or any(
                        character not in "0123456789abcdef" for character in digest
                    ):
                        raise SafetyError(
                            f"Invalid {label} checksum at archive manifest line {line_number}"
                        )
                records[relative] = ManifestRecord(
                    relative_path=relative,
                    byte_count=byte_count,
                    source_sha256=source_digest,
                    archive_sha256=archive_digest,
                    verified_equal=_parse_verified(row["verified_equal"]),
                )
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise SafetyError(f"Could not read archive file manifest: {exc}") from exc

    try:
        expected_files = int(summary["file_count"])
        expected_bytes = int(summary["total_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SafetyError("Archive summary has invalid file or byte totals") from exc
    if expected_files != len(records):
        raise SafetyError(
            "Archive summary and manifest file counts differ "
            f"({expected_files} != {len(records)})"
        )
    manifest_bytes = sum(record.byte_count for record in records.values())
    if expected_bytes != manifest_bytes:
        raise SafetyError(
            "Archive summary and manifest byte totals differ "
            f"({expected_bytes} != {manifest_bytes})"
        )
    return records


def _plain_directory(path: Path, label: str) -> None:
    if not path.is_dir() or path.is_symlink():
        raise SafetyError(f"{label} must be an existing plain directory: {path}")


def _relative(repo: Path, path: Path) -> str:
    try:
        return path.relative_to(repo).as_posix()
    except ValueError as exc:
        raise SafetyError(f"Path escapes repository: {path}") from exc


def _entry_from_stat(
    relative: str,
    kind: str,
    byte_count: int,
    digest: str,
    status: os.stat_result,
) -> RemovalEntry:
    return RemovalEntry(
        relative_path=relative,
        kind=kind,
        byte_count=byte_count,
        sha256=digest,
        mode=status.st_mode,
        device=status.st_dev,
        inode=status.st_ino,
        mtime_ns=status.st_mtime_ns,
    )


def _audit_candidate(
    repo: Path,
    path: Path,
    original_root: Path,
    manifest: Mapping[str, ManifestRecord],
) -> list[RemovalEntry]:
    """Recursively verify a candidate without following symbolic links."""

    relative = _relative(repo, path)
    try:
        status = path.lstat()
    except OSError as exc:
        raise SafetyError(f"Could not inspect removal candidate {relative}: {exc}") from exc
    archived = original_root / PurePosixPath(relative)

    if stat.S_ISLNK(status.st_mode):
        raise SafetyError(
            f"Refusing symbolic-link removal because link metadata was not checksummed: {relative}"
        )
    if stat.S_ISDIR(status.st_mode):
        if not archived.is_dir() or archived.is_symlink():
            raise SafetyError(f"Archived directory counterpart is absent or unsafe: {relative}")
        entries = [_entry_from_stat(relative, "directory", 0, "", status)]
        try:
            children = sorted(path.iterdir(), key=lambda child: child.name)
        except OSError as exc:
            raise SafetyError(f"Could not list removal candidate {relative}: {exc}") from exc
        for child in children:
            entries.extend(_audit_candidate(repo, child, original_root, manifest))
        return entries
    if not stat.S_ISREG(status.st_mode):
        raise SafetyError(f"Unsupported filesystem object in removal set: {relative}")

    record = manifest.get(relative)
    if record is None:
        raise SafetyError(f"Archive manifest has no counterpart for: {relative}")
    if not record.verified_equal:
        raise SafetyError(f"Archive manifest does not mark this file verified: {relative}")
    if record.source_sha256 != record.archive_sha256:
        raise SafetyError(f"Recorded source and archive checksums differ: {relative}")
    if not archived.is_file() or archived.is_symlink():
        raise SafetyError(f"Archived file counterpart is absent or unsafe: {relative}")
    if status.st_size != record.byte_count or archived.stat().st_size != record.byte_count:
        raise SafetyError(f"Live, archived, and recorded sizes differ: {relative}")

    archived_digest = sha256(archived)
    if archived_digest != record.archive_sha256:
        raise SafetyError(f"Archived file checksum no longer matches its manifest: {relative}")
    live_digest = sha256(path)
    if live_digest != record.source_sha256:
        raise SafetyError(f"Live file changed after the archive was created: {relative}")
    return [_entry_from_stat(relative, "file", status.st_size, live_digest, status)]


def _validate_retained_layout(repo: Path) -> tuple[tuple[str, ...], bool]:
    data_root = repo / "data"
    canonical = data_root / "plotting"
    _plain_directory(data_root, "Data root")
    _plain_directory(canonical, "Canonical plotting data")

    figures_root = repo / "figures"
    if figures_root.exists() or figures_root.is_symlink():
        _plain_directory(figures_root, "Figures root")
    directories_to_create: list[str] = []
    for name in PURPOSE_DIRECTORIES:
        path = figures_root / name
        if path.exists() or path.is_symlink():
            _plain_directory(path, f"Figure purpose directory {name}")
        else:
            directories_to_create.append(_relative(repo, path))

    readme = figures_root / FIGURES_README
    if readme.exists() or readme.is_symlink():
        if not readme.is_file() or readme.is_symlink():
            raise SafetyError(f"Figures README is not a plain file: {readme}")
        create_readme = False
    else:
        create_readme = True
    return tuple(directories_to_create), create_readme


def _candidate_roots(repo: Path) -> tuple[Path, ...]:
    roots: list[Path] = []
    for relative in ROOT_LEGACY_PATHS:
        path = repo / relative
        if path.exists() or path.is_symlink():
            roots.append(path)

    data_root = repo / "data"
    for child in sorted(data_root.iterdir(), key=lambda path: path.name):
        if child.name != "plotting" and child.name not in IGNORED_METADATA_NAMES:
            roots.append(child)

    figures_root = repo / "figures"
    if figures_root.exists():
        retained = set(PURPOSE_DIRECTORIES) | {FIGURES_README} | IGNORED_METADATA_NAMES
        for child in sorted(figures_root.iterdir(), key=lambda path: path.name):
            if child.name not in retained:
                roots.append(child)
    return tuple(roots)


def build_plan(repo: Path, archive_root: Path) -> RetirementPlan:
    """Build an immutable, fully verified retirement plan."""

    repo = repo.resolve()
    archive_root = archive_root.resolve()
    _plain_directory(repo, "Repository root")
    data_root = repo / "data"
    figures_root = repo / "figures"
    canonical = data_root / "plotting"
    if _paths_overlap(archive_root, data_root) or _paths_overlap(archive_root, figures_root):
        raise SafetyError("Archive must not overlap data/ or figures/")

    directories_to_create, create_readme = _validate_retained_layout(repo)
    manifest = load_verified_manifest(archive_root)
    original_root = archive_root / "original"
    roots = _candidate_roots(repo)
    entries: list[RemovalEntry] = []
    for root in roots:
        entries.extend(_audit_candidate(repo, root, original_root, manifest))

    canonical_resolved = canonical.resolve()
    archive_resolved = archive_root.resolve()
    for entry in entries:
        candidate = repo / PurePosixPath(entry.relative_path)
        candidate_resolved = candidate.resolve(strict=False)
        if _is_within(candidate_resolved, canonical_resolved):
            raise SafetyError(
                "Internal error: canonical data entered removal plan: "
                f"{entry.relative_path}"
            )
        if _is_within(candidate_resolved, archive_resolved):
            raise SafetyError(
                f"Internal error: archive entered removal plan: {entry.relative_path}"
            )

    return RetirementPlan(
        repo=repo,
        archive_root=archive_root,
        removal_roots=tuple(_relative(repo, path) for path in roots),
        entries=tuple(entries),
        directories_to_create=directories_to_create,
        create_figures_readme=create_readme,
    )


def _same_filesystem_object(path: Path, entry: RemovalEntry) -> bool:
    try:
        status = path.lstat()
    except OSError:
        return False
    return (
        status.st_mode == entry.mode
        and status.st_dev == entry.device
        and status.st_ino == entry.inode
        and status.st_size == entry.byte_count
        and status.st_mtime_ns == entry.mtime_ns
    )


def _delete_verified_entries(plan: RetirementPlan) -> None:
    """Delete only the exact files in the twice-audited plan.

    Directories are removed with ``rmdir`` rather than recursive deletion.  If
    an unverified file appears after the audit, its directory therefore remains
    instead of the new file being swept into the operation.
    """

    files = sorted(
        (entry for entry in plan.entries if entry.kind == "file"),
        key=lambda entry: entry.relative_path,
    )
    for entry in files:
        path = plan.repo / PurePosixPath(entry.relative_path)
        if not _same_filesystem_object(path, entry) or sha256(path) != entry.sha256:
            raise SafetyError(
                "Removal candidate changed immediately before deletion: "
                f"{entry.relative_path}"
            )

    for entry in files:
        (plan.repo / PurePosixPath(entry.relative_path)).unlink()

    directories = sorted(
        (entry for entry in plan.entries if entry.kind == "directory"),
        key=lambda entry: (-entry.depth, entry.relative_path),
    )
    for entry in directories:
        path = plan.repo / PurePosixPath(entry.relative_path)
        try:
            path.rmdir()
        except OSError as exc:
            raise SafetyError(
                f"Directory was not empty after verified files were removed: {entry.relative_path}"
            ) from exc


def execute_plan(plan: RetirementPlan) -> None:
    """Re-audit and then apply a retirement plan."""

    repeated = build_plan(plan.repo, plan.archive_root)
    if repeated != plan:
        raise SafetyError("Filesystem or archive changed between the two complete audits")
    _delete_verified_entries(plan)

    figures_root = plan.repo / "figures"
    figures_root.mkdir(exist_ok=True)
    for relative in plan.directories_to_create:
        (plan.repo / PurePosixPath(relative)).mkdir(parents=True, exist_ok=False)
    if plan.create_figures_readme:
        (figures_root / FIGURES_README).write_text(
            FIGURES_README_TEXT,
            encoding="utf-8",
            newline="\n",
        )


def run_retirement(
    repo: Path,
    archive_root: Path,
    *,
    execute: bool = False,
) -> RetirementPlan:
    """Audit a retirement and optionally execute it."""

    plan = build_plan(repo, archive_root)
    if execute:
        execute_plan(plan)
    return plan


def _human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def plan_as_dict(plan: RetirementPlan, *, execute: bool) -> dict[str, object]:
    return {
        "mode": "executed" if execute else "dry-run",
        "safety_checks": "passed",
        "archive_root": str(plan.archive_root),
        "preserved_data": "data/plotting",
        "preserved_figure_directories": list(PURPOSE_DIRECTORIES),
        "removal_root_count": len(plan.removal_roots),
        "removal_file_count": plan.file_count,
        "removal_directory_count": plan.directory_count,
        "removal_bytes": plan.byte_count,
        "removal_roots": list(plan.removal_roots),
        "directories_to_create": list(plan.directories_to_create),
        "create_figures_readme": plan.create_figures_readme,
    }


def print_summary(plan: RetirementPlan, *, execute: bool, verbose: bool) -> None:
    action = "Removed" if execute else "Would remove"
    print(f"Mode: {'EXECUTED' if execute else 'DRY RUN (no files changed)'}")
    print("Safety checks: passed")
    print(f"Verified archive: {plan.archive_root}")
    print("Preserved canonical data: data/plotting/")
    print(f"{action}: {plan.file_count:,} files and {plan.directory_count:,} directories")
    print(f"Logical size: {plan.byte_count:,} bytes ({_human_bytes(plan.byte_count)})")
    print(f"Removal roots: {len(plan.removal_roots):,}")
    for relative in plan.removal_roots:
        print(f"  - {relative}")
    if plan.directories_to_create:
        verb = "Created" if execute else "Would create"
        print(f"{verb} figure purpose directories:")
        for relative in plan.directories_to_create:
            print(f"  - {relative}/")
    if plan.create_figures_readme:
        print(f"{'Created' if execute else 'Would create'} figures/{FIGURES_README}")
    if verbose:
        print("Verified removal entries:")
        for entry in plan.entries:
            print(f"  - {entry.kind}: {entry.relative_path}")
    if not execute:
        print(
            "No changes made. Re-run with --execute to apply this exact policy "
            "after a fresh audit."
        )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
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
        help=f"Verified archive root (defaults to archive/{ARCHIVE_NAME})",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply the fully verified plan; omission always means dry-run",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the summary as JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="List every verified file and directory in the plan",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo.resolve()
    archive_root = (
        args.archive_root.resolve()
        if args.archive_root is not None
        else repo / "archive" / ARCHIVE_NAME
    )
    try:
        plan = run_retirement(repo, archive_root, execute=args.execute)
    except SafetyError as exc:
        if args.json:
            print(
                json.dumps(
                    {
                        "mode": "refused",
                        "safety_checks": "failed",
                        "error": str(exc),
                    },
                    indent=2,
                )
            )
        else:
            print(f"REFUSED: {exc}", file=sys.stderr)
            print("No unverified removal was authorized.", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(plan_as_dict(plan, execute=args.execute), indent=2))
    else:
        print_summary(plan, execute=args.execute, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
