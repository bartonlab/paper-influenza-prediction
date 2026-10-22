"""Safety checks for the archive-backed plotting-tree retirement."""

from __future__ import annotations

from contextlib import redirect_stdout
import csv
import hashlib
import importlib.util
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "retire_legacy_plotting_tree.py"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


class RetirementScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("retire_legacy_plotting_tree", SCRIPT)
        assert spec is not None and spec.loader is not None
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        spec.loader.exec_module(cls.module)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name) / "repository"
        (self.repo / "data" / "plotting").mkdir(parents=True)
        (self.repo / "data" / "plotting" / "canonical.csv").write_text(
            "id,value\n1,2\n", encoding="utf-8"
        )
        (self.repo / "data" / "legacy" / "nested").mkdir(parents=True)
        (self.repo / "data" / "legacy" / "old.csv").write_text(
            "old,data\n", encoding="utf-8"
        )
        (self.repo / "data" / "legacy" / "nested" / "more.dat").write_bytes(
            b"more legacy data\n"
        )
        (self.repo / "data" / "root-old.dat").write_bytes(b"root legacy\n")
        (self.repo / "figures" / "slides").mkdir(parents=True)
        (self.repo / "figures" / "slides" / "keep.pdf").write_bytes(b"new output")
        (self.repo / "figures" / "old.pdf").write_bytes(b"old output\n")
        (self.repo / "figures" / "old-group").mkdir()
        (self.repo / "figures" / "old-group" / "plot.pdf").write_bytes(
            b"old nested output\n"
        )
        (self.repo / "H1N1_sasa.csv").write_text("site,sasa\n1,2\n", encoding="utf-8")
        self.archive = self.repo / "archive" / "snapshot"
        self._write_archive(
            (
                "data/legacy/old.csv",
                "data/legacy/nested/more.dat",
                "data/root-old.dat",
                "figures/old.pdf",
                "figures/old-group/plot.pdf",
                "H1N1_sasa.csv",
            )
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_archive(self, relative_files: tuple[str, ...]) -> None:
        if self.archive.exists():
            shutil.rmtree(self.archive)
        original = self.archive / "original"
        original.mkdir(parents=True)
        rows: list[dict[str, object]] = []
        for relative in relative_files:
            source = self.repo / relative
            target = original / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            checksum = digest(source)
            rows.append(
                {
                    "original_path": relative,
                    "archive_path": f"archive/snapshot/original/{relative}",
                    "bytes": source.stat().st_size,
                    "source_sha256": checksum,
                    "archive_sha256": checksum,
                    "verified_equal": True,
                }
            )
        with (self.archive / "file_manifest.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_files_verified_equal": True,
        }
        (self.archive / "archive_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )

    def test_default_run_is_a_complete_read_only_plan(self) -> None:
        archive_before = {
            path.relative_to(self.archive): digest(path)
            for path in self.archive.rglob("*")
            if path.is_file()
        }
        plan = self.module.run_retirement(self.repo, self.archive)

        self.assertEqual(plan.file_count, 6)
        self.assertEqual(plan.directory_count, 3)
        self.assertEqual(
            plan.byte_count,
            sum(
                (self.repo / relative).stat().st_size
                for relative in (
                    "data/legacy/old.csv",
                    "data/legacy/nested/more.dat",
                    "data/root-old.dat",
                    "figures/old.pdf",
                    "figures/old-group/plot.pdf",
                    "H1N1_sasa.csv",
                )
            ),
        )
        self.assertTrue((self.repo / "data" / "legacy" / "old.csv").exists())
        self.assertTrue((self.repo / "figures" / "old.pdf").exists())
        self.assertTrue((self.repo / "data" / "plotting" / "canonical.csv").exists())
        self.assertTrue((self.repo / "figures" / "slides" / "keep.pdf").exists())
        archive_after = {
            path.relative_to(self.archive): digest(path)
            for path in self.archive.rglob("*")
            if path.is_file()
        }
        self.assertEqual(archive_before, archive_after)

    def test_execute_removes_only_verified_legacy_and_builds_output_layout(self) -> None:
        archive_before = {
            path.relative_to(self.archive): digest(path)
            for path in self.archive.rglob("*")
            if path.is_file()
        }
        self.module.run_retirement(self.repo, self.archive, execute=True)

        self.assertFalse((self.repo / "data" / "legacy").exists())
        self.assertFalse((self.repo / "data" / "root-old.dat").exists())
        self.assertFalse((self.repo / "figures" / "old.pdf").exists())
        self.assertFalse((self.repo / "figures" / "old-group").exists())
        self.assertFalse((self.repo / "H1N1_sasa.csv").exists())
        self.assertTrue((self.repo / "data" / "plotting" / "canonical.csv").exists())
        self.assertEqual(
            (self.repo / "figures" / "slides" / "keep.pdf").read_bytes(),
            b"new output",
        )
        for name in self.module.PURPOSE_DIRECTORIES:
            self.assertTrue((self.repo / "figures" / name).is_dir())
        self.assertTrue((self.repo / "figures" / "README.md").is_file())
        archive_after = {
            path.relative_to(self.archive): digest(path)
            for path in self.archive.rglob("*")
            if path.is_file()
        }
        self.assertEqual(archive_before, archive_after)

    def test_changed_live_file_refuses_whole_operation(self) -> None:
        changed = self.repo / "data" / "legacy" / "old.csv"
        changed.write_text("changed after archive\n", encoding="utf-8")

        with self.assertRaisesRegex(self.module.SafetyError, "differ|changed after the archive"):
            self.module.run_retirement(self.repo, self.archive, execute=True)

        self.assertTrue(changed.exists())
        self.assertTrue((self.repo / "figures" / "old.pdf").exists())
        self.assertFalse((self.repo / "figures" / "README.md").exists())

    def test_missing_archive_manifest_record_refuses_whole_operation(self) -> None:
        extra = self.repo / "figures" / "unrecorded.pdf"
        extra.write_bytes(b"not archived")

        with self.assertRaisesRegex(self.module.SafetyError, "no counterpart"):
            self.module.run_retirement(self.repo, self.archive, execute=True)

        self.assertTrue(extra.exists())
        self.assertTrue((self.repo / "data" / "legacy" / "old.csv").exists())

    def test_changed_archive_file_refuses_whole_operation(self) -> None:
        archived = self.archive / "original" / "figures" / "old.pdf"
        archived.write_bytes(b"archive was altered")

        with self.assertRaisesRegex(self.module.SafetyError, "differ|Archived file checksum"):
            self.module.run_retirement(self.repo, self.archive, execute=True)

        self.assertTrue((self.repo / "figures" / "old.pdf").exists())

    @unittest.skipIf(os.name == "nt", "symbolic-link permissions vary on Windows")
    def test_symbolic_link_in_removal_tree_is_never_followed(self) -> None:
        outside = self.repo / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        link = self.repo / "data" / "legacy" / "external-link"
        link.symlink_to(outside)

        with self.assertRaisesRegex(self.module.SafetyError, "symbolic-link"):
            self.module.run_retirement(self.repo, self.archive, execute=True)

        self.assertTrue(link.is_symlink())
        self.assertEqual(outside.read_text(encoding="utf-8"), "outside")

    def test_cli_without_execute_reports_dry_run_and_changes_nothing(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            exit_code = self.module.main(
                (
                    "--repo",
                    str(self.repo),
                    "--archive-root",
                    str(self.archive),
                    "--json",
                )
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue())["mode"], "dry-run")
        self.assertTrue((self.repo / "data" / "legacy" / "old.csv").exists())
        self.assertFalse((self.repo / "figures" / "README.md").exists())


if __name__ == "__main__":
    unittest.main()
