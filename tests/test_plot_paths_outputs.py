from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from influenza_figures.outputs import (  # noqa: E402
    OutputBusyError,
    OutputExistsError,
    OutputManager,
    OutputNotProducedError,
    OutputOwnershipError,
)
from influenza_figures.paths import PathOutsideRootError, ProjectPaths  # noqa: E402
from influenza_figures.registry import Canvas, FigureRegistry, FigureSpec  # noqa: E402


def _test_registry() -> FigureRegistry:
    canvas = Canvas("test", 1.0, 1.0)
    return FigureRegistry(
        (
            FigureSpec("one", "render_one", "paper_v1", canvas, "one.pdf", "test"),
            FigureSpec("two", "render_two", "paper_v1", canvas, "nested/two.pdf", "test"),
            FigureSpec("three", "render_three", "paper_v1", canvas, "other/two.pdf", "test"),
        )
    )


class ProjectPathTests(unittest.TestCase):
    def test_paths_do_not_depend_on_current_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = ProjectPaths.from_root(directory)
            self.assertEqual(paths.data, Path(directory).resolve() / "data" / "plotting")
            self.assertEqual(paths.figure_path("a.pdf"), Path(directory).resolve() / "figures" / "a.pdf")

    def test_path_traversal_and_absolute_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = ProjectPaths.from_root(directory)
            with self.assertRaises(PathOutsideRootError):
                paths.figure_path("../outside.pdf")
            with self.assertRaises(PathOutsideRootError):
                paths.data_path(Path(directory).resolve() / "absolute.csv")


class OutputManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.paths = ProjectPaths.from_root(self.temporary_directory.name)
        self.registry = _test_registry()
        self.manager = OutputManager(self.paths, self.registry)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_successful_commit_is_atomic_and_owned(self) -> None:
        result = self.manager.write_bytes("one", b"first")
        self.assertEqual(result.read_bytes(), b"first")
        self.assertEqual(self.registry.owner_of("one.pdf"), "one")

    def test_existing_output_requires_explicit_overwrite(self) -> None:
        self.manager.write_bytes("one", b"first")
        with self.assertRaises(OutputExistsError):
            self.manager.write_bytes("one", b"second")
        self.assertEqual(self.manager.write_bytes("one", b"second", overwrite=True).read_bytes(), b"second")

    def test_nested_reservation_detects_concurrent_owner(self) -> None:
        with self.manager.reserve("one") as temporary:
            with self.assertRaises(OutputBusyError):
                with self.manager.reserve("one"):
                    pass
            temporary.write_bytes(b"complete")

    def test_same_basename_in_different_owned_directories_has_separate_lock(self) -> None:
        with self.manager.reserve("two") as second:
            with self.manager.reserve("three") as third:
                second.write_bytes(b"second")
                third.write_bytes(b"third")
        self.assertEqual(self.manager.output_path("two").read_bytes(), b"second")
        self.assertEqual(self.manager.output_path("three").read_bytes(), b"third")

    def test_missing_rendered_file_does_not_create_output(self) -> None:
        with self.assertRaises(OutputNotProducedError):
            with self.manager.reserve("one"):
                pass
        self.assertFalse(self.manager.output_path("one").exists())

    def test_owner_mismatch_is_rejected(self) -> None:
        with self.assertRaises(OutputOwnershipError):
            self.manager.assert_owner("one", "nested/two.pdf")


if __name__ == "__main__":
    unittest.main()
