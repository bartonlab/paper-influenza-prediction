from __future__ import annotations

import importlib
import sys
import unittest
from importlib.util import find_spec
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


LEGACY_DEPENDENCIES_AVAILABLE = all(
    find_spec(name) is not None
    for name in ("numpy", "scipy", "matplotlib", "pandas", "seaborn")
)


@unittest.skipUnless(
    LEGACY_DEPENDENCIES_AVAILABLE,
    "legacy plotting dependencies unavailable",
)
class LegacyCompatibilityTests(unittest.TestCase):
    def test_root_figures_is_readable_facade_and_mplot_remains_alias(self) -> None:
        figures = importlib.import_module("figures")
        mplot = importlib.import_module("mplot")
        legacy_figures = importlib.import_module("influenza_figures.legacy.figures")
        legacy_mplot = importlib.import_module("influenza_figures.legacy.mplot")

        self.assertIsNot(figures, legacy_figures)
        self.assertIs(figures.legacy, legacy_figures)
        self.assertIs(figures.plot_AUC, legacy_figures.plot_AUC)
        self.assertEqual(figures.plot_emd_comparison.__module__, "figures")
        self.assertIs(mplot, legacy_mplot)

    def test_raw_global_assignment_is_explicitly_legacy_scoped(self) -> None:
        figures = importlib.import_module("figures")
        original = figures.legacy.FIG_DIR
        try:
            figures.legacy.FIG_DIR = "compatibility-test"
            self.assertEqual(
                figures.legacy.plot_frequency_selection.__globals__["FIG_DIR"],
                "compatibility-test",
            )
        finally:
            figures.legacy.FIG_DIR = original

    def test_unrelated_project_tail_is_archive_only(self) -> None:
        figures = importlib.import_module("figures")
        self.assertFalse(hasattr(figures, "plot_figure_ch505_structure"))


if __name__ == "__main__":
    unittest.main()
