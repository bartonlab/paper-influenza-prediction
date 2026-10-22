"""Tests for canonical clade selection and normalization."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

HAS_SCIENTIFIC_DEPENDENCIES = bool(find_spec("numpy") and find_spec("pandas"))
if HAS_SCIENTIFIC_DEPENDENCIES:
    import numpy as np
    import pandas as pd

    from influenza_figures.clades import (
        CladePreparationError,
        filter_and_normalize_clade_frequencies,
        prepare_clade_frequencies,
        select_clades_by_total_frequency,
    )


CANONICAL_CLADES = (
    REPO / "data" / "plotting" / "inputs" / "clades" / "h3n2_ha_frequencies.csv"
)


class PackageCladeImportTests(unittest.TestCase):
    def test_package_import_is_lazy_without_scientific_dependencies(self) -> None:
        script = f"""
import importlib.abc
import sys

class ScientificDependencyBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        dependency = fullname.split('.')[0]
        if dependency in {{'numpy', 'pandas'}}:
            raise ModuleNotFoundError(
                f"No module named {{dependency!r}}",
                name=dependency,
            )
        return None

sys.meta_path.insert(0, ScientificDependencyBlocker())
sys.path.insert(0, {str(REPO / 'src')!r})
import influenza_figures
if 'influenza_figures.clades' in sys.modules:
    raise AssertionError('package import loaded clade dependencies eagerly')
try:
    influenza_figures.prepare_clade_frequencies
except ModuleNotFoundError as error:
    if 'scientific Python dependencies' not in str(error):
        raise
else:
    raise AssertionError('clade export unexpectedly loaded without dependencies')
"""
        result = subprocess.run(
            [sys.executable, "-I", "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(
    HAS_SCIENTIFIC_DEPENDENCIES,
    "canonical clade preparation dependencies unavailable",
)
class CladePreparationTests(unittest.TestCase):
    def test_tidy_canonical_data_selects_six_clades_before_normalization(self) -> None:
        data = pd.read_csv(CANONICAL_CLADES)
        original = data.copy(deep=True)

        result = prepare_clade_frequencies(
            data,
            reference_start_date="2011-08-01",
            reference_end_date="2015-06-01",
        )

        self.assertEqual(
            result.selected_clades,
            ("3C", "3C.2", "3C.2a", "3C.3", "3C.3a", "3C.3b"),
        )
        self.assertEqual(set(result.data["clade"]), set(result.selected_clades))
        self.assertGreater(result.reference_totals["3C.3b"], 1.0)
        self.assertLess(result.reference_totals["3C.2a1b.1"], 1.0)
        pd.testing.assert_frame_equal(data, original)

        sums = result.data.groupby(
            ["subtype", "segment", "series", "date"], sort=False
        )["frequency"].sum()
        np.testing.assert_allclose(sums.to_numpy(), 1.0, rtol=0.0, atol=1e-12)

    def test_old_wide_equivalent_excludes_transient_and_boundary_clades(self) -> None:
        rows = [
            ("observed_full", "2014-01-01", "A", 0.7),
            ("observed_full", "2014-01-01", "B", 0.6),
            ("observed_full", "2014-01-01", "C", 0.4),
            ("observed_full", "2014-01-01", "D", 0.5),
            ("observed_full", "2014-02-01", "A", 0.5),
            ("observed_full", "2014-02-01", "B", 0.5),
            ("observed_full", "2014-02-01", "C", 0.5),
            ("observed_full", "2014-02-01", "D", 0.5),
            ("observed_history", "2014-01-01", "A", 0.4),
            ("observed_history", "2014-01-01", "B", 0.2),
            ("observed_history", "2014-01-01", "C", 0.4),
            ("observed_history", "2014-01-01", "D", 0.0),
            ("forecast", "2014-03-01", "A", 0.1),
            ("forecast", "2014-03-01", "B", 0.4),
            ("forecast", "2014-03-01", "C", 0.5),
            ("forecast", "2014-03-01", "D", 0.0),
        ]
        data = pd.DataFrame(
            [
                {
                    "subtype": "h3n2",
                    "segment": "ha",
                    "series": series,
                    "date": date,
                    "clade": clade,
                    "frequency": frequency,
                }
                for series, date, clade, frequency in rows
            ]
        )

        result = prepare_clade_frequencies(data)

        self.assertEqual(result.selected_clades, ("A", "B"))
        self.assertEqual(result.reference_totals["D"], 1.0)
        self.assertNotIn("C", set(result.data["clade"]))
        forecast = result.data[result.data["series"].eq("forecast")].set_index(
            "clade"
        )["frequency"]
        self.assertAlmostEqual(forecast["A"], 0.2)
        self.assertAlmostEqual(forecast["B"], 0.8)

    def test_selection_is_strictly_greater_than_threshold(self) -> None:
        reference = pd.DataFrame(
            {
                "clade": ["above", "above", "equal", "equal"],
                "frequency": [0.6, 0.5, 0.4, 0.6],
            }
        )
        self.assertEqual(
            select_clades_by_total_frequency(reference),
            ("above",),
        )

    def test_explicit_filter_returns_copy_and_rejects_zero_sum_group(self) -> None:
        data = pd.DataFrame(
            {
                "series": ["history", "history", "forecast", "forecast"],
                "date": ["2014-01-01", "2014-01-01", "2014-02-01", "2014-02-01"],
                "clade": ["A", "B", "A", "B"],
                "frequency": [2.0, 1.0, 0.0, 0.0],
            }
        )
        original = data.copy(deep=True)

        with self.assertRaisesRegex(CladePreparationError, "positive finite"):
            filter_and_normalize_clade_frequencies(
                data,
                selected_clades=("A", "B"),
            )
        pd.testing.assert_frame_equal(data, original)

    def test_combined_subtype_tables_must_be_prepartitioned(self) -> None:
        data = pd.DataFrame(
            {
                "subtype": ["h3n2", "h1n1"],
                "segment": ["ha", "ha"],
                "series": ["observed_full", "observed_full"],
                "date": ["2014-01-01", "2014-01-01"],
                "clade": ["A", "A"],
                "frequency": [1.1, 1.1],
            }
        )
        with self.assertRaisesRegex(CladePreparationError, "one subtype/segment"):
            prepare_clade_frequencies(data)


if __name__ == "__main__":
    unittest.main()
