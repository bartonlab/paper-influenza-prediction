"""Tests for deterministic canonical background plotting series."""

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from influenza_figures.background_plot_data import (  # noqa: E402
    OUTCOME_ORDER,
    PANEL_ORDER,
    BackgroundPlotDataError,
    prepare_background_plot_data,
)


CANONICAL_BACKGROUND = REPO / "data" / "plotting" / "inputs" / "background"

FIELDS = [
    "subtype",
    "segment",
    "source_variant",
    "position",
    "state",
    "start_time",
    "frequency",
    "initial_time",
    "final_time",
    "duration_days",
    "selection_coefficient",
    "initial_background_fitness",
    "initial_population_fitness",
    "final_background_fitness",
    "background_fitness_change",
    "outcome",
    "prediction",
]


def observation(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "subtype": "h3n2",
        "segment": "ha",
        "source_variant": "population_fitness",
        "position": "161",
        "state": "17",
        "start_time": 100,
        "frequency": 0.08,
        "initial_time": 10,
        "final_time": 100,
        "duration_days": 90,
        "selection_coefficient": 0.006,
        "initial_background_fitness": 1.02,
        "initial_population_fitness": 1.00,
        "final_background_fitness": 1.029,
        "background_fitness_change": 0.009,
        "outcome": "fixed",
        "prediction": 1,
    }
    row.update(updates)
    return row


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class BackgroundPlotDataTests(unittest.TestCase):
    def test_module_import_needs_no_scientific_dependencies(self) -> None:
        script = f"""
import importlib.abc
import sys

class ScientificDependencyBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        dependency = fullname.split('.')[0]
        if dependency in {{'matplotlib', 'numpy', 'pandas', 'scipy', 'seaborn'}}:
            raise ModuleNotFoundError(
                f"No module named {{dependency!r}}",
                name=dependency,
            )
        return None

sys.meta_path.insert(0, ScientificDependencyBlocker())
sys.path.insert(0, {str(REPO / 'src')!r})
from influenza_figures.background_plot_data import prepare_background_plot_data
if not callable(prepare_background_plot_data):
    raise AssertionError('background plot-data helper is not callable')
"""
        result = subprocess.run(
            [sys.executable, "-I", "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_panel_and_outcome_orders_are_exact(self) -> None:
        self.assertEqual(PANEL_ORDER, ("H3", "N2", "H1", "N1"))
        self.assertEqual(OUTCOME_ORDER, ("fixed", "lost"))

    def test_series_are_exact_aligned_formulas_without_row_order_dependence(
        self,
    ) -> None:
        rows = [
            observation(
                start_time=100,
                final_time=99,
                duration_days=89,
                frequency=0.08,
                prediction=1,
                # The legacy plot computes final minus initial instead of
                # reading this separately rounded source field.
                background_fitness_change=0.0089,
            ),
            # The later record is the earliest one eligible for the 90-day
            # plot and therefore supplies every plotted scientific value.
            observation(
                start_time=120,
                initial_time=20,
                final_time=120,
                duration_days=100,
                frequency=0.20,
                prediction=0,
                selection_coefficient=0.009,
                initial_background_fitness=1.03,
                initial_population_fitness=1.00,
                final_background_fitness=1.05,
                background_fitness_change=0.02,
            ),
            observation(
                position="8",
                state="4",
                start_time=60,
                outcome="lost",
                selection_coefficient=-0.002,
                initial_background_fitness=0.99,
                initial_population_fitness=1.00,
                final_background_fitness=0.9945,
                background_fitness_change=0.0045,
            ),
            observation(
                subtype="h3n2",
                segment="na",
                position="2",
                state="3",
                outcome="lost",
            ),
            observation(
                subtype="h1n1",
                segment="ha",
                position="2",
                state="3",
                outcome="fixed",
            ),
            observation(
                subtype="h1n1",
                segment="na",
                position="2",
                state="3",
                outcome="lost",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.csv"
            reversed_path = Path(directory) / "reversed.csv"
            write_rows(first_path, rows)
            write_rows(reversed_path, list(reversed(rows)))
            result = prepare_background_plot_data(
                first_path,
                minimum_duration_days=90,
            )
            reversed_result = prepare_background_plot_data(
                reversed_path,
                minimum_duration_days=90,
            )

        self.assertEqual(result.panels, reversed_result.panels)
        self.assertEqual(result.audit, reversed_result.audit)
        self.assertEqual(
            tuple(panel.label for panel in result.panels),
            PANEL_ORDER,
        )

        fixed = result.panel("h3").fixed
        self.assertEqual(fixed.count, 1)
        self.assertEqual(fixed.selection_coefficient, (0.009,))
        self.assertAlmostEqual(fixed.background_fitness_change[0], 0.02)
        self.assertAlmostEqual(
            fixed.annualized_background_fitness_change[0],
            365 * 0.02 / 100,
        )
        self.assertAlmostEqual(
            fixed.initial_background_minus_population_fitness[0],
            0.03,
        )
        self.assertEqual(fixed.events[0]["observation_count"], 2)
        self.assertEqual(fixed.events[0]["representative_start_time"], 120)

        lost = result.panel("H3").lost
        self.assertEqual(lost.selection_coefficient, (-0.002,))
        self.assertAlmostEqual(lost.background_fitness_change[0], 0.0045)
        self.assertAlmostEqual(
            lost.initial_background_minus_population_fitness[0],
            -0.01,
        )
        self.assertEqual(result.audit.input_row_count, 6)
        self.assertEqual(result.audit.event_count_before_filter, 5)
        self.assertEqual(result.audit.retained_event_count, 5)
        self.assertEqual(result.audit.collapsed_observation_count, 1)

    def test_minimum_duration_is_required_and_inclusive(self) -> None:
        rows = [
            observation(position="1", final_time=100),
            observation(position="2", final_time=99, duration_days=89),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duration.csv"
            write_rows(path, rows)
            with self.assertRaises(TypeError):
                prepare_background_plot_data(path)  # type: ignore[call-arg]
            result = prepare_background_plot_data(
                path,
                minimum_duration_days=90,
            )

        self.assertEqual(result.audit.event_count_before_filter, 2)
        self.assertEqual(result.audit.retained_event_count, 1)
        self.assertEqual(result.audit.excluded_by_duration_count, 1)
        self.assertEqual(result.panel("H3").fixed.events[0]["duration_days"], 90)

    def test_unrecognized_panel_is_rejected_instead_of_silently_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.csv"
            write_rows(path, [observation(subtype="h5n1")])
            with self.assertRaisesRegex(
                BackgroundPlotDataError,
                "does not belong to an established background panel",
            ):
                prepare_background_plot_data(path, minimum_duration_days=None)

    def test_unrecognized_outcome_is_rejected_instead_of_silently_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outcome.csv"
            write_rows(path, [observation(outcome="polymorphic")])
            with self.assertRaisesRegex(
                BackgroundPlotDataError,
                "unsupported outcome",
            ):
                prepare_background_plot_data(path, minimum_duration_days=None)

    def test_missing_population_quantity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.csv"
            write_rows(path, [observation(initial_population_fitness="")])
            with self.assertRaisesRegex(
                BackgroundPlotDataError,
                "missing required 'initial_population_fitness'",
            ):
                prepare_background_plot_data(path, minimum_duration_days=None)

    def test_canonical_population_table_has_expected_panel_audit(self) -> None:
        result = prepare_background_plot_data(
            CANONICAL_BACKGROUND / "population_fitness.csv",
            minimum_duration_days=90,
        )

        self.assertEqual(result.audit.input_row_count, 15_277)
        self.assertEqual(result.audit.event_count_before_filter, 418)
        self.assertEqual(result.audit.retained_event_count, 315)
        self.assertEqual(result.audit.collapsed_observation_count, 14_859)
        self.assertEqual(result.audit.excluded_by_duration_count, 103)
        self.assertEqual(dict(result.audit.outcome_counts), {"fixed": 54, "lost": 261})
        self.assertEqual(
            {
                label: dict(counts)
                for label, counts in result.audit.panel_outcome_counts.items()
            },
            {
                "H3": {"fixed": 13, "lost": 71},
                "N2": {"fixed": 11, "lost": 68},
                "H1": {"fixed": 13, "lost": 75},
                "N1": {"fixed": 17, "lost": 47},
            },
        )
        for panel in result.panels:
            for outcome in OUTCOME_ORDER:
                series = panel.outcome(outcome)
                self.assertEqual(
                    {
                        series.count,
                        len(series.selection_coefficient),
                        len(series.background_fitness_change),
                        len(series.annualized_background_fitness_change),
                        len(series.initial_background_minus_population_fitness),
                    },
                    {series.count},
                )


if __name__ == "__main__":
    unittest.main()
