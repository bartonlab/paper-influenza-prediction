"""Tests for canonical background-fitness figure renderers."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from importlib.util import find_spec
from pathlib import Path
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

HAS_PLOTTING = all(
    find_spec(package) is not None
    for package in ("matplotlib", "numpy", "pandas", "scipy", "seaborn")
)

from influenza_figures.background_plot import (  # noqa: E402
    ANNUALIZED_GROWTH_METRIC,
    HISTOGRAM_X_LIMIT,
    HISTOGRAM_Y_LIMIT,
    INITIAL_ADVANTAGE_METRIC,
    SELECTION_METRIC,
    _deterministic_offsets,
    render_background_clade_figure,
    render_background_growth_figure,
    render_background_initial_advantage_figure,
    render_background_selection_figure,
)
from influenza_figures.registry import FIGURE_REGISTRY  # noqa: E402


BACKGROUND = (
    REPO
    / "data"
    / "plotting"
    / "inputs"
    / "background"
    / "population_fitness.csv"
)
CLADES = (
    REPO
    / "data"
    / "plotting"
    / "inputs"
    / "clades"
    / "h3n2_ha_frequencies.csv"
)


class BackgroundPlotImportTests(unittest.TestCase):
    def test_plotting_dependencies_are_lazy(self) -> None:
        source = (
            REPO / "src" / "influenza_figures" / "background_plot.py"
        ).read_text(encoding="utf-8")
        module = ast.parse(source)
        imported = set()
        for statement in module.body:
            if isinstance(statement, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in statement.names)
            elif isinstance(statement, ast.ImportFrom) and statement.module:
                imported.add(statement.module.split(".")[0])
        self.assertTrue(
            {"matplotlib", "numpy", "pandas", "scipy", "seaborn"}.isdisjoint(
                imported
            )
        )

    def test_metric_filters_are_explicit(self) -> None:
        self.assertIsNone(SELECTION_METRIC.minimum_duration_days)
        self.assertEqual(INITIAL_ADVANTAGE_METRIC.minimum_duration_days, 90)
        self.assertEqual(ANNUALIZED_GROWTH_METRIC.minimum_duration_days, 90)

    def test_deterministic_offsets_do_not_use_random_state(self) -> None:
        expected = (0.0, -0.05, 0.05, -0.075, 0.025, -0.025)
        first = _deterministic_offsets(6)
        second = _deterministic_offsets(6)
        self.assertEqual(first, second)
        for actual, wanted in zip(first, expected, strict=True):
            self.assertAlmostEqual(actual, wanted)
        self.assertTrue(all(-0.1 <= value <= 0.1 for value in expected))


@unittest.skipUnless(HAS_PLOTTING, "scientific plotting dependencies unavailable")
class BackgroundPlotRenderTests(unittest.TestCase):
    def tearDown(self) -> None:
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_s13_uses_one_earliest_record_per_mutation(self) -> None:
        result = render_background_selection_figure(BACKGROUND)
        audit = result.audit

        self.assertEqual(len(result.summary_axes), 4)
        self.assertEqual(result.clade_axes, ())
        self.assertEqual(audit.layout, "two_by_two_fixed_lost_histograms")
        self.assertEqual(audit.event_count_before_filter, 418)
        self.assertEqual(audit.retained_event_count, 418)
        self.assertEqual(audit.excluded_by_duration_count, 0)
        self.assertEqual(
            [(panel.fixed_count, panel.lost_count) for panel in audit.panels],
            [(10, 109), (11, 91), (12, 101), (12, 72)],
        )
        h3 = audit.panels[0]
        self.assertAlmostEqual(h3.fixed_mean, 0.39636569)
        self.assertAlmostEqual(h3.lost_mean, 0.16579808348623855)
        self.assertAlmostEqual(h3.welch_t_statistic, 1.9460570078302333)
        self.assertAlmostEqual(h3.welch_p_value, 0.08154550625445521)

        spec = FIGURE_REGISTRY.get("paper.ds")
        self.assertEqual(
            tuple(result.figure.get_size_inches()),
            spec.canvas.size_inches,
        )
        for axis in result.summary_axes:
            self.assertEqual(tuple(axis.get_xlim()), HISTOGRAM_X_LIMIT)
            self.assertEqual(tuple(axis.get_ylim()), HISTOGRAM_Y_LIMIT)
        with self.assertRaises(FrozenInstanceError):
            h3.fixed_count = 0  # type: ignore[misc]

    def test_s14_uses_inclusive_90_day_filter_and_welch_statistics(self) -> None:
        result = render_background_initial_advantage_figure(BACKGROUND)
        audit = result.audit

        self.assertEqual(audit.event_count_before_filter, 418)
        self.assertEqual(audit.retained_event_count, 315)
        self.assertEqual(audit.excluded_by_duration_count, 103)
        self.assertEqual(
            [(panel.fixed_count, panel.lost_count) for panel in audit.panels],
            [(13, 71), (11, 68), (13, 75), (17, 47)],
        )
        h3 = audit.panels[0]
        self.assertAlmostEqual(h3.fixed_mean, 0.21555168461538335)
        self.assertAlmostEqual(h3.lost_mean, 0.08424326619718335)
        self.assertAlmostEqual(h3.welch_t_statistic, 1.1411305663936826)
        self.assertAlmostEqual(h3.welch_p_value, 0.26965495900454556)
        for axis in result.summary_axes:
            self.assertEqual(tuple(axis.get_xlim()), HISTOGRAM_X_LIMIT)
            self.assertEqual(tuple(axis.get_ylim()), HISTOGRAM_Y_LIMIT)

    def test_s15_creates_output_and_reports_annualized_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "background-growth.png"
            result = render_background_growth_figure(
                BACKGROUND,
                output_path=output,
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)
            self.assertEqual(result.audit.saved_path, str(output.resolve()))

        n2 = result.audit.panels[1]
        self.assertEqual((n2.fixed_count, n2.lost_count), (11, 68))
        self.assertAlmostEqual(n2.fixed_mean, 0.5448701564967545)
        self.assertAlmostEqual(n2.lost_mean, 0.10446383196964316)
        self.assertAlmostEqual(n2.welch_t_statistic, 3.147258050812061)
        self.assertAlmostEqual(n2.welch_p_value, 0.004587564447932253)

    def test_main_figure_combines_twelve_summaries_and_four_clade_axes(self) -> None:
        result = render_background_clade_figure(BACKGROUND, CLADES)
        audit = result.audit

        self.assertEqual(len(result.summary_axes), 12)
        self.assertEqual(len(result.clade_axes), 4)
        self.assertEqual(len(result.axes), 16)
        self.assertEqual(len(audit.panels), 12)
        self.assertEqual(audit.retained_event_count, 315)
        self.assertEqual(
            audit.selected_clades,
            ("3C", "3C.2", "3C.2a", "3C.3", "3C.3a", "3C.3b"),
        )
        self.assertEqual(
            audit.clade_row_counts,
            (("observed_history", 624), ("observed_full", 846), ("forecast", 24)),
        )
        self.assertEqual(
            tuple(panel.metric for panel in audit.panels),
            (SELECTION_METRIC.field,) * 4
            + (INITIAL_ADVANTAGE_METRIC.field,) * 4
            + (ANNUALIZED_GROWTH_METRIC.field,) * 4,
        )
        self.assertEqual(
            [(panel.fixed_count, panel.lost_count) for panel in audit.panels[:4]],
            [(13, 71), (11, 68), (13, 75), (17, 47)],
        )
        self.assertAlmostEqual(audit.panels[0].welch_p_value, 0.08429381523268585)

        spec = FIGURE_REGISTRY.get("paper.dw_clade")
        self.assertEqual(
            tuple(result.figure.get_size_inches()),
            spec.canvas.size_inches,
        )
        self.assertEqual(
            [
                text.get_text()
                for text in result.figure.texts
                if text.get_text() in "ABCD"
            ],
            ["A", "B", "C", "D"],
        )

    def test_main_background_y_spines_end_at_outer_major_ticks(self) -> None:
        result = render_background_clade_figure(BACKGROUND, CLADES)

        # Each summary row has one visible y-axis in its first column.  The
        # data limits retain padding, but the drawn spine should stop at the
        # first and last labeled major ticks.
        for row_index, (axis, metric) in enumerate(
            zip(
                result.summary_axes[::4],
                (
                    SELECTION_METRIC,
                    INITIAL_ADVANTAGE_METRIC,
                    ANNUALIZED_GROWTH_METRIC,
                ),
                strict=True,
            )
        ):
            ticks = tuple(float(value) for value in axis.get_yticks())
            self.assertEqual(tuple(axis.get_ylim()), metric.y_limits)
            self.assertTrue(axis.spines["left"].get_visible())
            bounds = axis.spines["left"].get_bounds()
            self.assertIsNotNone(bounds)
            self.assertEqual(
                tuple(bounds),
                (min(ticks), max(ticks)),
            )
            hidden_axes = result.summary_axes[
                row_index * 4 + 1 : row_index * 4 + 4
            ]
            for hidden_axis in hidden_axes:
                self.assertFalse(hidden_axis.spines["left"].get_visible())

    def test_main_background_advantage_range_is_tight_without_changing_s14(
        self,
    ) -> None:
        main = render_background_clade_figure(BACKGROUND, CLADES)
        supplement = render_background_initial_advantage_figure(BACKGROUND)

        main_advantage_axes = main.summary_axes[4:8]
        main_limits = tuple(main_advantage_axes[0].get_ylim())
        self.assertEqual(main_limits, (-0.016, 0.016))
        self.assertTrue(
            all(tuple(axis.get_ylim()) == main_limits for axis in main_advantage_axes)
        )
        for axis in main_advantage_axes:
            self.assertEqual(
                tuple(float(value) for value in axis.get_yticks()),
                (-0.015, 0.0, 0.015),
            )
        self.assertEqual(
            tuple(label.get_text() for label in main_advantage_axes[0].get_yticklabels()),
            ("-1.5", "0", "1.5"),
        )

        # The tighter Main Figure range must still show every mutation.  The
        # first two collections are the fixed and lost point clouds.
        plotted_values = [
            float(offset[1])
            for axis in main_advantage_axes
            for collection in axis.collections[:2]
            for offset in collection.get_offsets()
        ]
        self.assertLessEqual(main_limits[0], min(plotted_values))
        self.assertGreaterEqual(main_limits[1], max(plotted_values))

        # S14 is a frequency histogram and should keep its established scale;
        # a Main-Figure-only display adjustment must not leak into it.
        for axis in supplement.summary_axes:
            self.assertEqual(tuple(axis.get_xlim()), HISTOGRAM_X_LIMIT)
            self.assertEqual(tuple(axis.get_ylim()), HISTOGRAM_Y_LIMIT)


if __name__ == "__main__":
    unittest.main()
