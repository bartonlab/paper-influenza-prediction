"""Tests for canonical selection-focused manuscript renderers."""

from __future__ import annotations

import ast
from importlib.util import find_spec
import math
from pathlib import Path
import sys
import unittest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from influenza_figures.registry import FIGURE_REGISTRY  # noqa: E402
from influenza_figures.selection_plot import (  # noqa: E402
    ALLELE_FREQUENCY_DISPLAY_THRESHOLD,
    HA_DOMAIN_LABEL_Y,
    HA_DOMAIN_SPAN_Y,
    HA_SELECTION_LIMIT_MAX,
    HA_SELECTION_TICK_MAX,
    HA1_POSITION_RANGE,
    HA2_POSITION_RANGE,
    INITIAL_WILD_TYPE_FILTER_THRESHOLD,
    MIN_MUTATION_COUNT,
    NA_SELECTION_LIMIT_MAX,
    NA_SELECTION_TICK_MAX,
    PROTEIN_DEFINITIONS,
    SASA_LIMIT_MAX,
    SASA_TICK_MAX,
    _p_value_order_label,
    collect_frequency_selection_data,
    collect_selection_curve_data,
    render_frequency_selection_figure,
    render_selection_sasa_figure,
    render_selection_structure_space_figure,
)


PLOTTING_DATA = REPO / "data" / "plotting"
HAS_DATA_DEPS = find_spec("numpy") is not None and find_spec("pandas") is not None
HAS_PLOTTING = (
    HAS_DATA_DEPS
    and find_spec("matplotlib") is not None
    and find_spec("scipy") is not None
)


class SelectionDataTests(unittest.TestCase):
    def test_module_imports_plotting_dependencies_lazily(self) -> None:
        source = (REPO / "src" / "influenza_figures" / "selection_plot.py").read_text(
            encoding="utf-8"
        )
        module = ast.parse(source)
        imported = set()
        for statement in module.body:
            if isinstance(statement, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in statement.names)
            elif isinstance(statement, ast.ImportFrom) and statement.module:
                imported.add(statement.module.split(".")[0])
        self.assertTrue(
            {"matplotlib", "numpy", "pandas", "scipy"}.isdisjoint(imported)
        )

    def test_final_recipe_constants_are_explicit(self) -> None:
        self.assertEqual(MIN_MUTATION_COUNT, 20)
        self.assertEqual(ALLELE_FREQUENCY_DISPLAY_THRESHOLD, 0.01)
        self.assertEqual(INITIAL_WILD_TYPE_FILTER_THRESHOLD, 0.95)
        self.assertEqual(HA1_POSITION_RANGE, (18, 345))
        self.assertEqual(HA2_POSITION_RANGE, (346, 566))
        self.assertEqual((HA_DOMAIN_SPAN_Y, HA_DOMAIN_LABEL_Y), (0.39, 0.42))
        self.assertEqual(
            (HA_SELECTION_LIMIT_MAX, HA_SELECTION_TICK_MAX),
            (0.42, 0.4),
        )
        self.assertEqual(
            (NA_SELECTION_LIMIT_MAX, NA_SELECTION_TICK_MAX),
            (0.26, 0.2),
        )
        self.assertEqual((SASA_LIMIT_MAX, SASA_TICK_MAX), (100.0, 80.0))
        self.assertEqual(
            [definition.label for definition in PROTEIN_DEFINITIONS],
            ["H3", "N2", "H1", "N1"],
        )
        self.assertEqual(
            [definition.time_shift_days for definition in PROTEIN_DEFINITIONS],
            [0, 0, 120, 120],
        )

    @unittest.skipUnless(HAS_DATA_DEPS, "NumPy and pandas are required")
    def test_frequency_and_selection_filters_reproduce_final_s5_counts(self) -> None:
        data = collect_frequency_selection_data(PLOTTING_DATA)
        self.assertEqual(
            [len(panel.trajectories) for panel in data.panels],
            [322, 270, 313, 252],
        )
        self.assertEqual(
            [len(panel.selection_coefficients) for panel in data.panels],
            [1028, 884, 1070, 844],
        )
        self.assertEqual(
            [len(panel.time_days) for panel in data.panels],
            [408, 408, 396, 396],
        )
        self.assertEqual(
            [(panel.time_days[0], panel.time_days[-1]) for panel in data.panels],
            [(0, 4070), (0, 4070), (120, 4070), (120, 4070)],
        )

    @unittest.skipUnless(HAS_DATA_DEPS, "NumPy and pandas are required")
    def test_consolidated_structure_table_has_all_four_plot_series(self) -> None:
        curves = collect_selection_curve_data(PLOTTING_DATA)
        self.assertEqual([curve.definition.label for curve in curves], ["H3", "N2", "H1", "N1"])
        self.assertEqual([len(curve.positions) for curve in curves], [566, 469, 566, 469])
        self.assertTrue(all(curve.positions[0] == 1 for curve in curves))
        self.assertEqual(
            [curve.positions[-1] for curve in curves],
            [566, 469, 566, 469],
        )

    def test_functional_site_p_value_labels_use_legacy_orders(self) -> None:
        self.assertEqual(_p_value_order_label(1.93e-9), r"$p < 10^{-8}$")
        self.assertEqual(_p_value_order_label(3.83e-22), r"$p < 10^{-21}$")
        self.assertEqual(_p_value_order_label(0.0), r"$p < 10^{-300}$")


@unittest.skipUnless(HAS_PLOTTING, "Matplotlib, NumPy, pandas, and SciPy are required")
class SelectionRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frequency_result = render_frequency_selection_figure(
            PLOTTING_DATA,
            figure_spec=FIGURE_REGISTRY.get("paper.frequency_selection"),
        )
        cls.structure_result = render_selection_structure_space_figure(
            PLOTTING_DATA,
            figure_spec=FIGURE_REGISTRY.get("paper.selection_structure_space"),
        )
        cls.sasa_result = render_selection_sasa_figure(
            PLOTTING_DATA,
            figure_spec=FIGURE_REGISTRY.get("paper.selection_sasa"),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        import matplotlib.pyplot as plt

        for result in (
            cls.frequency_result,
            cls.structure_result,
            cls.sasa_result,
        ):
            plt.close(result.figure)

    def test_s5_layout_and_audit_are_transparent(self) -> None:
        result = self.frequency_result
        self.assertEqual(len(result.axes), 8)
        self.assertEqual(
            tuple(result.figure.get_size_inches()),
            FIGURE_REGISTRY.get("paper.frequency_selection").canvas.size_inches,
        )
        self.assertEqual(
            [panel.plotted_trajectory_count for panel in result.audit.panels],
            [322, 270, 313, 252],
        )
        self.assertEqual(
            [panel.plotted_mutation_count for panel in result.audit.panels],
            [1028, 884, 1070, 844],
        )
        self.assertIsNone(result.audit.saved_path)

    def test_figure_2_functional_site_statistics_match_reference_labels(self) -> None:
        result = self.structure_result
        self.assertEqual(len(result.axes), 9)
        self.assertEqual(
            [audit.subset_count for audit in result.audit.functional_sites],
            [78, 21, 45, 149],
        )
        self.assertEqual(
            [audit.p_value_label for audit in result.audit.functional_sites],
            [
                r"$p < 10^{-8}$",
                r"$p < 10^{-21}$",
                r"$p < 10^{-5}$",
                r"$p < 10^{-8}$",
            ],
        )
        self.assertEqual(
            [audit.label for audit in result.audit.selection_curves],
            ["H3", "H1", "N2", "N1"],
        )

    def test_figure_2_sequence_axes_preserve_padding_above_the_last_tick(self) -> None:
        result = self.structure_result
        # The final two axes are panels C and D.  The legacy renderer kept
        # display padding above the outer tick; collapsing that padding into
        # the tick range clips the NA curves in panel D and crowds its label.
        expected_scales = (
            (HA_SELECTION_LIMIT_MAX, HA_SELECTION_TICK_MAX),
            (NA_SELECTION_LIMIT_MAX, NA_SELECTION_TICK_MAX),
        )
        for axis, (expected_limit, expected_tick) in zip(
            result.axes[-2:], expected_scales
        ):
            upper_limit = axis.get_ylim()[1]
            outer_tick = max(axis.get_yticks())
            plotted_values = [
                float(value)
                for line in axis.lines
                for value in line.get_ydata()
                if math.isfinite(float(value))
            ]

            self.assertTrue(plotted_values)
            self.assertAlmostEqual(upper_limit, expected_limit)
            self.assertAlmostEqual(outer_tick, expected_tick)
            self.assertGreater(upper_limit, outer_tick)
            self.assertGreater(upper_limit, max(plotted_values))
            self.assertEqual(
                axis.spines["left"].get_bounds(),
                (min(axis.get_yticks()), outer_tick),
            )

    def test_figure_2_ha_domain_marker_clears_protein_labels(self) -> None:
        axis = self.structure_result.axes[-2]
        self.structure_result.figure.canvas.draw()
        renderer = self.structure_result.figure.canvas.get_renderer()
        span_y_display = axis.transData.transform((0, HA_DOMAIN_SPAN_Y))[1]

        span_lines = axis.lines[-2:]
        self.assertEqual(
            [tuple(line.get_ydata()) for line in span_lines],
            [(HA_DOMAIN_SPAN_Y, HA_DOMAIN_SPAN_Y)] * 2,
        )
        for text in axis.texts:
            if text.get_text() in {"H3", "H1"}:
                self.assertGreater(
                    span_y_display,
                    text.get_window_extent(renderer).y1,
                )
            elif text.get_text() in {"HA1", "HA2"}:
                self.assertEqual(text.get_position()[1], HA_DOMAIN_LABEL_Y)
                self.assertGreater(
                    text.get_window_extent(renderer).y0,
                    span_y_display,
                )

    def test_s6_uses_full_bottom_position_ticks_and_reports_correlations(self) -> None:
        result = self.sasa_result
        self.assertEqual(len(result.axes), 8)
        # Axes are returned as selection/SASA pairs in visual row-major order.
        h1_selection_axis = result.axes[4]
        n1_selection_axis = result.axes[6]
        self.assertEqual(tuple(h1_selection_axis.get_xticks()), tuple(range(0, 601, 100)))
        self.assertEqual(tuple(n1_selection_axis.get_xticks()), tuple(range(0, 501, 100)))
        self.assertEqual(
            [(row.label, row.domain) for row in result.audit.correlations],
            [
                ("H3", "full"),
                ("H3", "HA1"),
                ("H3", "HA2"),
                ("N2", "full"),
                ("H1", "full"),
                ("H1", "HA1"),
                ("H1", "HA2"),
                ("N1", "full"),
            ],
        )
        self.assertTrue(
            all(row.paired_count > 100 for row in result.audit.correlations)
        )

    def test_s6_curves_fit_inside_axes_with_padding_above_outer_ticks(self) -> None:
        result = self.sasa_result
        # Axes alternate between selection and SASA for H3, N2, H1, and N1.
        # Check both families because the consolidated inputs include NA
        # selection peaks above 0.2 and an H1 SASA peak above 80.
        expected_scales = (
            (HA_SELECTION_LIMIT_MAX, HA_SELECTION_TICK_MAX),
            (SASA_LIMIT_MAX, SASA_TICK_MAX),
            (NA_SELECTION_LIMIT_MAX, NA_SELECTION_TICK_MAX),
            (SASA_LIMIT_MAX, SASA_TICK_MAX),
            (HA_SELECTION_LIMIT_MAX, HA_SELECTION_TICK_MAX),
            (SASA_LIMIT_MAX, SASA_TICK_MAX),
            (NA_SELECTION_LIMIT_MAX, NA_SELECTION_TICK_MAX),
            (SASA_LIMIT_MAX, SASA_TICK_MAX),
        )
        for axis_index, (axis, (expected_limit, expected_tick)) in enumerate(
            zip(result.axes, expected_scales)
        ):
            upper_limit = axis.get_ylim()[1]
            outer_tick = max(axis.get_yticks())
            plotted_values = [
                float(value)
                for line in axis.lines
                for value in line.get_ydata()
                if math.isfinite(float(value))
            ]

            with self.subTest(axis_index=axis_index):
                self.assertTrue(plotted_values)
                self.assertAlmostEqual(upper_limit, expected_limit)
                self.assertAlmostEqual(outer_tick, expected_tick)
                self.assertGreater(upper_limit, outer_tick)
                self.assertGreater(upper_limit, max(plotted_values))
                spine_name = "left" if axis_index % 2 == 0 else "right"
                self.assertEqual(
                    axis.spines[spine_name].get_bounds(),
                    (min(axis.get_yticks()), outer_tick),
                )

    def test_s6_ha_domain_markers_clear_selection_labels(self) -> None:
        self.sasa_result.figure.canvas.draw()
        renderer = self.sasa_result.figure.canvas.get_renderer()
        # H3 and H1 selection axes are indices 0 and 4 in visual order.
        for axis_index in (0, 4):
            axis = self.sasa_result.axes[axis_index]
            span_y_display = axis.transData.transform((0, HA_DOMAIN_SPAN_Y))[1]
            span_lines = axis.lines[-2:]
            self.assertEqual(
                [tuple(line.get_ydata()) for line in span_lines],
                [(HA_DOMAIN_SPAN_Y, HA_DOMAIN_SPAN_Y)] * 2,
            )
            selection_text = next(
                text for text in axis.texts if text.get_text().endswith(" selection")
            )
            self.assertGreater(
                span_y_display,
                selection_text.get_window_extent(renderer).y1,
            )

        h3_axis = self.sasa_result.axes[0]
        h3_span_y_display = h3_axis.transData.transform((0, HA_DOMAIN_SPAN_Y))[1]
        for text in h3_axis.texts:
            if text.get_text() in {"HA1", "HA2"}:
                self.assertEqual(text.get_position()[1], HA_DOMAIN_LABEL_Y)
                self.assertGreater(
                    text.get_window_extent(renderer).y0,
                    h3_span_y_display,
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
