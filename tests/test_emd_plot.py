"""Tests for canonical comparator-driven EMD figure recipes."""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
from dataclasses import replace
from importlib.util import find_spec
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from influenza_figures.emd_plot import (  # noqa: E402
    BASELINE_FULL_MODEL_METHOD,
    COMPARATOR_DISPLAY_LABELS,
    DATE_AXIS_PADDING_DAYS,
    DELTA_LIMIT_BY_COMPARATOR,
    DELTA_LIMIT_BY_COMPARATOR_AND_HORIZON,
    DELTA_TICKS_BY_COMPARATOR,
    FULL_MODEL_WORSE_MARKER_KEY,
    FULL_MODEL_METHOD,
    HISTOGRAM_BIN_WIDTH,
    HISTOGRAM_FREQUENCY_LIMIT,
    HISTOGRAM_FREQUENCY_LIMIT_BY_COMPARATOR,
    HISTOGRAM_FREQUENCY_LIMIT_BY_COMPARATOR_AND_HORIZON,
    INFERENCE_RELATIVE_CUTOFF_DAY,
    LEGACY_DISTRIBUTION_BOUNDS,
    MLR_METHOD,
    NULL_METHOD,
    P_VALUE_DECIMAL_DISPLAY_THRESHOLD,
    P_VALUE_SIGNIFICANCE_THRESHOLD,
    SLM_METHOD,
    TARGET_METHOD_BY_HORIZON,
    TRAJECTORY_GUIDE_LEVELS,
    EmdFigureRecipeError,
    _p_value_label,
    collect_emd_figure_data,
    render_emd_comparison_figure,
)
from influenza_figures.registry import CANVASES, FIGURE_REGISTRY, FigureSpec  # noqa: E402


EMD = REPO / "data" / "plotting" / "inputs" / "emd"
HAS_PLOTTING = find_spec("matplotlib") is not None and find_spec("scipy") is not None


def figure_spec(
    *,
    profile: str,
    canvas: str,
    output_name: str,
    horizon_months: int,
    comparator_method: str,
) -> FigureSpec:
    return FigureSpec(
        key=f"test.emd.{profile}",
        renderer="render_emd_comparison_figure",
        profile=profile,
        canvas=CANVASES[canvas],
        output_name=output_name,
        category="test",
        layout="four_by_two_trajectory_and_distribution",
        recipe_parameters={
            "horizon_months": horizon_months,
            "comparator_method": comparator_method,
        },
    )


class EmdFigureDataTests(unittest.TestCase):
    def test_p_value_labels_follow_significance_and_direction_rules(self) -> None:
        self.assertEqual(P_VALUE_SIGNIFICANCE_THRESHOLD, 0.05)
        self.assertEqual(P_VALUE_DECIMAL_DISPLAY_THRESHOLD, 0.01)

        for p_value in (float("nan"), float("inf"), -0.01, 0.0500001, 0.8):
            with self.subTest(p_value=p_value):
                self.assertIsNone(
                    _p_value_label(p_value, full_model_is_worse=False)
                )

        self.assertEqual(
            _p_value_label(0.05, full_model_is_worse=False),
            r"$p = 0.05$",
        )
        self.assertEqual(
            _p_value_label(0.034, full_model_is_worse=False),
            r"$p = 0.03$",
        )
        self.assertEqual(
            _p_value_label(0.01001, full_model_is_worse=False),
            r"$p = 0.01$",
        )
        self.assertEqual(
            _p_value_label(0.009, full_model_is_worse=False),
            r"$p < 10^{-2}$",
        )
        self.assertEqual(
            _p_value_label(0.0, full_model_is_worse=False),
            r"$p < 10^{-300}$",
        )

        self.assertEqual(
            _p_value_label(0.034, full_model_is_worse=True),
            r"$p = 0.03$*",
        )
        self.assertEqual(
            _p_value_label(0.05, full_model_is_worse=True),
            r"$p = 0.05$*",
        )
        self.assertEqual(
            _p_value_label(0.009, full_model_is_worse=True),
            r"$p < 10^{-2}$*",
        )

    def test_module_has_no_top_level_matplotlib_or_scipy_import(self) -> None:
        source = (REPO / "src" / "influenza_figures" / "emd_plot.py").read_text(
            encoding="utf-8"
        )
        module = ast.parse(source)
        imported = set()
        for statement in module.body:
            if isinstance(statement, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in statement.names)
            elif isinstance(statement, ast.ImportFrom) and statement.module:
                imported.add(statement.module.split(".")[0])
        self.assertTrue({"matplotlib", "scipy"}.isdisjoint(imported))

    def test_data_module_imports_when_plotting_dependencies_are_absent(self) -> None:
        script = f"""
import importlib.abc
import sys

class PlottingDependencyBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        dependency = fullname.split('.')[0]
        if dependency in {{'matplotlib', 'scipy'}}:
            raise ModuleNotFoundError(
                f"No module named {{dependency!r}}",
                name=dependency,
            )
        return None

sys.meta_path.insert(0, PlottingDependencyBlocker())
sys.path.insert(0, {str(REPO / 'src')!r})
import influenza_figures.emd_plot
if 'matplotlib' in sys.modules or 'scipy' in sys.modules:
    raise AssertionError('plotting dependencies loaded eagerly')
"""
        result = subprocess.run(
            [sys.executable, "-I", "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_horizon_recipes_use_complete_results_grid(self) -> None:
        counts = [84, 84, 81, 81]
        for horizon in (6, 9, 12):
            with self.subTest(horizon=horizon):
                data = collect_emd_figure_data(
                    EMD,
                    horizon_months=horizon,
                    comparator_method=MLR_METHOD,
                )
                self.assertEqual(data.evaluation_protocol, "results_grid")
                self.assertEqual(
                    [panel.definition.label for panel in data.panels],
                    ["H3", "N2", "H1", "N1"],
                )
                self.assertEqual(
                    [panel.comparison.paired_count for panel in data.panels],
                    counts,
                )
                self.assertTrue(
                    all(
                        cutoff.source == "evaluation model_start_date"
                        for cutoff in data.cutoffs
                    )
                )

    def test_null_slm_and_mlr_are_distinct_explicit_method_ids(self) -> None:
        self.assertEqual(
            COMPARATOR_DISPLAY_LABELS[MLR_METHOD],
            "multinomial logistic regression (MLR)",
        )
        for comparator in (NULL_METHOD, SLM_METHOD, MLR_METHOD):
            with self.subTest(comparator=comparator):
                data = collect_emd_figure_data(
                    EMD,
                    horizon_months=6,
                    comparator_method=comparator,
                )
                self.assertEqual(
                    data.target_method,
                    TARGET_METHOD_BY_HORIZON[6],
                )
                self.assertEqual(data.comparator_method, comparator)
                self.assertTrue(
                    all(
                        panel.comparison.comparator_method == comparator
                        for panel in data.panels
                    )
                )

        for display_label in ("Null model", "SLM", "MLR", "neutral model"):
            with self.subTest(display_label=display_label):
                with self.assertRaisesRegex(
                    EmdFigureRecipeError, "explicit canonical method_id"
                ):
                    collect_emd_figure_data(
                        EMD,
                        horizon_months=6,
                        comparator_method=display_label,
                    )

    def test_every_horizon_recipe_uses_cross_validated_full_model(self) -> None:
        self.assertEqual(
            FULL_MODEL_METHOD,
            "full_model_cross_validated_rare_filtered",
        )
        self.assertEqual(
            TARGET_METHOD_BY_HORIZON,
            {
                6: FULL_MODEL_METHOD,
                9: FULL_MODEL_METHOD,
                12: FULL_MODEL_METHOD,
            },
        )
        for horizon in (6, 9, 12):
            with self.subTest(horizon=horizon):
                data = collect_emd_figure_data(
                    EMD,
                    horizon_months=horizon,
                    comparator_method=MLR_METHOD,
                )
                self.assertEqual(data.target_method, FULL_MODEL_METHOD)
                self.assertTrue(
                    all(
                        panel.comparison.target_method == FULL_MODEL_METHOD
                        for panel in data.panels
                    )
                )

    def test_corrected_nine_month_null_release_has_expected_audit_statistics(
        self,
    ) -> None:
        data = collect_emd_figure_data(
            EMD,
            horizon_months=9,
            comparator_method=NULL_METHOD,
        )

        self.assertEqual(
            [panel.comparison.paired_count for panel in data.panels],
            [84, 84, 81, 81],
        )
        expected_delta_ranges = {
            "H3": (-2.5563854116156683, 3.9700313122517334),
            "N2": (-0.4222585327116928, 3.5053453377223973),
            "H1": (0.09331316507074838, 2.867609535984632),
            "N1": (0.1720099833112041, 2.93592290910271),
        }
        for panel in data.panels:
            with self.subTest(panel=panel.definition.label):
                observed_range = (min(panel.deltas), max(panel.deltas))
                expected_range = expected_delta_ranges[panel.definition.label]
                for observed, expected in zip(
                    observed_range,
                    expected_range,
                    strict=True,
                ):
                    self.assertAlmostEqual(observed, expected, places=12)

        largest_delta = max(
            abs(delta) for panel in data.panels for delta in panel.deltas
        )
        self.assertAlmostEqual(largest_delta, 3.9700313122517334, places=12)
        self.assertEqual(
            DELTA_LIMIT_BY_COMPARATOR_AND_HORIZON[NULL_METHOD][9],
            4.25,
        )
        self.assertLess(
            largest_delta,
            DELTA_LIMIT_BY_COMPARATOR_AND_HORIZON[NULL_METHOD][9],
        )

    def test_corrected_null_release_preserves_other_nine_month_anchor_values(
        self,
    ) -> None:
        # This evaluation changed from Null=7.418495297805638 in the
        # superseded release to the corrected value below. The target, SLM,
        # MLR, and baseline values are fixed anchors from the accepted v3 run.
        expected_comparator_emd = {
            NULL_METHOD: 7.933124346917505,
            SLM_METHOD: 3.819301424381651,
            MLR_METHOD: 4.588726920011134,
            BASELINE_FULL_MODEL_METHOD: 3.836401051578392,
        }
        for comparator, expected_emd in expected_comparator_emd.items():
            with self.subTest(comparator=comparator):
                data = collect_emd_figure_data(
                    EMD,
                    horizon_months=9,
                    comparator_method=comparator,
                )
                h3 = next(
                    panel
                    for panel in data.panels
                    if panel.definition.label == "H3"
                )
                anchor = next(
                    row
                    for row in h3.comparison.rows
                    if row["observation_date"] == "2012-05-01"
                )
                self.assertAlmostEqual(
                    float(anchor["target_emd"]),
                    3.9630930346657713,
                    places=12,
                )
                self.assertAlmostEqual(
                    float(anchor["comparator_emd"]),
                    expected_emd,
                    places=12,
                )
                self.assertAlmostEqual(
                    float(anchor["delta_emd"]),
                    expected_emd - 3.9630930346657713,
                    places=12,
                )
                self.assertEqual(
                    anchor["target_source_id"],
                    anchor["comparator_source_id"],
                )

    def test_every_best_model_can_be_compared_with_its_baseline(self) -> None:
        for horizon in (6, 9, 12):
            with self.subTest(horizon=horizon):
                data = collect_emd_figure_data(
                    EMD,
                    horizon_months=horizon,
                    comparator_method=BASELINE_FULL_MODEL_METHOD,
                )
                self.assertEqual(data.target_method, FULL_MODEL_METHOD)
                self.assertEqual(
                    data.comparator_method,
                    BASELINE_FULL_MODEL_METHOD,
                )
                self.assertEqual(
                    [panel.comparison.paired_count for panel in data.panels],
                    [84, 84, 81, 81],
                )
                for panel in data.panels:
                    self.assertEqual(
                        panel.comparison.target_method,
                        FULL_MODEL_METHOD,
                    )
                    self.assertEqual(
                        panel.comparison.comparator_method,
                        BASELINE_FULL_MODEL_METHOD,
                    )
                    for row, delta in zip(
                        panel.comparison.rows,
                        panel.deltas,
                        strict=True,
                    ):
                        # Delta is baseline minus best, so positive values mean
                        # the cross-validation-selected model has lower EMD.
                        self.assertAlmostEqual(
                            delta,
                            float(row["comparator_emd"])
                            - float(row["target_emd"]),
                        )

                all_deltas = [
                    delta for panel in data.panels for delta in panel.deltas
                ]
                self.assertTrue(any(delta > 0.0 for delta in all_deltas))
                self.assertTrue(any(delta < 0.0 for delta in all_deltas))

    def test_comparator_scales_are_explicit_symmetric_and_do_not_clip_data(
        self,
    ) -> None:
        expected_limits = {
            NULL_METHOD: 5.0,
            SLM_METHOD: 5.75,
            MLR_METHOD: 6.25,
            BASELINE_FULL_MODEL_METHOD: 4.0,
        }
        self.assertEqual(dict(DELTA_LIMIT_BY_COMPARATOR), expected_limits)
        self.assertEqual(HISTOGRAM_BIN_WIDTH, 0.25)
        self.assertEqual(HISTOGRAM_FREQUENCY_LIMIT, 0.25)
        self.assertEqual(
            dict(HISTOGRAM_FREQUENCY_LIMIT_BY_COMPARATOR),
            {
                NULL_METHOD: 0.25,
                SLM_METHOD: 0.40,
                MLR_METHOD: 0.25,
                BASELINE_FULL_MODEL_METHOD: 0.40,
            },
        )
        self.assertEqual(
            {
                comparator: dict(by_horizon)
                for comparator, by_horizon in (
                    HISTOGRAM_FREQUENCY_LIMIT_BY_COMPARATOR_AND_HORIZON.items()
                )
            },
            {
                NULL_METHOD: {6: 0.30, 9: 0.25, 12: 0.25},
                SLM_METHOD: {6: 0.40, 9: 0.40, 12: 0.40},
                MLR_METHOD: {6: 0.25, 9: 0.25, 12: 0.25},
                BASELINE_FULL_MODEL_METHOD: {6: 0.40, 9: 0.40, 12: 0.40},
            },
        )
        self.assertEqual(
            dict(DELTA_LIMIT_BY_COMPARATOR_AND_HORIZON[NULL_METHOD]),
            {6: 5.0, 9: 4.25, 12: 5.0},
        )
        self.assertEqual(
            dict(DELTA_LIMIT_BY_COMPARATOR_AND_HORIZON[MLR_METHOD]),
            {6: 6.25, 9: 6.25, 12: 6.25},
        )

        for comparator in expected_limits:
            self.assertEqual(len(DELTA_TICKS_BY_COMPARATOR[comparator]), 5)
            self.assertEqual(
                DELTA_TICKS_BY_COMPARATOR[comparator],
                TRAJECTORY_GUIDE_LEVELS,
            )
            for horizon in (6, 9, 12):
                with self.subTest(comparator=comparator, horizon=horizon):
                    delta_limit = DELTA_LIMIT_BY_COMPARATOR_AND_HORIZON[
                        comparator
                    ][horizon]
                    data = collect_emd_figure_data(
                        EMD,
                        horizon_months=horizon,
                        comparator_method=comparator,
                    )
                    largest_delta = max(
                        abs(delta)
                        for panel in data.panels
                        for delta in panel.deltas
                    )
                    self.assertLessEqual(largest_delta, delta_limit)

    def test_cutoffs_are_distinct_subtype_relative_and_auditable(self) -> None:
        data = collect_emd_figure_data(
            EMD,
            horizon_months=6,
            comparator_method=NULL_METHOD,
        )
        self.assertEqual(data.relative_cutoff_day, INFERENCE_RELATIVE_CUTOFF_DAY)
        self.assertEqual(
            [
                (
                    cutoff.subtype,
                    cutoff.relative_day,
                    cutoff.cutoff_date,
                    cutoff.source,
                )
                for cutoff in data.cutoffs
            ],
            [
                ("h3n2", 1095, "2012-01-02", "evaluation model_start_date"),
                ("h1n1", 1095, "2012-05-01", "evaluation model_start_date"),
            ],
        )
        self.assertEqual(len(data.cutoff_report_lines), 2)
        self.assertIn("1095 days", data.cutoff_report_lines[0])
        self.assertIn("2012-01-02", data.cutoff_report_lines[0])
        self.assertIn("evaluation model_start_date", data.cutoff_report_lines[0])
        self.assertEqual(data.panels[2].observation_dates[0], "2012-05-01")

    def test_unsupported_horizon_is_rejected(self) -> None:
        with self.assertRaisesRegex(EmdFigureRecipeError, "Unsupported EMD horizon"):
            collect_emd_figure_data(
                EMD,
                horizon_months=3,
                comparator_method=MLR_METHOD,
            )

    def test_plot_recipe_rejects_any_unmatched_evaluations(self) -> None:
        valid = collect_emd_figure_data(
            EMD,
            horizon_months=6,
            comparator_method=MLR_METHOD,
        ).panels[0].comparison
        incomplete = replace(valid, unmatched_target_ids=("missing-evaluation",))
        with patch(
            "influenza_figures.emd_plot.compare_emd",
            return_value=incomplete,
        ):
            with self.assertRaisesRegex(
                EmdFigureRecipeError,
                "Incomplete exact EMD pairing",
            ):
                collect_emd_figure_data(
                    EMD,
                    horizon_months=6,
                    comparator_method=MLR_METHOD,
                )

    def test_renderer_refuses_mismatched_registered_recipe_parameters(self) -> None:
        spec = figure_spec(
            profile="paper_v1",
            canvas="double_over_2_2",
            output_name="paper/fig-emd-vs-null-12m.pdf",
            horizon_months=12,
            comparator_method=NULL_METHOD,
        )
        with self.assertRaisesRegex(
            EmdFigureRecipeError,
            "binds horizon_months=12",
        ):
            render_emd_comparison_figure(
                EMD,
                horizon_months=6,
                comparator_method=MLR_METHOD,
                figure_spec=spec,
            )
        with self.assertRaisesRegex(
            EmdFigureRecipeError,
            "binds comparator_method='null'",
        ):
            render_emd_comparison_figure(
                EMD,
                horizon_months=12,
                comparator_method=MLR_METHOD,
                figure_spec=spec,
            )


@unittest.skipUnless(HAS_PLOTTING, "scientific plotting dependencies unavailable")
class EmdFigureRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)

    def tearDown(self) -> None:
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_renderer_uses_supplied_paper_spec_and_saves_optionally(self) -> None:
        spec = figure_spec(
            profile="paper_v1",
            canvas="double_over_1_9",
            output_name="paper/test-emd.pdf",
            horizon_months=6,
            comparator_method=MLR_METHOD,
        )
        report = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "emd.pdf"
            result = render_emd_comparison_figure(
                EMD,
                horizon_months=6,
                comparator_method=MLR_METHOD,
                figure_spec=spec,
                output_path=output,
                report_stream=report,
            )

            self.assertTrue(output.is_file())
            self.assertEqual(result.audit.saved_path, str(output))
            self.assertEqual(result.audit.figure_key, spec.key)
            self.assertEqual(result.audit.profile, "paper_v1")
            self.assertEqual(result.audit.canvas_name, "double_over_1_9")
            self.assertEqual(result.audit.canvas_size_inches, spec.canvas.size_inches)
            self.assertEqual(
                tuple(float(value) for value in result.figure.get_size_inches()),
                spec.canvas.size_inches,
            )
            self.assertEqual(result.audit.layout, spec.layout)
            self.assertEqual(len(result.trajectory_axes), 4)
            self.assertEqual(len(result.distribution_axes), 4)
            self.assertEqual(len(result.axes), 8)
            self.assertIs(result.axes[0], result.trajectory_axes[0])
            self.assertIs(result.axes[1], result.distribution_axes[0])
            self.assertEqual(
                [panel.label for panel in result.audit.panels],
                ["H3", "N2", "H1", "N1"],
            )
            self.assertEqual(result.audit.delta_limit_abs, 6.25)
            self.assertEqual(result.audit.distribution_delta_limit_abs, 6.25)
            self.assertEqual(result.audit.delta_ticks, TRAJECTORY_GUIDE_LEVELS)
            self.assertEqual(result.audit.histogram_frequency_limit, 0.25)
            for trajectory_axis, distribution_axis in zip(
                result.trajectory_axes,
                result.distribution_axes,
                strict=True,
            ):
                self.assertEqual(trajectory_axis.get_ylim(), (-6.25, 6.25))
                self.assertEqual(distribution_axis.get_xlim(), (-6.25, 6.25))
                self.assertEqual(distribution_axis.get_ylim(), (0.0, 0.25))
                self.assertAlmostEqual(
                    sum(patch.get_height() for patch in distribution_axis.patches),
                    1.0,
                )
            self.assertEqual(
                result.trajectory_axes[-1].xaxis.label.get_fontsize(),
                8.0,
            )
            self.assertEqual(
                result.distribution_axes[-1].xaxis.label.get_fontsize(),
                8.0,
            )
            self.assertTrue(
                all(panel.paired_count > 0 for panel in result.audit.panels)
            )
            self.assertTrue(
                all(
                    abs(panel.histogram_weight_sum - 1.0) < 1e-12
                    for panel in result.audit.panels
                )
            )
            self.assertEqual(len(result.audit.cutoff_report_lines), 2)
            self.assertIn(
                "Frequency",
                [item.get_text() for item in result.figure.texts],
            )

        printed = report.getvalue().splitlines()
        self.assertEqual(len(printed), 2)
        self.assertIn("2012-01-02", printed[0])
        self.assertIn("2012-05-01", printed[1])

    def test_renderer_restores_archived_paper_artist_contract(self) -> None:
        spec = figure_spec(
            profile="paper_v1",
            canvas="double_over_1_9",
            output_name="paper/test-emd.pdf",
            horizon_months=9,
            comparator_method=MLR_METHOD,
        )
        result = render_emd_comparison_figure(
            EMD,
            horizon_months=9,
            comparator_method=MLR_METHOD,
            figure_spec=spec,
            report_stream=io.StringIO(),
        )
        result.figure.canvas.draw()

        first_trajectory = result.trajectory_axes[0].get_position()
        last_trajectory = result.trajectory_axes[-1].get_position()
        first_distribution = result.distribution_axes[0].get_position()
        self.assertAlmostEqual(first_trajectory.x0, 0.09)
        self.assertAlmostEqual(first_trajectory.x1, 0.65)
        self.assertAlmostEqual(first_trajectory.y1, 0.95)
        self.assertAlmostEqual(last_trajectory.y0, 0.15)
        self.assertAlmostEqual(first_distribution.x0, 0.73)
        self.assertAlmostEqual(first_distribution.x1, 0.95)
        first_gap = (
            result.trajectory_axes[0].get_position().y0
            - result.trajectory_axes[1].get_position().y1
        )
        self.assertAlmostEqual(first_gap / first_trajectory.height, 0.40)

        for trajectory_axis, distribution_axis in zip(
            result.trajectory_axes,
            result.distribution_axes,
            strict=True,
        ):
            self.assertEqual(trajectory_axis.get_title(), "")
            self.assertEqual(len(trajectory_axis.lines), 6)
            data_line, *guide_lines = trajectory_axis.lines
            self.assertAlmostEqual(data_line.get_linewidth(), 1.5)
            self.assertEqual(data_line.get_alpha(), 1.0)
            self.assertEqual(data_line.get_linestyle(), "-")
            self.assertEqual(
                tuple(float(line.get_ydata()[0]) for line in guide_lines),
                TRAJECTORY_GUIDE_LEVELS,
            )
            for guide_level, guide_line in zip(
                TRAJECTORY_GUIDE_LEVELS,
                guide_lines,
                strict=True,
            ):
                self.assertAlmostEqual(guide_line.get_linewidth(), 0.6)
                self.assertEqual(guide_line.get_linestyle(), ":")
                self.assertEqual(guide_line.get_color(), "#969696")
                self.assertAlmostEqual(
                    guide_line.get_alpha(),
                    0.70 if guide_level == 0.0 else 0.20,
                )

            self.assertEqual(len(distribution_axis.lines), 1)
            zero_line = distribution_axis.lines[0]
            self.assertAlmostEqual(zero_line.get_linewidth(), 0.6)
            self.assertEqual(zero_line.get_linestyle(), ":")
            self.assertAlmostEqual(zero_line.get_alpha(), 0.70)
            for bar in distribution_axis.patches:
                self.assertAlmostEqual(bar.get_width(), 0.20)
                self.assertEqual(bar.get_alpha(), 1.0)
                self.assertAlmostEqual(bar.get_linewidth(), 0.30)
                self.assertEqual(bar.get_edgecolor()[-1], 0.0)

            for axis in (trajectory_axis, distribution_axis):
                self.assertFalse(axis.spines["top"].get_visible())
                self.assertFalse(axis.spines["right"].get_visible())
                self.assertTrue(axis.spines["left"].get_visible())
                self.assertTrue(axis.spines["bottom"].get_visible())
                self.assertAlmostEqual(axis.spines["left"].get_linewidth(), 0.4)
                self.assertAlmostEqual(axis.spines["bottom"].get_linewidth(), 0.4)

        figure_text = {item.get_text(): item for item in result.figure.texts}
        for panel_letter in ("A", "B"):
            self.assertEqual(figure_text[panel_letter].get_fontweight(), "bold")
            self.assertEqual(figure_text[panel_letter].get_fontsize(), 10.0)
        for label, color in {
            "H3": "#C35999",
            "N2": "#DDA0C3",
            "H1": "#81AB5B",
            "N1": "#ABCE7D",
        }.items():
            text = figure_text[label]
            self.assertEqual(text.get_fontweight(), "bold")
            self.assertEqual(text.get_fontsize(), 8.0)
            self.assertEqual(text.get_color(), color)
            self.assertGreater(text.get_position()[0], first_distribution.x1)
        p_value_labels = [
            item.get_text()
            for item in result.figure.texts
            if item.get_text().startswith("$p")
        ]
        self.assertEqual(len(p_value_labels), 4)
        for label in p_value_labels:
            self.assertRegex(label, r"^\$p < 10\^\{-?\d+\}\$$")
        self.assertNotIn(FULL_MODEL_WORSE_MARKER_KEY, figure_text)

        self.assertIn("Frequency", figure_text)
        self.assertIn(
            "Model improvement vs. MLR\ndistance to future $(\\Delta EMD)$",
            figure_text,
        )
        self.assertEqual(
            result.trajectory_axes[-1].get_xlabel(),
            "Time prediction made",
        )
        self.assertEqual(
            result.distribution_axes[-1].get_xlabel(),
            "Model improvement vs. MLR\ndistance to future $(\\Delta EMD)$",
        )
        bottom_date_labels = [
            item.get_text()
            for item in result.trajectory_axes[-1].get_xticklabels()
        ]
        self.assertEqual(len(bottom_date_labels), 8)
        self.assertEqual(bottom_date_labels[0], "Jan 2012")
        for label in bottom_date_labels:
            self.assertRegex(label, r"^[A-Z][a-z]{2} \d{4}$")
        for axis in result.trajectory_axes[:-1]:
            self.assertTrue(
                all(not item.get_visible() for item in axis.get_xticklabels())
            )

        self.assertGreater(DATE_AXIS_PADDING_DAYS, 0)
        for axis in result.trajectory_axes:
            left_limit, right_limit = axis.get_xlim()
            ticks = axis.get_xticks()
            self.assertAlmostEqual(ticks[0] - left_limit, DATE_AXIS_PADDING_DAYS)
            self.assertAlmostEqual(right_limit - ticks[-1], DATE_AXIS_PADDING_DAYS)

    def test_renderer_omits_nonsignificant_p_values_and_flags_worse_results(
        self,
    ) -> None:
        spec = figure_spec(
            profile="paper_v1",
            canvas="double_over_1_9",
            output_name="supplement/test-emd-slm.pdf",
            horizon_months=6,
            comparator_method=SLM_METHOD,
        )
        result = render_emd_comparison_figure(
            EMD,
            horizon_months=6,
            comparator_method=SLM_METHOD,
            figure_spec=spec,
            report_stream=io.StringIO(),
        )

        # H3 and N2 significantly favor the full model, H1 significantly
        # favors SLM, and N1 is not significant. Because figure.text artists
        # preserve insertion order, this also verifies that the N1 annotation
        # is omitted rather than replaced with an empty artist.
        p_value_labels = [
            item.get_text()
            for item in result.figure.texts
            if item.get_text().startswith("$p")
        ]
        self.assertEqual(
            p_value_labels,
            [
                r"$p < 10^{-8}$",
                r"$p < 10^{-3}$",
                r"$p = 0.04$*",
            ],
        )
        self.assertEqual(
            [panel.mean_delta_emd < 0.0 for panel in result.audit.panels],
            [False, False, True, True],
        )
        self.assertLessEqual(result.audit.panels[2].p_value, 0.05)
        self.assertGreater(result.audit.panels[3].p_value, 0.05)
        figure_text = {item.get_text(): item for item in result.figure.texts}
        self.assertIn(FULL_MODEL_WORSE_MARKER_KEY, figure_text)
        marker_key = figure_text[FULL_MODEL_WORSE_MARKER_KEY]
        self.assertEqual(marker_key.get_fontsize(), 7.0)
        self.assertEqual(marker_key.get_ha(), "right")
        self.assertEqual(
            marker_key.get_position(),
            (
                LEGACY_DISTRIBUTION_BOUNDS["right"],
                LEGACY_DISTRIBUTION_BOUNDS["top"] + 0.02,
            ),
        )

    def test_histograms_use_safe_nonclipping_frequency_scales(self) -> None:
        cases = (
            (6, NULL_METHOD, "double_over_2_2", 0.30),
            (6, SLM_METHOD, "double_over_1_9", 0.40),
        )
        for horizon, comparator, canvas, expected_limit in cases:
            with self.subTest(horizon=horizon, comparator=comparator):
                spec = figure_spec(
                    profile="paper_v1",
                    canvas=canvas,
                    output_name="diagnostic/test-emd-histogram.pdf",
                    horizon_months=horizon,
                    comparator_method=comparator,
                )
                result = render_emd_comparison_figure(
                    EMD,
                    horizon_months=horizon,
                    comparator_method=comparator,
                    figure_spec=spec,
                    report_stream=io.StringIO(),
                )
                self.assertEqual(
                    result.audit.histogram_frequency_limit,
                    expected_limit,
                )
                for axis in result.distribution_axes:
                    self.assertEqual(axis.get_ylim(), (0.0, expected_limit))
                    self.assertLessEqual(
                        max(bar.get_height() for bar in axis.patches),
                        axis.get_ylim()[1],
                    )

    def test_best_vs_baseline_diagnostics_use_registered_scale_and_label(self) -> None:
        for horizon in (6, 9, 12):
            with self.subTest(horizon=horizon):
                spec = FIGURE_REGISTRY.get(
                    f"diagnostic.emd_best_vs_baseline.{horizon}m"
                )
                result = render_emd_comparison_figure(
                    EMD,
                    horizon_months=horizon,
                    comparator_method=BASELINE_FULL_MODEL_METHOD,
                    figure_spec=spec,
                    report_stream=io.StringIO(),
                )

                self.assertEqual(result.audit.figure_key, spec.key)
                self.assertEqual(result.audit.target_method, FULL_MODEL_METHOD)
                self.assertEqual(
                    result.audit.comparator_method,
                    BASELINE_FULL_MODEL_METHOD,
                )
                self.assertEqual(
                    tuple(panel.paired_count for panel in result.audit.panels),
                    (84, 84, 81, 81),
                )
                self.assertEqual(result.audit.delta_limit_abs, 4.0)
                self.assertEqual(result.audit.distribution_delta_limit_abs, 4.0)
                self.assertEqual(result.audit.histogram_frequency_limit, 0.40)
                self.assertTrue(
                    all(
                        axis.get_ylim() == (-4.0, 4.0)
                        for axis in result.trajectory_axes
                    )
                )
                self.assertTrue(
                    all(
                        axis.get_xlim() == (-4.0, 4.0)
                        for axis in result.distribution_axes
                    )
                )
                self.assertTrue(
                    all(
                        axis.get_ylim() == (0.0, 0.40)
                        for axis in result.distribution_axes
                    )
                )
                expected_label = (
                    "Model improvement vs. baseline\n"
                    "distance to future $(\\Delta EMD)$"
                )
                self.assertEqual(
                    result.distribution_axes[-1].get_xlabel(),
                    expected_label,
                )
                self.assertIn(
                    expected_label,
                    {item.get_text() for item in result.figure.texts},
                )

    def test_renderer_uses_supplied_slide_typography_and_canvas(self) -> None:
        spec = figure_spec(
            profile="slides_v1",
            canvas="slide_emd",
            output_name="slides/test-emd.pdf",
            horizon_months=12,
            comparator_method=NULL_METHOD,
        )
        report = io.StringIO()
        with redirect_stdout(report):
            result = render_emd_comparison_figure(
                EMD,
                horizon_months=12,
                comparator_method=NULL_METHOD,
                figure_spec=spec,
            )
        self.assertIn("1095 days", report.getvalue())
        self.assertIn("2012-01-02", report.getvalue())
        self.assertIn("2012-05-01", report.getvalue())
        self.assertIsNone(result.audit.saved_path)
        self.assertEqual(
            tuple(float(value) for value in result.figure.get_size_inches()),
            (10.5, 10.5 / 1.8),
        )
        self.assertEqual(len(result.axes), 8)
        self.assertEqual(
            result.trajectory_axes[-1].xaxis.label.get_fontsize(),
            18.0,
        )
        self.assertEqual(
            result.distribution_axes[-1].xaxis.label.get_fontsize(),
            18.0,
        )
        self.assertEqual(result.audit.delta_limit_abs, 5.0)
        self.assertEqual(result.audit.distribution_delta_limit_abs, 5.25)
        self.assertTrue(
            all(axis.get_ylim() == (-5.0, 5.0) for axis in result.trajectory_axes)
        )
        self.assertTrue(
            all(
                axis.get_xlim() == (-5.25, 5.25)
                for axis in result.distribution_axes
            )
        )
        self.assertEqual(result.audit.evaluation_protocol, "results_grid")


if __name__ == "__main__":
    unittest.main()
