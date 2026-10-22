"""Tests for the readable root-level plotting facade."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import figures
from influenza_figures.emd_plot import (
    BASELINE_FULL_MODEL_METHOD,
    FULL_MODEL_METHOD,
    MLR_METHOD,
    NULL_METHOD,
    SLM_METHOD,
)


class FiguresFacadeTests(unittest.TestCase):
    def test_human_emd_names_map_to_correct_methods(self) -> None:
        self.assertEqual(figures._emd_method("MLR"), ("mlr", MLR_METHOD))
        self.assertEqual(figures._emd_method("SLM"), ("slm", SLM_METHOD))
        self.assertEqual(figures._emd_method("null"), ("null", NULL_METHOD))
        self.assertEqual(
            MLR_METHOD,
            "multinomial_logistic_regression_rare_filtered",
        )

    def test_emd_wrapper_binds_visible_choice_to_registered_recipe(self) -> None:
        marker = object()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "emd.pdf"
            with patch.object(
                figures,
                "render_emd_comparison_figure",
                return_value=marker,
            ) as renderer:
                result = figures.plot_emd_comparison(
                    "MLR",
                    9,
                    output_file=output,
                )

        self.assertIs(result, marker)
        _, keyword = renderer.call_args
        self.assertEqual(keyword["horizon_months"], 9)
        self.assertEqual(keyword["comparator_method"], MLR_METHOD)
        self.assertEqual(keyword["figure_spec"].key, "paper.emd_vs_mlr.9m")
        self.assertEqual(keyword["output_path"], output.resolve())

    def test_best_vs_baseline_wrapper_binds_each_horizon_diagnostic(self) -> None:
        marker = object()
        for horizon in (6, 9, 12):
            with self.subTest(horizon=horizon):
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "emd-best-vs-baseline.pdf"
                    with patch.object(
                        figures,
                        "render_emd_comparison_figure",
                        return_value=marker,
                    ) as renderer:
                        result = figures.plot_emd_best_vs_baseline(
                            horizon,
                            output_file=output,
                        )

                self.assertIs(result, marker)
                _, keyword = renderer.call_args
                self.assertEqual(keyword["horizon_months"], horizon)
                self.assertEqual(
                    keyword["comparator_method"],
                    BASELINE_FULL_MODEL_METHOD,
                )
                self.assertEqual(
                    keyword["figure_spec"].key,
                    f"diagnostic.emd_best_vs_baseline.{horizon}m",
                )
                self.assertEqual(keyword["output_path"], output.resolve())

    def test_nine_month_best_vs_baseline_compatibility_wrapper(self) -> None:
        marker = object()
        with patch.object(
            figures,
            "plot_emd_best_vs_baseline",
            return_value=marker,
        ) as generalized:
            result = figures.plot_emd_best_vs_baseline_9m(overwrite=False)

        self.assertIs(result, marker)
        positional, keyword = generalized.call_args
        self.assertEqual(positional, (9,))
        self.assertFalse(keyword["overwrite"])

    def test_unvalidated_presentation_or_comparator_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "version='paper'"):
            figures.plot_emd_comparison("MLR", 9, version="slides")
        with self.assertRaisesRegex(ValueError, "null.*SLM.*MLR"):
            figures.plot_emd_comparison("FMSS", 9)
        with self.assertRaisesRegex(ValueError, "6, 9, or 12"):
            figures.plot_emd_best_vs_baseline(3)

    def test_visible_presentation_constants_match_historical_canvases(self) -> None:
        self.assertEqual(figures.PAPER_STYLE, "paper_v1")
        self.assertEqual(figures.SLIDES_STYLE, "slides_v1")
        self.assertEqual(figures.FREQUENCY_SLIDE_CANVAS_INCHES, (8.5, 1.6))
        self.assertAlmostEqual(figures.CLADE_PAPER_CANVAS_INCHES[0], 8.8 / 2.54, places=4)
        self.assertEqual(
            figures.FULL_MODEL_SOURCE_BY_HORIZON[9],
            "cross-validation-selected best GenModel",
        )
        self.assertEqual(
            set(figures.FULL_MODEL_SOURCE_BY_HORIZON.values()),
            {"cross-validation-selected best GenModel"},
        )

    def test_selection_wrappers_bind_canonical_inputs_and_registered_recipes(self) -> None:
        marker = object()
        recipes = (
            (
                "plot_selection_structure_space",
                "render_selection_structure_space_figure",
                "paper.selection_structure_space",
                True,
            ),
            (
                "plot_frequency_selection",
                "render_frequency_selection_figure",
                "paper.frequency_selection",
                True,
            ),
            (
                "plot_selection_sasa",
                "render_selection_sasa_figure",
                "paper.selection_sasa",
                False,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, (wrapper_name, renderer_name, figure_key, has_filter) in enumerate(
                recipes
            ):
                with self.subTest(wrapper=wrapper_name):
                    output = Path(directory) / f"selection-{index}.pdf"
                    with patch.object(
                        figures,
                        renderer_name,
                        return_value=marker,
                    ) as renderer:
                        result = getattr(figures, wrapper_name)(output_file=output)

                    self.assertIs(result, marker)
                    positional, keyword = renderer.call_args
                    self.assertEqual(positional, (figures.PLOTTING_DATA.resolve(),))
                    self.assertEqual(keyword["figure_spec"].key, figure_key)
                    self.assertEqual(keyword["output_path"], output.resolve())
                    if has_filter:
                        self.assertEqual(
                            keyword["minimum_mutation_count"],
                            figures.MIN_MUTATION_COUNT,
                        )

    def test_fixation_wrappers_make_final_model_choices_explicit(self) -> None:
        marker = object()
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory)
            with patch.object(
                figures,
                "render_pfix_dauc_figure",
                return_value=marker,
            ) as renderer:
                result = figures.plot_pfix_dauc(
                    output_file=output_directory / "pfix-dauc.pdf"
                )
            self.assertIs(result, marker)
            positional, keyword = renderer.call_args
            self.assertEqual(
                positional,
                (
                    figures.FIXATION_PROBABILITY_DATA.resolve(),
                    figures.FIXATION_AUC_DATA.resolve(),
                ),
            )
            self.assertEqual(keyword["figure_spec"].key, "paper.pfix_dauc")
            self.assertEqual(keyword["model_id"], "full_model")
            self.assertEqual(keyword["source_variant"], "all_time")
            self.assertEqual(keyword["minimum_observation_day"], 1095)

            auc_recipes = (
                (
                    figures.plot_fixation_auc,
                    "paper.auc",
                    "full_model",
                    "all_time",
                    "model",
                ),
                (
                    figures.plot_single_site_auc,
                    "paper.auc_ss",
                    "slm",
                    "paper_date_filtered",
                    "single site",
                ),
            )
            for index, (wrapper, figure_key, model_id, source_variant, label) in enumerate(
                auc_recipes
            ):
                with self.subTest(wrapper=wrapper.__name__):
                    with patch.object(
                        figures,
                        "render_fixation_auc_figure",
                        return_value=marker,
                    ) as renderer:
                        result = wrapper(
                            output_file=output_directory / f"auc-{index}.pdf"
                        )
                    self.assertIs(result, marker)
                    positional, keyword = renderer.call_args
                    self.assertEqual(positional, (figures.FIXATION_AUC_DATA.resolve(),))
                    self.assertEqual(keyword["figure_spec"].key, figure_key)
                    self.assertEqual(keyword["model_id"], model_id)
                    self.assertEqual(keyword["source_variant"], source_variant)
                    self.assertEqual(keyword["model_display_label"], label)
                    self.assertEqual(keyword["minimum_observation_day"], 1095)

    def test_background_wrappers_use_event_preserving_canonical_table(self) -> None:
        marker = object()
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory)
            with patch.object(
                figures,
                "render_background_clade_figure",
                return_value=marker,
            ) as renderer:
                result = figures.plot_background_clade(
                    output_file=output_directory / "background-clade.pdf"
                )
            self.assertIs(result, marker)
            positional, keyword = renderer.call_args
            self.assertEqual(
                positional,
                (figures.BACKGROUND_DATA.resolve(), figures.CLADE_DATA.resolve()),
            )
            self.assertEqual(keyword["figure_spec"].key, "paper.dw_clade")
            self.assertEqual(keyword["minimum_total_frequency"], 1.0)
            self.assertEqual(keyword["start_date"], "2011-08-01")
            self.assertEqual(keyword["end_date"], "2015-06-01")
            self.assertEqual(keyword["forecast_baseline_date"], "2014-06-01")

            recipes = (
                (
                    figures.plot_background_selection,
                    "render_background_selection_figure",
                    "paper.ds",
                ),
                (
                    figures.plot_background_initial_advantage,
                    "render_background_initial_advantage_figure",
                    "paper.dw_bg",
                ),
                (
                    figures.plot_background_growth,
                    "render_background_growth_figure",
                    "paper.dw_rate",
                ),
            )
            for index, (wrapper, renderer_name, figure_key) in enumerate(recipes):
                with self.subTest(wrapper=wrapper.__name__):
                    with patch.object(
                        figures,
                        renderer_name,
                        return_value=marker,
                    ) as renderer:
                        result = wrapper(
                            output_file=output_directory / f"background-{index}.pdf"
                        )
                    self.assertIs(result, marker)
                    positional, keyword = renderer.call_args
                    self.assertEqual(positional, (figures.BACKGROUND_DATA.resolve(),))
                    self.assertEqual(keyword["figure_spec"].key, figure_key)

    def test_new_manuscript_wrappers_are_public_and_paper_only(self) -> None:
        wrapper_names = {
            "plot_selection_structure_space",
            "plot_pfix_dauc",
            "plot_background_clade",
            "plot_frequency_selection",
            "plot_selection_sasa",
            "plot_fixation_auc",
            "plot_background_selection",
            "plot_background_initial_advantage",
            "plot_background_growth",
            "plot_single_site_auc",
        }
        self.assertTrue(wrapper_names.issubset(figures.__all__))
        self.assertEqual(
            set(figures.PLOT_SOURCE_FILES),
            {"background", "emd", "fixation", "legacy_figures", "selection"},
        )
        with self.assertRaisesRegex(ValueError, "version='paper'"):
            figures.plot_frequency_selection(version="slides")


SCIENTIFIC_PLOTTING_AVAILABLE = all(
    find_spec(name) is not None
    for name in ("numpy", "scipy", "matplotlib", "pandas", "seaborn")
)


@unittest.skipUnless(
    SCIENTIFIC_PLOTTING_AVAILABLE,
    "scientific plotting dependencies unavailable",
)
class FiguresFacadeRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import matplotlib

        matplotlib.use("Agg")

    def test_canonical_emd_wrapper_renders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "emd.pdf"
            result = figures.plot_emd_comparison(
                "MLR",
                9,
                output_file=output,
                overwrite=False,
            )
            self.assertGreater(output.stat().st_size, 10_000)
            self.assertEqual(
                tuple(panel.paired_count for panel in result.audit.panels),
                (84, 84, 81, 81),
            )

    def test_best_vs_baseline_diagnostic_wrappers_render(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for horizon in (6, 9, 12):
                with self.subTest(horizon=horizon):
                    output = Path(directory) / f"emd-best-vs-baseline-{horizon}.pdf"
                    result = figures.plot_emd_best_vs_baseline(
                        horizon,
                        output_file=output,
                        overwrite=False,
                    )
                    self.assertGreater(output.stat().st_size, 10_000)
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
                    self.assertEqual(result.audit.histogram_frequency_limit, 0.40)

    def test_clade_and_frequency_wrappers_render(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory)
            clade_output = output_directory / "clade.pdf"
            audit = figures.plot_clade_figure(
                output_file=clade_output,
                overwrite=False,
            )
            self.assertGreater(clade_output.stat().st_size, 10_000)
            self.assertEqual(
                audit["selected_clades"],
                ("3C", "3C.2", "3C.2a", "3C.3", "3C.3a", "3C.3b"),
            )

            frequency_output = output_directory / "frequency.pdf"
            source = (
                figures.PLOTTING_DATA
                / "inputs"
                / "frequencies"
                / "h3n2_ha_allele_frequencies.txt"
            )
            figures.plot_frequency_slide(
                source,
                subtype="h3n2",
                segment="ha",
                time_shift_days=0,
                output_file=frequency_output,
                overwrite=False,
            )
            self.assertGreater(frequency_output.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
