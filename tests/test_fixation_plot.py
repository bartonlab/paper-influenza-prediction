"""Tests for canonical fixation-probability and AUROC figure recipes."""

from __future__ import annotations

import ast
import csv
from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from influenza_figures.fixation_plot import (  # noqa: E402
    DEFAULT_MINIMUM_OBSERVATION_DAY,
    EXPECTED_FREQUENCIES,
    PANEL_DEFINITIONS,
    SLM_MODEL_ID,
    SLM_SOURCE_VARIANT,
    FixationFigureDataError,
    collect_auc_data,
    collect_fixation_probability_data,
    render_fixation_auc_figure,
    render_pfix_dauc_figure,
)
from influenza_figures.registry import CANVASES, FigureSpec  # noqa: E402


FIXATION_DIR = REPO / "data" / "plotting" / "derived" / "fixation_metrics"
PFIX = FIXATION_DIR / "fixation_probability.csv"
AUC = FIXATION_DIR / "auc.csv"
HAS_PLOTTING = find_spec("matplotlib") is not None


def figure_spec(*, key: str, canvas: str, output_name: str) -> FigureSpec:
    return FigureSpec(
        key=key,
        renderer="test_renderer",
        profile="paper_v1",
        canvas=CANVASES[canvas],
        output_name=output_name,
        category="test",
    )


class FixationDataTests(unittest.TestCase):
    def test_module_has_no_top_level_scientific_plotting_imports(self) -> None:
        source = (
            REPO / "src" / "influenza_figures" / "fixation_plot.py"
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

    def test_data_module_imports_without_scientific_dependencies(self) -> None:
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
import influenza_figures.fixation_plot as module
if not callable(module.collect_auc_data):
    raise AssertionError('AUROC collector is not callable')
"""
        result = subprocess.run(
            [sys.executable, "-I", "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_full_model_selection_is_explicit_complete_and_ordered(self) -> None:
        pfix = collect_fixation_probability_data(PFIX)
        auc = collect_auc_data(AUC)

        self.assertEqual(DEFAULT_MINIMUM_OBSERVATION_DAY, 1095)
        self.assertEqual(
            tuple(panel.definition for panel in pfix.panels), PANEL_DEFINITIONS
        )
        self.assertEqual(
            tuple(panel.definition for panel in auc.panels), PANEL_DEFINITIONS
        )
        self.assertTrue(
            all(panel.frequencies == EXPECTED_FREQUENCIES for panel in pfix.panels)
        )
        self.assertTrue(
            all(panel.frequencies == EXPECTED_FREQUENCIES for panel in auc.panels)
        )
        self.assertEqual(auc.model_id, "full_model")
        self.assertEqual(auc.source_variant, "all_time")
        self.assertEqual(auc.minimum_observation_day, 1095)

        h3_pfix = pfix.panels[0]
        self.assertAlmostEqual(h3_pfix.top_25_score[0], 0.2162698412698412)
        self.assertAlmostEqual(h3_pfix.bottom_25_score[-1], 0.7943107221006565)
        h3_auc = auc.panels[0]
        self.assertAlmostEqual(h3_auc.model[3], 0.8304046712252088)
        self.assertAlmostEqual(h3_auc.null[3], 0.44719078925706)

    def test_slm_figure_selects_paper_date_filtered_not_diagnostic_variant(self) -> None:
        paper = collect_auc_data(
            AUC,
            model_id=SLM_MODEL_ID,
            source_variant=SLM_SOURCE_VARIANT,
        )
        diagnostic = collect_auc_data(
            AUC,
            model_id=SLM_MODEL_ID,
            source_variant="extended_all_time",
        )

        self.assertEqual(paper.source_variant, "paper_date_filtered")
        self.assertEqual(diagnostic.source_variant, "extended_all_time")
        # These two archived source variants happen to contain the same AUROC
        # values, but their lineage and accompanying test statistics differ;
        # the manuscript selector must therefore remain explicit.
        self.assertEqual(paper.panels[0].model, diagnostic.panels[0].model)
        self.assertAlmostEqual(paper.panels[0].model[0], 0.574526986541298)

    def test_row_order_does_not_change_selected_series(self) -> None:
        expected = collect_auc_data(AUC)
        with AUC.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or ())

        with tempfile.TemporaryDirectory() as directory:
            reversed_path = Path(directory) / "auc-reversed.csv"
            with reversed_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(reversed(rows))
            observed = collect_auc_data(reversed_path)

        self.assertEqual(expected.panels, observed.panels)

    def test_missing_frequency_bin_is_rejected(self) -> None:
        with AUC.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or ())
        rows = [
            row
            for row in rows
            if not (
                row["model_id"] == "full_model"
                and row["source_variant"] == "all_time"
                and row["subtype"] == "h3n2"
                and row["segment"] == "ha"
                and row["minimum_observation_day"] == "1095"
                and row["frequency"] == "0.9"
            )
        ]

        with tempfile.TemporaryDirectory() as directory:
            incomplete = Path(directory) / "auc-incomplete.csv"
            with incomplete.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(
                FixationFigureDataError, "incomplete or unexpected frequency grid"
            ):
                collect_auc_data(incomplete)


@unittest.skipUnless(HAS_PLOTTING, "Matplotlib unavailable")
class FixationRenderTests(unittest.TestCase):
    def tearDown(self) -> None:
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_main_figure_3_matches_registered_structure_and_recipe(self) -> None:
        spec = figure_spec(
            key="paper.pfix_dauc",
            canvas="double_over_2_8",
            output_name="paper/fig-pfix-dauc.pdf",
        )
        result = render_pfix_dauc_figure(PFIX, AUC, figure_spec=spec)

        self.assertEqual(result.figure.get_size_inches().tolist(), list(spec.canvas.size_inches))
        self.assertEqual(len(result.axes), 8)
        self.assertEqual(result.audit.layout, "four_columns_by_two_rows")
        self.assertEqual(result.audit.model_id, "full_model")
        self.assertEqual(result.audit.source_variant, "all_time")
        self.assertEqual(result.audit.minimum_observation_day, 1095)
        self.assertEqual(
            [panel.label for panel in result.audit.panels],
            ["H3", "N2", "H1", "N1"],
        )
        self.assertTrue(
            all(panel.fixation_probability_row_count == 9 for panel in result.audit.panels)
        )
        self.assertTrue(all(panel.auc_row_count == 9 for panel in result.audit.panels))

        top_axes = result.axes[:4]
        bottom_axes = result.axes[4:]
        self.assertTrue(all(len(axis.lines) == 6 for axis in top_axes))
        self.assertTrue(all(len(axis.collections) == 1 for axis in bottom_axes))
        self.assertTrue(
            all(axis.get_xlabel() == "Allele frequency" for axis in bottom_axes)
        )
        self.assertIn("A", [text.get_text() for text in result.figure.texts])
        self.assertIn("B", [text.get_text() for text in result.figure.texts])

    def test_s7_and_s16_share_layout_but_use_distinct_explicit_data(self) -> None:
        spec = figure_spec(
            key="paper.auc",
            canvas="single_tall_1_5",
            output_name="supplement/fig-fixation-auc.pdf",
        )
        full = render_fixation_auc_figure(AUC, figure_spec=spec)
        slm = render_fixation_auc_figure(
            AUC,
            figure_spec=spec,
            model_id=SLM_MODEL_ID,
            source_variant=SLM_SOURCE_VARIANT,
        )

        for result in (full, slm):
            self.assertEqual(len(result.axes), 4)
            self.assertEqual(result.audit.layout, "two_by_two")
            self.assertEqual(
                result.figure.get_size_inches().tolist(),
                list(spec.canvas.size_inches),
            )
            self.assertTrue(all(len(axis.patches) == 18 for axis in result.axes))

        self.assertEqual(full.audit.source_variant, "all_time")
        self.assertEqual(slm.audit.source_variant, "paper_date_filtered")
        self.assertIn(
            "H3 model", [text.get_text() for text in full.axes[0].texts]
        )
        self.assertIn(
            "H3 single site", [text.get_text() for text in slm.axes[0].texts]
        )
        self.assertNotEqual(full.auc_data.panels[0].model, slm.auc_data.panels[0].model)

    def test_optional_output_records_resolved_saved_path(self) -> None:
        spec = figure_spec(
            key="test.auc",
            canvas="single_tall_1_5",
            output_name="test/auc.png",
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nested" / "auc.png"
            result = render_fixation_auc_figure(
                AUC,
                figure_spec=spec,
                output_path=destination,
            )
            self.assertTrue(destination.is_file())
            self.assertEqual(result.audit.saved_path, str(destination.resolve()))


if __name__ == "__main__":
    unittest.main()

