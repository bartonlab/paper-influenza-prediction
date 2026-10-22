"""Checks for the editable, manuscript-ordered figure-recipe notebook."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "create_figures_notebook.py"
NOTEBOOK = REPO / "figures.ipynb"
ARCHIVED = (
    REPO
    / "archive"
    / "pre_plot_reorg_2026-08-26"
    / "original"
    / "figures.ipynb"
)

EXPECTED_ORDER = (
    "Main Figure 1",
    "Main Figure 2",
    "Main Figure 3",
    "Main Figure 4",
    "Main Figure 5",
    "Supplementary Figure S1",
    "Supplementary Figure S2",
    "Supplementary Figure S3",
    "Supplementary Figure S4",
    "Supplementary Figure S5",
    "Supplementary Figure S6",
    "Supplementary Figure S7",
    "Supplementary Figure S8",
    "Supplementary Figure S9",
    "Supplementary Figure S10",
    "Supplementary Figure S11",
    "Supplementary Figure S12",
    "Supplementary Figure S13",
    "Supplementary Figure S14",
    "Supplementary Figure S15",
    "Supplementary Figure S16",
    "Supplementary Figure S17",
    "Supplementary Figure S18",
    "Supplementary Figure S19",
)

EXPECTED_STATUS = {
    "Main Figure 1": "EDITABLE ARTWORK",
    "Main Figure 2": "RUNNABLE",
    "Main Figure 3": "RUNNABLE",
    "Main Figure 4": "RUNNABLE",
    "Main Figure 5": "RUNNABLE",
    "Supplementary Figure S1": "MISSING SOURCE",
    "Supplementary Figure S2": "OUT OF SCOPE",
    "Supplementary Figure S3": "OUT OF SCOPE",
    "Supplementary Figure S4": "OUT OF SCOPE",
    "Supplementary Figure S5": "RUNNABLE",
    "Supplementary Figure S6": "RUNNABLE",
    "Supplementary Figure S7": "RUNNABLE",
    "Supplementary Figure S8": "RUNNABLE",
    "Supplementary Figure S9": "RUNNABLE",
    "Supplementary Figure S10": "RUNNABLE",
    "Supplementary Figure S11": "RUNNABLE",
    "Supplementary Figure S12": "RUNNABLE",
    "Supplementary Figure S13": "RUNNABLE",
    "Supplementary Figure S14": "RUNNABLE",
    "Supplementary Figure S15": "RUNNABLE",
    "Supplementary Figure S16": "RUNNABLE",
    "Supplementary Figure S17": "RUNNABLE",
    "Supplementary Figure S18": "RUNNABLE",
    "Supplementary Figure S19": "RUNNABLE",
}


class FiguresNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("create_figures_notebook", SCRIPT)
        assert spec is not None and spec.loader is not None
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        spec.loader.exec_module(cls.module)
        # Test the generator directly. The working notebook is intentionally not
        # overwritten merely because its template changed.
        cls.notebook = cls.module.build_notebook()
        cls.cells_by_id = {cell["id"]: cell for cell in cls.notebook["cells"]}

    def test_archived_notebook_sources_remain_pinned_and_distinct(self) -> None:
        self.assertEqual(
            self.module.notebook_source_sha256(ARCHIVED),
            self.module.ARCHIVED_SOURCE_SHA256,
        )
        self.assertNotEqual(
            self.module.sha256_file(NOTEBOOK),
            self.module.ARCHIVED_ORIGINAL_SHA256,
        )

    def test_generated_notebook_has_clean_editable_cells(self) -> None:
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertGreaterEqual(len(self.notebook["cells"]), 40)
        ids = [cell["id"] for cell in self.notebook["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                compile(cell["source"], f"generated:{cell['id']}", "exec")

    def test_manuscript_order_is_exact_and_diagnostics_follow_s19(self) -> None:
        self.assertEqual(self.module.MANUSCRIPT_FIGURE_ORDER, EXPECTED_ORDER)
        observed = []
        for figure in self.module.MANUSCRIPT_FIGURES:
            source = self.cells_by_id[figure["cell_id"]]["source"]
            heading = source.splitlines()[0]
            self.assertTrue(heading.startswith(f"## {figure['label']} - "))
            observed.append(heading[3:].split(" - ", 1)[0])
        self.assertEqual(tuple(observed), EXPECTED_ORDER)

        ids = [cell["id"] for cell in self.notebook["cells"]]
        self.assertLess(ids.index("main000"), ids.index("main001"))
        self.assertLess(ids.index("main005"), ids.index("supp000"))
        self.assertLess(ids.index("supp019"), ids.index("diag000"))

    def test_status_classification_and_inventory_contract_are_explicit(self) -> None:
        self.assertEqual(self.module.MANUSCRIPT_FIGURE_STATUS, EXPECTED_STATUS)
        for figure in self.module.MANUSCRIPT_FIGURES:
            source = self.cells_by_id[figure["cell_id"]]["source"]
            self.assertIn(f"**Status: {EXPECTED_STATUS[figure['label']]}**", source)
            self.assertIn("Canonical output:", source)
            self.assertIn("Entry point:", source)
            self.assertIn("Statistical readout:", source)

    def test_every_in_scope_manuscript_figure_has_one_explicit_cell(self) -> None:
        runnable_cells = {
            "main001run",
            "main002run",
            "main003run",
            "main004run",
            "main005run",
            "supp005run",
            "supp006run",
            "supp007run",
            "supp008run",
            "supp009run",
            "supp010run",
            "supp011run",
            "supp012run",
            "supp013run",
            "supp014run",
            "supp015run",
            "supp016run",
            "supp017run",
            "supp018run",
            "supp019run",
        }
        observed = {
            f"{figure['cell_id']}run"
            for figure in self.module.MANUSCRIPT_FIGURES
            if f"{figure['cell_id']}run" in self.cells_by_id
        }
        self.assertEqual(observed, runnable_cells)
        for figure in self.module.MANUSCRIPT_FIGURES:
            run_id = f"{figure['cell_id']}run"
            self.assertEqual(
                run_id in observed,
                figure["status"] in {"RUNNABLE", "EDITABLE ARTWORK"},
            )

    def test_each_manuscript_emd_cell_has_one_explicit_horizon(self) -> None:
        expected = {
            "main004run": ("null", 9, "figures/paper/fig-emd-vs-null-9m.pdf"),
            "supp008run": (
                "null",
                6,
                "figures/supplement/fig-emd-vs-null-6m.pdf",
            ),
            "supp009run": (
                "null",
                12,
                "figures/supplement/fig-emd-vs-null-12m.pdf",
            ),
            "supp010run": (
                "MLR",
                6,
                "figures/supplement/fig-emd-vs-mlr-6m.pdf",
            ),
            "supp011run": (
                "MLR",
                9,
                "figures/supplement/fig-emd-vs-mlr-9m.pdf",
            ),
            "supp012run": (
                "MLR",
                12,
                "figures/supplement/fig-emd-vs-mlr-12m.pdf",
            ),
            "supp017run": (
                "SLM",
                6,
                "figures/supplement/fig-emd-vs-slm-6m.pdf",
            ),
            "supp018run": (
                "SLM",
                9,
                "figures/supplement/fig-emd-vs-slm-9m.pdf",
            ),
            "supp019run": (
                "SLM",
                12,
                "figures/supplement/fig-emd-vs-slm-12m.pdf",
            ),
        }
        for cell_id, (comparator, horizon, output) in expected.items():
            source = self.cells_by_id[cell_id]["source"]
            self.assertIn(f'COMPARATOR = "{comparator}"', source)
            self.assertIn(f"HORIZON_MONTHS = {horizon}", source)
            self.assertNotIn("HORIZONS_MONTHS", source)
            self.assertNotIn("for horizon in", source)
            self.assertEqual(source.count("f.plot_emd_comparison("), 1)
            self.assertIn("FIXED_RECIPE_METADATA", source)
            self.assertIn('EVALUATION_PROTOCOL = "results_grid"', source)
            for part in output.split("/"):
                self.assertIn(f'"{part}"', source)

    def test_emd_statistics_are_visible_for_manuscript_and_diagnostics(self) -> None:
        setup = self.cells_by_id["setup001"]["source"]
        for fragment in (
            "def display_emd_statistics(result):",
            '"paired_n": panel.paired_count',
            '"mean_delta_emd": panel.mean_delta_emd',
            '"t_statistic": panel.t_statistic',
            '"p_value": panel.p_value',
            '"significant_at_0.05": significant',
            '"full_model_worse": significant and panel.mean_delta_emd < 0',
        ):
            self.assertIn(fragment, setup)

        emd_cell_ids = (
            "main004run",
            "supp008run",
            "supp009run",
            "supp010run",
            "supp011run",
            "supp012run",
            "supp017run",
            "supp018run",
            "supp019run",
            "diagbase6",
            "diagbase9",
            "diagbase12",
        )
        for cell_id in emd_cell_ids:
            source = self.cells_by_id[cell_id]["source"]
            self.assertIn("STATISTICS = display_emd_statistics(result)", source)

    def test_only_best_vs_baseline_emd_remains_diagnostic(self) -> None:
        for cell_id, horizon in (
            ("diagbase6", 6),
            ("diagbase9", 9),
            ("diagbase12", 12),
        ):
            source = self.cells_by_id[cell_id]["source"]
            self.assertIn(f"HORIZON_MONTHS = {horizon}", source)
            self.assertEqual(source.count("f.plot_emd_best_vs_baseline("), 1)
            self.assertIn('"figures" / "diagnostic"', source)

        self.assertNotIn("diagslm6run", self.cells_by_id)
        self.assertNotIn("diagslm9run", self.cells_by_id)
        source = "\n".join(cell["source"] for cell in self.notebook["cells"])
        self.assertNotIn("Regularization robustness", source)
        self.assertNotIn("Rolling-origin cross-validation timeline", source)

    def test_notebook_preserves_other_reviewed_direct_recipes(self) -> None:
        source = "\n".join(cell["source"] for cell in self.notebook["cells"])
        required_fragments = (
            "SAVE = True",
            "OVERWRITE = True",
            "INPUT_DIRECTORY = ROOT",
            "INPUT_FILE = ROOT",
            "OUTPUT_FILE = ROOT",
            "FIGSIZE_INCHES",
            "WARMUP_DAYS = 1095",
            "FIXED_RECIPE_METADATA",
            "f.plot_selection_structure_space(",
            "f.plot_pfix_dauc(",
            "f.plot_background_clade(",
            "f.plot_frequency_selection(",
            "f.plot_selection_sasa(",
            "f.plot_fixation_auc(",
            "f.plot_emd_comparison(",
            "f.plot_background_selection(",
            "f.plot_background_initial_advantage(",
            "f.plot_background_growth(",
            "f.plot_single_site_auc(",
            "f.plot_emd_best_vs_baseline(",
            "f.plot_clade_figure(",
            "f.plot_frequency_slide(",
            "multinomial logistic regression",
            "emd_v6",
            "emd_migration_v6",
            "2026-10-13",
            "no lower-bound",
            "fig-model-overview.pptx",
            "full_model_worse",
        )
        for fragment in required_fragments:
            self.assertIn(fragment, source)
        self.assertNotIn("multilinear regression", source.lower())
        self.assertNotIn("multilinear_regression", source)

    def test_recipe_cells_use_only_canonical_active_paths(self) -> None:
        code = "\n".join(
            cell["source"]
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn('"data" / "plotting" / "inputs"', code)
        self.assertIn('"figures" / "paper"', code)
        self.assertIn('"figures" / "supplement"', code)
        self.assertIn('"figures" / "diagnostic"', code)
        self.assertIn('"figures" / "slides"', code)
        self.assertNotIn('"archive"', code)
        self.assertNotIn("figure_manifest.yaml", code)
        self.assertNotIn("FIGURE_REGISTRY", code)
        self.assertNotIn("render_emd_comparison_figure", code)

    def test_each_non_emd_recipe_calls_one_named_facade(self) -> None:
        expected = {
            "main002run": ("plot_selection_structure_space", "fig-selection-structure-space.pdf"),
            "main003run": ("plot_pfix_dauc", "fig-pfix-dauc.pdf"),
            "main005run": ("plot_background_clade", "fig-dw-clade.pdf"),
            "supp005run": ("plot_frequency_selection", "fig-frequency-selection.pdf"),
            "supp006run": ("plot_selection_sasa", "fig-selection-sasa.pdf"),
            "supp007run": ("plot_fixation_auc", "fig-fixation-auc.pdf"),
            "supp013run": ("plot_background_selection", "fig-ds.pdf"),
            "supp014run": ("plot_background_initial_advantage", "fig-dw-bg.pdf"),
            "supp015run": ("plot_background_growth", "fig-dw-rate.pdf"),
            "supp016run": ("plot_single_site_auc", "fig-auc-ss.pdf"),
        }
        for cell_id, (entrypoint, output_name) in expected.items():
            source = self.cells_by_id[cell_id]["source"]
            self.assertEqual(source.count(f"f.{entrypoint}("), 1)
            self.assertIn(f'"{output_name}"', source)
            self.assertIn("AUDIT = display_figure_audit(result)", source)
            self.assertIn("overwrite=OVERWRITE", source)

    def test_model_overview_is_an_explicit_manual_artwork_recipe(self) -> None:
        source = self.cells_by_id["main001run"]["source"]
        self.assertIn('"artwork" / "fig-model-overview.pptx"', source)
        self.assertIn('"paper" / "fig-model-overview.pdf"', source)
        self.assertIn("do not rescale", source)
        self.assertNotIn("figures.legacy", source)

    def test_out_of_scope_s1_to_s4_have_no_placeholder_calls(self) -> None:
        for cell_id in ("supp001", "supp002", "supp003", "supp004"):
            self.assertNotIn(f"{cell_id}run", self.cells_by_id)
        markdown = "\n".join(
            self.cells_by_id[cell_id]["source"]
            for cell_id in ("supp001", "supp002", "supp003", "supp004")
        )
        self.assertIn("MISSING SOURCE", markdown)
        self.assertIn("OUT OF SCOPE", markdown)
        self.assertIn("supplied final-figure set", markdown)

    def test_generator_refuses_to_overwrite_working_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "figures.ipynb"
            target.write_bytes(b"unarchived user work\n")
            with self.assertRaises(self.module.NotebookReplacementRefused):
                self.module.replace_notebook(target, ARCHIVED)

            self.assertTrue(
                self.module.replace_notebook(target, ARCHIVED, force=True)
            )
            self.assertEqual(target.read_bytes(), self.module.notebook_bytes())
            self.assertFalse(self.module.replace_notebook(target, ARCHIVED))

    def test_archived_original_can_be_migrated_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "figures.ipynb"
            target.write_bytes(ARCHIVED.read_bytes())
            self.assertTrue(self.module.replace_notebook(target, ARCHIVED))
            self.assertEqual(target.read_bytes(), self.module.notebook_bytes())


if __name__ == "__main__":
    unittest.main()
