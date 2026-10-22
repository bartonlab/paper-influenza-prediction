from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from influenza_figures.registry import (  # noqa: E402
    CANVASES,
    FIGURE_REGISTRY,
    Canvas,
    FigureRegistry,
    FigureSpec,
    RegistryCollisionError,
    SINGLE_COLUMN,
)


class FigureRegistryTests(unittest.TestCase):
    def test_column_width_matches_legacy_definition(self) -> None:
        self.assertAlmostEqual(SINGLE_COLUMN, 8.8 / 2.54)
        self.assertAlmostEqual(CANVASES["single_square"].size_points[0], 249.4488189)

    def test_audited_canvas_assignments(self) -> None:
        auc = FIGURE_REGISTRY.get("paper.auc")
        self.assertAlmostEqual(auc.canvas.width_in, 8.8 / 2.54)
        self.assertAlmostEqual(auc.canvas.height_in, 1.5 * 8.8 / 2.54)

        joint = FIGURE_REGISTRY.get("slides.emd_joint")
        self.assertEqual(joint.canvas.size_inches, (10.5, 5.25))

        frequency = FIGURE_REGISTRY.get("slides.frequency")
        self.assertEqual(frequency.canvas.size_inches, (8.5, 1.6))

    def test_every_output_has_exactly_one_owner(self) -> None:
        outputs = [spec.output_name.casefold() for spec in FIGURE_REGISTRY]
        self.assertEqual(len(outputs), len(set(outputs)))
        for spec in FIGURE_REGISTRY:
            self.assertEqual(FIGURE_REGISTRY.owner_of(spec.output_name), spec.key)

    def test_outputs_are_partitioned_by_purpose(self) -> None:
        expected_directory = {
            "paper": "paper",
            "supplement": "supplement",
            "slides": "slides",
            "grant": "grant",
            "diagnostic": "diagnostic",
            "legacy": "diagnostic",
        }
        for spec in FIGURE_REGISTRY:
            self.assertEqual(
                Path(spec.output_name).parts[0], expected_directory[spec.category]
            )

    def test_emd_outputs_are_owned_per_comparator_and_horizon(self) -> None:
        families = {
            "mlr": (
                "double_over_1_9",
                {6: "supplement", 9: "supplement", 12: "supplement"},
            ),
            "slm": (
                "double_over_1_9",
                {6: "supplement", 9: "supplement", 12: "supplement"},
            ),
            "null": (
                "double_over_2_2",
                {6: "supplement", 9: "paper", 12: "supplement"},
            ),
        }
        for comparator, (canvas, directory_by_horizon) in families.items():
            for horizon in (6, 9, 12):
                key = f"paper.emd_vs_{comparator}.{horizon}m"
                with self.subTest(key=key):
                    spec = FIGURE_REGISTRY.get(key)
                    self.assertEqual(spec.renderer, "render_emd_comparison_figure")
                    self.assertEqual(spec.canvas.name, canvas)
                    self.assertEqual(
                        spec.output_name,
                        (
                            f"{directory_by_horizon[horizon]}/"
                            f"fig-emd-vs-{comparator}-{horizon}m.pdf"
                        ),
                    )

        self.assertNotIn("paper.emd_vs_ss", FIGURE_REGISTRY)
        self.assertNotIn("paper.emd", FIGURE_REGISTRY)
        self.assertNotIn("paper.selection_space_time", FIGURE_REGISTRY)

    def test_cross_validated_best_vs_baseline_diagnostics_are_explicit(self) -> None:
        for horizon in (6, 9, 12):
            with self.subTest(horizon=horizon):
                spec = FIGURE_REGISTRY.get(
                    f"diagnostic.emd_best_vs_baseline.{horizon}m"
                )

                self.assertEqual(spec.renderer, "render_emd_comparison_figure")
                self.assertEqual(spec.profile, "paper_v1")
                self.assertEqual(spec.canvas.name, "double_over_1_9")
                self.assertEqual(spec.category, "diagnostic")
                self.assertEqual(
                    spec.output_name,
                    f"diagnostic/fig-emd-best-vs-baseline-{horizon}m.pdf",
                )
                self.assertEqual(
                    dict(spec.recipe_parameters),
                    {
                        "horizon_months": horizon,
                        "target_method": (
                            "full_model_cross_validated_rare_filtered"
                        ),
                        "comparator_method": "full_model_rare_filtered",
                        "difference_sign": "comparator_minus_target",
                    },
                )

    def test_duplicate_output_is_rejected_case_insensitively(self) -> None:
        canvas = Canvas("test", 1.0, 1.0)
        first = FigureSpec("a", "render_a", "paper_v1", canvas, "Result.pdf", "test")
        second = FigureSpec("b", "render_b", "paper_v1", canvas, "result.PDF", "test")
        with self.assertRaises(RegistryCollisionError):
            FigureRegistry((first, second))

    def test_unsafe_output_path_is_rejected(self) -> None:
        canvas = Canvas("test", 1.0, 1.0)
        with self.assertRaises(ValueError):
            FigureSpec("a", "render", "paper_v1", canvas, "../outside.pdf", "test")


if __name__ == "__main__":
    unittest.main()
