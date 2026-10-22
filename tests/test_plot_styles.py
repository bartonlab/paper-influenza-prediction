from __future__ import annotations

import sys
import types
import unittest
from importlib.util import find_spec
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from influenza_figures.registry import Canvas, FigureSpec  # noqa: E402
from influenza_figures.render import figure_style_context, legacy_style_context  # noqa: E402
from influenza_figures.styles import STYLES, get_style, scoped_rc_context  # noqa: E402


class StyleProfileTests(unittest.TestCase):
    def test_required_profiles_exist(self) -> None:
        self.assertEqual(
            set(STYLES),
            {"paper_v1", "slides_v1", "poster_v1", "grant_v1", "legacy_current_v1"},
        )

    def test_legacy_current_preserves_mixed_constants(self) -> None:
        profile = get_style("legacy_current_v1")
        self.assertEqual(profile.typography.font_families[0], "Avenir")
        self.assertEqual(profile.typography.axis_label_pt, 24.0)
        self.assertEqual(profile.legacy_figures.axwidth, 0.4)
        self.assertEqual(profile.legacy_figures.sizeline, 1.5)
        self.assertEqual(profile.legacy_figures.smallsizedot, 42.0)
        self.assertEqual(profile.legacy_mplot.axwidth, 1.5)
        self.assertEqual(profile.legacy_mplot.sizeline, 3.0)
        self.assertEqual(profile.legacy_mplot.sizedot, 8.0)
        self.assertEqual(profile.legacy_mplot.smallsizedot, 24.0)

    def test_renderer_layers_keep_distinct_rc_scales(self) -> None:
        profile = get_style("legacy_current_v1")
        figure = profile.rc_params("figure")
        mplot = profile.rc_params("mplot")
        self.assertEqual(figure["axes.linewidth"], 0.4)
        self.assertEqual(figure["lines.linewidth"], 1.5)
        self.assertEqual(mplot["axes.linewidth"], 1.5)
        self.assertEqual(mplot["lines.linewidth"], 3.0)
        self.assertEqual(figure["pdf.fonttype"], 3)

    def test_profile_and_palette_are_immutable(self) -> None:
        profile = get_style("paper_v1")
        with self.assertRaises(FrozenInstanceError):
            profile.name = "changed_v1"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            profile.palette.protein_colors["H3"] = "black"  # type: ignore[index]
        with self.assertRaises(TypeError):
            profile.extra_rc_params["font.size"] = 99  # type: ignore[index]

    def test_rc_params_returns_an_independent_copy(self) -> None:
        profile = get_style("paper_v1")
        first = profile.rc_params()
        first["font.size"] = 99
        second = profile.rc_params()
        self.assertEqual(second["font.size"], 8.0)

    def test_scoped_context_and_per_figure_override_do_not_persist(self) -> None:
        captured: list[dict[str, object]] = []
        fake_matplotlib = types.ModuleType("matplotlib")

        @contextmanager
        def fake_rc_context(*, rc: dict[str, object]):
            captured.append(dict(rc))
            yield

        fake_matplotlib.rc_context = fake_rc_context  # type: ignore[attr-defined]
        spec = FigureSpec(
            key="test.override",
            renderer="render_test",
            profile="paper_v1",
            canvas=Canvas("test", 1.0, 1.0),
            output_name="test.pdf",
            category="test",
            rc_overrides={"font.size": 11.0},
            recipe_parameters={"variant": "test"},
        )
        with self.assertRaises(TypeError):
            spec.recipe_parameters["variant"] = "changed"  # type: ignore[index]

        with patch.dict(sys.modules, {"matplotlib": fake_matplotlib}):
            with figure_style_context(spec) as (resolved, profile):
                self.assertIs(resolved, spec)
                self.assertEqual(profile.name, "paper_v1")
            with scoped_rc_context("paper_v1"):
                pass

        self.assertEqual(captured[0]["font.size"], 11.0)
        self.assertEqual(captured[1]["font.size"], 8.0)

    @unittest.skipUnless(find_spec("numpy"), "legacy plotting dependencies unavailable")
    def test_legacy_context_updates_both_modules_and_restores_defaults(self) -> None:
        from influenza_figures.legacy import figures, mplot

        original_figure_line = figures.SIZELINE
        original_mplot_line = mplot.sizeline
        figure_props_identity = id(figures.DEF_LINEPROPS)
        mplot_props_identity = id(mplot.def_lineprops)

        with legacy_style_context("paper_v1") as profile:
            self.assertEqual(profile.name, "paper_v1")
            self.assertEqual(figures.SIZELABEL, 8.0)
            self.assertEqual(figures.SIZELINE, 0.6)
            self.assertEqual(figures.DEF_LINEPROPS["lw"], 0.6)
            self.assertEqual(mplot.sizelabel, 8.0)
            self.assertEqual(mplot.sizeline, 0.8)
            self.assertEqual(mplot.def_lineprops["lw"], 0.8)

        self.assertEqual(figures.SIZELINE, original_figure_line)
        self.assertEqual(mplot.sizeline, original_mplot_line)
        self.assertEqual(id(figures.DEF_LINEPROPS), figure_props_identity)
        self.assertEqual(id(mplot.def_lineprops), mplot_props_identity)


if __name__ == "__main__":
    unittest.main()
