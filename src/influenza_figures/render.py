"""Small composition helpers for registered and legacy renderers."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from types import ModuleType
from typing import Iterator, Literal

from .registry import FIGURE_REGISTRY, FigureRegistry, FigureSpec
from .styles import StyleProfile, get_style, scoped_rc_context


@contextmanager
def figure_style_context(
    figure: str | FigureSpec,
    *,
    registry: FigureRegistry = FIGURE_REGISTRY,
    renderer_layer: Literal["figure", "mplot"] = "figure",
) -> Iterator[tuple[FigureSpec, StyleProfile]]:
    """Resolve a figure and apply its profile plus immutable rc overrides."""

    spec = registry.get(figure) if isinstance(figure, str) else figure
    with scoped_rc_context(
        spec.profile,
        renderer=renderer_layer,
        overrides=spec.rc_overrides,
    ) as profile:
        yield spec, profile


def _snapshot_globals(module: ModuleType, names: tuple[str, ...]) -> dict[str, object]:
    return {name: deepcopy(getattr(module, name)) for name in names}


def _restore_globals(module: ModuleType, snapshot: dict[str, object]) -> None:
    """Restore mappings in place so legacy default arguments keep their identity."""

    for name, value in snapshot.items():
        current = getattr(module, name)
        if isinstance(current, dict) and isinstance(value, dict):
            current.clear()
            current.update(deepcopy(value))
        else:
            setattr(module, name, value)


@contextmanager
def legacy_style_context(
    style: str | StyleProfile,
) -> Iterator[StyleProfile]:
    """Apply one profile to both layers of the archived plotting code.

    The historical modules store typography and line widths twice: once as
    scalar globals and again in mutable default-property dictionaries.  A
    Matplotlib ``rc_context`` alone therefore cannot reproduce a paper or
    slide figure when a different block happened to be active at import time.
    This adapter updates both layers for one render and restores them on exit.
    """

    try:
        from .legacy import figures as legacy_figures
        from .legacy import mplot as legacy_mplot
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The scientific plotting dependencies are required to render a legacy figure."
        ) from exc

    profile = get_style(style)
    figure_names = (
        "FONTFAMILY",
        "SIZESUBLABEL",
        "SIZELABEL",
        "SIZETICK",
        "SMALLSIZEDOT",
        "SIZELINE",
        "AXWIDTH",
        "TICKLENGTH",
        "TICKPAD",
        "DEF_BARPROPS",
        "DEF_HISTPROPS",
        "DEF_ERRORPROPS",
        "DEF_LINEPROPS",
        "DEF_LABELPROPS",
        "DEF_SUBLABELPROPS",
        "DEF_TICKLABELPROPS",
        "DEF_TICKPROPS",
        "DEF_MINORTICKPROPS",
        "DEF_AXPROPS",
    )
    mplot_names = (
        "fontfamily",
        "sizesublabel",
        "sizetext",
        "sizelabel",
        "sizetick",
        "smallsizedot",
        "sizedot",
        "sizeline",
        "axwidth",
        "ticklength",
        "tickpad",
        "def_labelprops",
        "def_sublabelprops",
        "def_ticklabelprops",
        "def_axprops",
        "def_tickprops",
        "def_minortickprops",
        "def_legendprops",
        "def_scatterprops",
        "def_callscatterprops",
        "def_callarrowprops",
        "def_errorprops",
        "def_lineprops",
        "def_fillprops",
        "def_histprops",
        "def_barprops",
        "def_hexbinprops",
        "def_contourprops",
        "def_vplineprops",
        "def_tickprops_boxed",
        "def_minortickprops_boxed",
        "def_axprops_ppt",
        "def_tickprops_ppt",
        "def_minortickprops_ppt",
    )
    figure_snapshot = _snapshot_globals(legacy_figures, figure_names)
    mplot_snapshot = _snapshot_globals(legacy_mplot, mplot_names)

    typography = profile.typography
    figure = profile.legacy_figures
    mplot = profile.legacy_mplot
    family = typography.font_families[0]

    try:
        legacy_figures.FONTFAMILY = family
        legacy_figures.SIZESUBLABEL = typography.panel_label_pt
        legacy_figures.SIZELABEL = typography.axis_label_pt
        legacy_figures.SIZETICK = typography.tick_label_pt
        legacy_figures.SMALLSIZEDOT = figure.smallsizedot
        legacy_figures.SIZELINE = figure.sizeline
        legacy_figures.AXWIDTH = figure.axwidth
        legacy_figures.TICKLENGTH = profile.strokes.tick_length_pt
        legacy_figures.TICKPAD = profile.strokes.tick_pad_pt
        legacy_figures.DEF_BARPROPS.update(lw=figure.sizeline / 2.0)
        legacy_figures.DEF_HISTPROPS.update(lw=figure.sizeline / 2.0)
        legacy_figures.DEF_ERRORPROPS.update(
            mew=figure.axwidth,
            markersize=figure.smallsizedot / 2.0,
            elinewidth=figure.sizeline / 2.0,
        )
        legacy_figures.DEF_LINEPROPS.update(lw=figure.sizeline)
        legacy_figures.DEF_LABELPROPS.update(
            family=family, size=typography.axis_label_pt
        )
        legacy_figures.DEF_SUBLABELPROPS.update(
            family=family, size=typography.panel_label_pt
        )
        legacy_figures.DEF_TICKLABELPROPS.update(
            family=family, size=typography.tick_label_pt
        )
        legacy_figures.DEF_TICKPROPS.update(
            length=profile.strokes.tick_length_pt,
            width=figure.axwidth / 2.0,
            pad=profile.strokes.tick_pad_pt,
        )
        legacy_figures.DEF_MINORTICKPROPS.update(
            length=profile.strokes.tick_length_pt - 1.25,
            width=figure.axwidth / 2.0,
        )
        legacy_figures.DEF_AXPROPS.update(linewidth=figure.axwidth)

        legacy_mplot.fontfamily = family
        legacy_mplot.sizesublabel = typography.panel_label_pt
        legacy_mplot.sizetext = typography.annotation_pt
        legacy_mplot.sizelabel = typography.axis_label_pt
        legacy_mplot.sizetick = typography.tick_label_pt
        legacy_mplot.smallsizedot = mplot.smallsizedot
        legacy_mplot.sizedot = mplot.sizedot
        legacy_mplot.sizeline = mplot.sizeline
        legacy_mplot.axwidth = mplot.axwidth
        legacy_mplot.ticklength = profile.strokes.tick_length_pt
        legacy_mplot.tickpad = profile.strokes.tick_pad_pt
        legacy_mplot.def_labelprops.update(
            family=family, size=typography.axis_label_pt
        )
        legacy_mplot.def_sublabelprops.update(
            family=family, size=typography.panel_label_pt
        )
        legacy_mplot.def_ticklabelprops.update(
            family=family, size=typography.tick_label_pt
        )
        legacy_mplot.def_axprops.update(linewidth=mplot.axwidth)
        legacy_mplot.def_tickprops.update(
            length=profile.strokes.tick_length_pt,
            width=mplot.axwidth / 2.0,
            pad=profile.strokes.tick_pad_pt,
        )
        legacy_mplot.def_minortickprops.update(
            length=profile.strokes.tick_length_pt - 1.25,
            width=mplot.axwidth / 2.0,
        )
        legacy_mplot.def_legendprops["prop"] = {
            "size": typography.axis_label_pt
        }
        legacy_mplot.def_scatterprops.update(s=mplot.sizedot)
        legacy_mplot.def_callscatterprops.update(
            lw=mplot.sizeline, s=mplot.sizedot
        )
        legacy_mplot.def_callarrowprops.update(
            lw=mplot.sizeline, s=mplot.sizedot
        )
        legacy_mplot.def_errorprops.update(
            markersize=mplot.smallsizedot / 2.0,
            elinewidth=mplot.sizeline / 2.0,
        )
        legacy_mplot.def_lineprops.update(lw=mplot.sizeline)
        legacy_mplot.def_fillprops.update(lw=mplot.sizeline)
        legacy_mplot.def_histprops.update(lw=mplot.sizeline / 2.0)
        legacy_mplot.def_barprops.update(lw=mplot.sizeline / 2.0)
        legacy_mplot.def_hexbinprops.update(lw=mplot.axwidth / 4.0)
        legacy_mplot.def_contourprops.update(linewidths=mplot.sizeline / 2.0)
        legacy_mplot.def_vplineprops.update(lw=mplot.axwidth)
        legacy_mplot.def_tickprops_boxed.update(
            length=profile.strokes.tick_length_pt / 1.75,
            width=mplot.axwidth / 1.75,
            pad=profile.strokes.tick_pad_pt,
        )
        legacy_mplot.def_minortickprops_boxed.update(
            length=(profile.strokes.tick_length_pt / 1.75) - 1.25,
            width=mplot.axwidth / 1.75,
        )
        legacy_mplot.def_axprops_ppt.update(linewidth=2.0 * mplot.axwidth)
        legacy_mplot.def_tickprops_ppt.update(
            length=profile.strokes.tick_length_pt,
            width=mplot.axwidth,
            pad=profile.strokes.tick_pad_pt,
        )
        legacy_mplot.def_minortickprops_ppt.update(
            length=profile.strokes.tick_length_pt - 1.25,
            width=mplot.axwidth,
        )

        with scoped_rc_context(profile, renderer="figure"):
            yield profile
    finally:
        _restore_globals(legacy_figures, figure_snapshot)
        _restore_globals(legacy_mplot, mplot_snapshot)
