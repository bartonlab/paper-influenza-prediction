"""Immutable, versioned plotting styles.

The values in this module deliberately preserve two historical layers:

* ``LegacyFigureConstants`` records the constants used directly by
  ``figures.py``.
* ``LegacyMplotConstants`` records the different defaults applied by
  ``mplot.py`` to axes and plotting primitives.

Keeping both layers is necessary for visual parity with archived figures.
New renderers should normally use the semantic typography, stroke, and marker
fields and can retire the legacy adapter after golden-image validation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterator, Literal, Mapping


RendererLayer = Literal["figure", "mplot"]


def _positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value!r}")


def _frozen_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return an immutable copy rather than a view of caller-owned state."""

    return MappingProxyType(dict(values))


@dataclass(frozen=True, slots=True)
class Typography:
    """Text sizes are in points; font families are ordered fallbacks."""

    font_families: tuple[str, ...]
    panel_label_pt: float
    axis_label_pt: float
    tick_label_pt: float
    annotation_pt: float

    def __post_init__(self) -> None:
        if not self.font_families or any(not name for name in self.font_families):
            raise ValueError("font_families must contain at least one non-empty name")
        for name in (
            "panel_label_pt",
            "axis_label_pt",
            "tick_label_pt",
            "annotation_pt",
        ):
            _positive(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class StrokeStyle:
    """Explicit stroke widths and tick geometry, all in points."""

    data_line_pt: float
    detail_line_pt: float
    spine_pt: float
    tick_width_pt: float
    tick_length_pt: float = 3.0
    tick_pad_pt: float = 3.0

    def __post_init__(self) -> None:
        for name in (
            "data_line_pt",
            "detail_line_pt",
            "spine_pt",
            "tick_width_pt",
            "tick_length_pt",
            "tick_pad_pt",
        ):
            _positive(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class MarkerStyle:
    """Marker dimensions with units made explicit.

    Matplotlib scatter ``s`` values are areas in points squared, whereas
    errorbar ``markersize`` values are lengths in points.  The legacy code used
    one variable for both, so the profile stores the two interpretations.
    """

    scatter_area_pt2: float
    error_marker_pt: float

    def __post_init__(self) -> None:
        _positive("scatter_area_pt2", self.scatter_area_pt2)
        _positive("error_marker_pt", self.error_marker_pt)


@dataclass(frozen=True, slots=True)
class LegacyFigureConstants:
    """Exact ``figures.py`` primitive constants for a historical profile."""

    axwidth: float
    sizeline: float
    smallsizedot: float

    def __post_init__(self) -> None:
        _positive("axwidth", self.axwidth)
        _positive("sizeline", self.sizeline)
        _positive("smallsizedot", self.smallsizedot)

    @property
    def tick_width(self) -> float:
        return self.axwidth / 2.0

    @property
    def error_marker_size(self) -> float:
        return self.smallsizedot / 2.0


@dataclass(frozen=True, slots=True)
class LegacyMplotConstants:
    """Exact independent defaults historically supplied by ``mplot.py``."""

    axwidth: float
    sizeline: float
    sizedot: float
    smallsizedot: float

    def __post_init__(self) -> None:
        for name in ("axwidth", "sizeline", "sizedot", "smallsizedot"):
            _positive(name, getattr(self, name))

    @property
    def tick_width(self) -> float:
        return self.axwidth / 2.0

    @property
    def error_marker_size(self) -> float:
        return self.smallsizedot / 2.0


@dataclass(frozen=True, slots=True)
class Palette:
    """Semantic colors shared across media profiles."""

    ink: str
    muted: str
    beneficial: str
    beneficial_light: str
    neutral: str
    neutral_light: str
    deleterious: str
    deleterious_light: str
    model: str
    slide_primary: str
    slide_secondary: str
    protein_colors: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "protein_colors", _frozen_mapping(self.protein_colors))


@dataclass(frozen=True, slots=True)
class OutputStyle:
    """Defaults used by the centralized figure saver."""

    format: str = "pdf"
    dpi: int = 1000
    transparent: bool = True
    bbox_inches: str | None = None
    pdf_fonttype: int = 3

    def __post_init__(self) -> None:
        if not self.format:
            raise ValueError("format must not be empty")
        if self.dpi <= 0:
            raise ValueError("dpi must be positive")
        if self.pdf_fonttype not in (3, 42):
            raise ValueError("pdf_fonttype must be 3 or 42")


@dataclass(frozen=True, slots=True)
class StyleProfile:
    """One immutable and explicitly versioned visual profile."""

    name: str
    typography: Typography
    strokes: StrokeStyle
    markers: MarkerStyle
    palette: Palette
    output: OutputStyle
    legacy_figures: LegacyFigureConstants
    legacy_mplot: LegacyMplotConstants
    extra_rc_params: Mapping[str, Any]
    source_notes: str = ""

    def __post_init__(self) -> None:
        if not self.name or "_v" not in self.name:
            raise ValueError("style names must be explicit and versioned (for example, paper_v1)")
        object.__setattr__(self, "extra_rc_params", _frozen_mapping(self.extra_rc_params))

    def rc_params(self, renderer: RendererLayer = "figure") -> dict[str, Any]:
        """Return a fresh rcParams mapping for a scoped render.

        ``renderer='mplot'`` reproduces the independent legacy axes/primitive
        scale.  The returned dictionary is safe for callers to modify.
        """

        if renderer == "figure":
            line_width = self.legacy_figures.sizeline
            spine_width = self.legacy_figures.axwidth
            tick_width = self.legacy_figures.tick_width
            marker_size = self.legacy_figures.error_marker_size
        elif renderer == "mplot":
            line_width = self.legacy_mplot.sizeline
            spine_width = self.legacy_mplot.axwidth
            tick_width = self.legacy_mplot.tick_width
            marker_size = self.legacy_mplot.error_marker_size
        else:
            raise ValueError(f"unknown renderer layer: {renderer!r}")

        params: dict[str, Any] = {
            "text.usetex": False,
            "mathtext.fontset": "stixsans",
            "mathtext.default": "regular",
            "font.family": list(self.typography.font_families),
            "font.size": self.typography.annotation_pt,
            "axes.labelsize": self.typography.axis_label_pt,
            "axes.linewidth": spine_width,
            "xtick.labelsize": self.typography.tick_label_pt,
            "ytick.labelsize": self.typography.tick_label_pt,
            "xtick.major.size": self.strokes.tick_length_pt,
            "ytick.major.size": self.strokes.tick_length_pt,
            "xtick.major.width": tick_width,
            "ytick.major.width": tick_width,
            "xtick.major.pad": self.strokes.tick_pad_pt,
            "ytick.major.pad": self.strokes.tick_pad_pt,
            "lines.linewidth": line_width,
            "lines.markersize": marker_size,
            "savefig.dpi": self.output.dpi,
            "savefig.transparent": self.output.transparent,
            "savefig.bbox": "standard" if self.output.bbox_inches is None else self.output.bbox_inches,
            "pdf.fonttype": self.output.pdf_fonttype,
        }
        params.update(self.extra_rc_params)
        return params


PALETTE_V1 = Palette(
    ink="#252525",
    muted="#969696",
    beneficial="#EB4025",
    beneficial_light="#F08F78",
    neutral="#969696",
    neutral_light="#E8E8E8",
    deleterious="#3E8DCF",
    deleterious_light="#78B4E7",
    model="#FFB511",
    slide_primary="#F08F78",
    slide_secondary="#78B4E7",
    protein_colors={
        "H3": "#C35999",
        "N2": "#DDA0C3",
        "H1": "#81AB5B",
        "N1": "#ABCE7D",
        "H3N2_HA": "#C35999",
        "H3N2_NA": "#DDA0C3",
        "H1N1_HA": "#81AB5B",
        "H1N1_NA": "#ABCE7D",
    },
)


def _make_profile(
    *,
    name: str,
    fonts: tuple[str, ...],
    panel_label: float,
    text: float,
    tick: float,
    figure_line: float,
    figure_small_dot: float,
    mplot_line: float,
    mplot_axwidth: float,
    mplot_small_dot: float,
    notes: str,
) -> StyleProfile:
    figure = LegacyFigureConstants(
        axwidth=0.4,
        sizeline=figure_line,
        smallsizedot=figure_small_dot,
    )
    mplot = LegacyMplotConstants(
        axwidth=mplot_axwidth,
        sizeline=mplot_line,
        sizedot=8.0,
        smallsizedot=mplot_small_dot,
    )
    return StyleProfile(
        name=name,
        typography=Typography(
            font_families=fonts,
            panel_label_pt=panel_label,
            axis_label_pt=text,
            tick_label_pt=tick,
            annotation_pt=text,
        ),
        strokes=StrokeStyle(
            data_line_pt=figure.sizeline,
            detail_line_pt=figure.axwidth,
            spine_pt=figure.axwidth,
            tick_width_pt=figure.tick_width,
        ),
        markers=MarkerStyle(
            scatter_area_pt2=figure.smallsizedot,
            error_marker_pt=figure.error_marker_size,
        ),
        palette=PALETTE_V1,
        output=OutputStyle(),
        legacy_figures=figure,
        legacy_mplot=mplot,
        extra_rc_params={"figure.autolayout": False},
        source_notes=notes,
    )


PAPER_V1 = _make_profile(
    name="paper_v1",
    fonts=("Helvetica", "Arial", "DejaVu Sans"),
    panel_label=10.0,
    text=8.0,
    tick=8.0,
    figure_line=0.6,
    figure_small_dot=6.0,
    mplot_line=0.8,
    mplot_axwidth=0.4,
    mplot_small_dot=6.0,
    notes="Commented paper constants in figures.py and mplot.py.",
)

SLIDES_V1 = _make_profile(
    name="slides_v1",
    fonts=("Avenir", "Arial", "DejaVu Sans"),
    panel_label=18.0,
    text=18.0,
    tick=18.0,
    figure_line=1.5,
    figure_small_dot=42.0,
    mplot_line=3.0,
    mplot_axwidth=1.5,
    mplot_small_dot=24.0,
    notes="Commented slide constants in figures.py and mplot.py.",
)

POSTER_V1 = _make_profile(
    name="poster_v1",
    fonts=("Avenir", "Arial", "DejaVu Sans"),
    panel_label=24.0,
    text=24.0,
    tick=24.0,
    figure_line=1.5,
    figure_small_dot=42.0,
    mplot_line=3.0,
    mplot_axwidth=1.5,
    mplot_small_dot=24.0,
    notes="Poster constants in figures.py and mplot.py.",
)

GRANT_V1 = _make_profile(
    name="grant_v1",
    fonts=("Arial", "Helvetica", "DejaVu Sans"),
    panel_label=12.0,
    text=8.0,
    tick=8.0,
    figure_line=0.6,
    figure_small_dot=7.8,
    mplot_line=1.0,
    mplot_axwidth=0.4,
    mplot_small_dot=12.0,
    notes="Commented grant constants in figures.py and mplot.py.",
)

# At audit time the active figures.py block was poster, while mplot.py applied
# its own thicker poster axes and primitive defaults.  Retain a separately
# named profile even though its numeric values match POSTER_V1: its name is a
# provenance statement and must not silently follow future poster revisions.
LEGACY_CURRENT_V1 = _make_profile(
    name="legacy_current_v1",
    fonts=("Avenir", "Arial", "DejaVu Sans"),
    panel_label=24.0,
    text=24.0,
    tick=24.0,
    figure_line=1.5,
    figure_small_dot=42.0,
    mplot_line=3.0,
    mplot_axwidth=1.5,
    mplot_small_dot=24.0,
    notes="Exact mixed figures.py/mplot.py constants active during the 2026-08-26 audit.",
)


STYLES: Mapping[str, StyleProfile] = MappingProxyType(
    {
        profile.name: profile
        for profile in (
            PAPER_V1,
            SLIDES_V1,
            POSTER_V1,
            GRANT_V1,
            LEGACY_CURRENT_V1,
        )
    }
)


def get_style(style: str | StyleProfile) -> StyleProfile:
    """Resolve a profile name while giving unknown names a useful error."""

    if isinstance(style, StyleProfile):
        return style
    try:
        return STYLES[style]
    except KeyError as exc:
        choices = ", ".join(sorted(STYLES))
        raise KeyError(f"unknown style {style!r}; choose one of: {choices}") from exc


@contextmanager
def scoped_rc_context(
    style: str | StyleProfile,
    *,
    renderer: RendererLayer = "figure",
    overrides: Mapping[str, Any] | None = None,
) -> Iterator[StyleProfile]:
    """Apply a profile only for the duration of one render.

    Matplotlib is imported lazily so the configuration package and its tests
    remain usable in inventory-only environments without plotting libraries.
    """

    profile = get_style(style)
    params = profile.rc_params(renderer=renderer)
    if overrides:
        params.update(dict(overrides))
    try:
        import matplotlib as mpl
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Matplotlib is required to enter a plotting rc context; install the "
            "project plotting environment first."
        ) from exc

    with mpl.rc_context(rc=params):
        yield profile

