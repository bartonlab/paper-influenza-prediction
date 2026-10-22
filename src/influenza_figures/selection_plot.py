"""Canonical manuscript renderers for selection-focused figures.

The functions in this module replace three opaque legacy notebook calls with
small, explicit recipes that read only the maintained ``data/plotting``
interface:

* main Figure 2: structures, functional-site enrichment, and selection along
  HA/NA sequences;
* Supplementary Figure S5: allele-frequency trajectories and selection
  distributions; and
* Supplementary Figure S6: smoothed selection and solvent accessibility.

Data loading and scientific filtering are separated from rendering so a
notebook can inspect exactly what is plotted.  Heavy scientific/plotting
dependencies are imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import floor, isfinite, log10
from pathlib import Path
from typing import Iterable

from .registry import FigureSpec
from .render import figure_style_context


MIN_MUTATION_COUNT = 20
ALLELE_FREQUENCY_DISPLAY_THRESHOLD = 0.01
INITIAL_WILD_TYPE_FILTER_THRESHOLD = 0.95
FREQUENCY_SAMPLE_INTERVAL_DAYS = 10
FREQUENCY_TIME_ORIGIN = datetime(2009, 1, 1)
FREQUENCY_X_LIMIT_DAYS = 4100
SELECTION_HISTOGRAM_BINS = tuple(
    -0.005 + 0.0005 * index for index in range(42)
)

# The historical arrays were sliced [17:345] and [345:566].  With canonical
# one-based positions this corresponds to 18--345 and 346--566, respectively.
HA1_POSITION_RANGE = (18, 345)
HA2_POSITION_RANGE = (346, 566)

# Keep the manuscript's labeled scales while preserving display padding above
# the outer tick. The archived helper nudged its display envelope by 1.05; for
# Figure 2D its data-driven maximum became about 0.260. S6 also contains an H1
# SASA peak near 93.69, so both fixed envelopes must exceed the legacy ticks.
HA_SELECTION_TICK_MAX = 0.4
HA_SELECTION_LIMIT_MAX = 0.42
HA_DOMAIN_SPAN_Y = 0.39
HA_DOMAIN_LABEL_Y = 0.42
NA_SELECTION_TICK_MAX = 0.2
NA_SELECTION_LIMIT_MAX = 0.26
SASA_TICK_MAX = 80.0
SASA_LIMIT_MAX = 100.0


class SelectionFigureDataError(ValueError):
    """Raised when a canonical selection input violates its documented schema."""


@dataclass(frozen=True, slots=True)
class ProteinDefinition:
    label: str
    subtype: str
    segment: str
    frequency_file: str
    mutation_file: str
    time_shift_days: int


PROTEIN_DEFINITIONS = (
    ProteinDefinition(
        "H3",
        "h3n2",
        "ha",
        "inputs/frequencies/h3n2_ha_allele_frequencies.txt",
        "inputs/selection/h3n2_ha_mutations.csv",
        0,
    ),
    ProteinDefinition(
        "N2",
        "h3n2",
        "na",
        "inputs/frequencies/h3n2_na_allele_frequencies.txt",
        "inputs/selection/h3n2_na_mutations.csv",
        0,
    ),
    ProteinDefinition(
        "H1",
        "h1n1",
        "ha",
        "inputs/frequencies/h1n1_ha_allele_frequencies.txt",
        "inputs/selection/h1n1_ha_mutations.csv",
        120,
    ),
    ProteinDefinition(
        "N1",
        "h1n1",
        "na",
        "inputs/frequencies/h1n1_na_allele_frequencies.txt",
        "inputs/selection/h1n1_na_mutations.csv",
        120,
    ),
)


@dataclass(frozen=True, slots=True)
class FrequencySelectionPanelData:
    definition: ProteinDefinition
    time_days: tuple[int, ...]
    trajectories: tuple[tuple[float, ...], ...]
    total_trajectory_count: int
    selection_coefficients: tuple[float, ...]
    total_mutation_count: int


@dataclass(frozen=True, slots=True)
class FrequencySelectionFigureData:
    minimum_mutation_count: int
    panels: tuple[FrequencySelectionPanelData, ...]


@dataclass(frozen=True, slots=True)
class ProteinCurveData:
    definition: ProteinDefinition
    positions: tuple[int, ...]
    selection_coefficients_smoothed: tuple[float, ...]
    sasa_smoothed: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FrequencySelectionPanelAudit:
    label: str
    subtype: str
    segment: str
    time_shift_days: int
    time_point_count: int
    total_trajectory_count: int
    plotted_trajectory_count: int
    total_mutation_count: int
    plotted_mutation_count: int
    minimum_selection_coefficient: float
    maximum_selection_coefficient: float


@dataclass(frozen=True, slots=True)
class FrequencySelectionFigureAudit:
    figure_key: str
    intended_output_name: str
    profile: str
    canvas_name: str
    canvas_size_inches: tuple[float, float]
    minimum_mutation_count: int
    allele_frequency_display_threshold: float
    initial_wild_type_filter_threshold: float
    panels: tuple[FrequencySelectionPanelAudit, ...]
    saved_path: str | None


@dataclass(frozen=True, slots=True)
class FunctionalSiteAudit:
    label: str
    subtype: str
    subset_id: str
    subset_label: str
    subset_count: int
    other_count: int
    subset_mean_selection: float
    other_mean_selection: float
    t_statistic: float
    p_value: float
    p_value_label: str


@dataclass(frozen=True, slots=True)
class SelectionCurveAudit:
    label: str
    subtype: str
    segment: str
    point_count: int
    minimum_position: int
    maximum_position: int
    minimum_selection: float
    maximum_selection: float


@dataclass(frozen=True, slots=True)
class SelectionStructureFigureAudit:
    figure_key: str
    intended_output_name: str
    profile: str
    canvas_name: str
    canvas_size_inches: tuple[float, float]
    minimum_mutation_count: int
    functional_sites: tuple[FunctionalSiteAudit, ...]
    selection_curves: tuple[SelectionCurveAudit, ...]
    structure_assets: tuple[str, ...]
    saved_path: str | None


@dataclass(frozen=True, slots=True)
class SelectionSasaCorrelationAudit:
    label: str
    subtype: str
    segment: str
    domain: str
    paired_count: int
    pearson_r: float
    pearson_p_value: float
    spearman_r: float
    spearman_p_value: float


@dataclass(frozen=True, slots=True)
class SelectionSasaPanelAudit:
    label: str
    subtype: str
    segment: str
    point_count: int
    positive_sasa_count: int
    maximum_selection: float
    maximum_sasa: float


@dataclass(frozen=True, slots=True)
class SelectionSasaFigureAudit:
    figure_key: str
    intended_output_name: str
    profile: str
    canvas_name: str
    canvas_size_inches: tuple[float, float]
    panels: tuple[SelectionSasaPanelAudit, ...]
    correlations: tuple[SelectionSasaCorrelationAudit, ...]
    saved_path: str | None


@dataclass(frozen=True, slots=True)
class SelectionFigureRenderResult:
    """Matplotlib objects plus immutable scientific audit metadata."""

    figure: object
    axes: tuple[object, ...]
    audit: object


def _load_data_dependencies():
    try:
        import numpy as np
        import pandas as pd
    except ModuleNotFoundError as error:  # pragma: no cover - environment specific
        raise RuntimeError(
            "NumPy and pandas are required to collect canonical selection data."
        ) from error
    return np, pd


def _load_plotting_dependencies():
    try:
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats
    except ModuleNotFoundError as error:  # pragma: no cover - environment specific
        raise RuntimeError(
            "Matplotlib, NumPy, and SciPy are required to render selection figures."
        ) from error
    return plt, mpimg, np, stats


def _data_path(plotting_data_dir: Path | str, relative_path: str) -> Path:
    root = Path(plotting_data_dir).expanduser().resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:  # defensive: all module paths are fixed constants
        raise SelectionFigureDataError(
            f"canonical path escapes plotting-data root: {relative_path!r}"
        ) from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _require_columns(table, required: Iterable[str], *, source: Path) -> None:
    missing = sorted(set(required).difference(table.columns))
    if missing:
        raise SelectionFigureDataError(
            f"{source} is missing required columns: {', '.join(missing)}"
        )


def _mutation_table(
    plotting_data_dir: Path | str,
    definition: ProteinDefinition,
    *,
    minimum_mutation_count: int,
):
    np, pd = _load_data_dependencies()
    source = _data_path(plotting_data_dir, definition.mutation_file)
    table = pd.read_csv(source)
    _require_columns(
        table,
        (
            "subtype",
            "segment",
            "selection_coefficient",
            "mutation_count",
        ),
        source=source,
    )
    identities = set(zip(table["subtype"], table["segment"], strict=True))
    if identities != {(definition.subtype, definition.segment)}:
        raise SelectionFigureDataError(
            f"{source} contains unexpected subtype/segment identities: "
            f"{sorted(identities)!r}"
        )
    coefficients = table["selection_coefficient"].to_numpy(dtype=float)
    counts = table["mutation_count"].to_numpy(dtype=float)
    if not np.isfinite(coefficients).all() or not np.isfinite(counts).all():
        raise SelectionFigureDataError(
            f"{source} contains non-finite selection coefficients or mutation counts"
        )
    if (counts < 0).any():
        raise SelectionFigureDataError(f"{source} contains negative mutation counts")
    filtered = table.loc[table["mutation_count"] >= minimum_mutation_count].copy()
    if filtered.empty:
        raise SelectionFigureDataError(
            f"{source} has no rows with mutation_count >= {minimum_mutation_count}"
        )
    return source, table, filtered


def collect_frequency_selection_data(
    plotting_data_dir: Path | str,
    *,
    minimum_mutation_count: int = MIN_MUTATION_COUNT,
) -> FrequencySelectionFigureData:
    """Collect the exact S5 trajectories and mutation-level distributions.

    A trajectory is shown only when it rises above one percent at least once
    and does not begin at or above 95 percent.  H1N1 series receive the
    historical 120-day calendar offset; the values themselves are unchanged.
    """

    if minimum_mutation_count < 0:
        raise ValueError("minimum_mutation_count must be non-negative")
    np, _ = _load_data_dependencies()
    panels: list[FrequencySelectionPanelData] = []
    for definition in PROTEIN_DEFINITIONS:
        frequency_source = _data_path(
            plotting_data_dir, definition.frequency_file
        )
        matrix = np.loadtxt(frequency_source, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise SelectionFigureDataError(
                f"{frequency_source} must contain a non-empty two-dimensional matrix"
            )
        if not np.isfinite(matrix).all():
            raise SelectionFigureDataError(
                f"{frequency_source} contains non-finite allele frequencies"
            )
        retained = (
            (np.max(matrix, axis=1) > ALLELE_FREQUENCY_DISPLAY_THRESHOLD)
            & (matrix[:, 0] < INITIAL_WILD_TYPE_FILTER_THRESHOLD)
        )
        plotted = matrix[retained]
        if plotted.shape[0] == 0:
            raise SelectionFigureDataError(
                f"{frequency_source} has no trajectories after display filtering"
            )

        _, mutation_table, filtered_mutations = _mutation_table(
            plotting_data_dir,
            definition,
            minimum_mutation_count=minimum_mutation_count,
        )
        time_days = tuple(
            int(index * FREQUENCY_SAMPLE_INTERVAL_DAYS + definition.time_shift_days)
            for index in range(matrix.shape[1])
        )
        panels.append(
            FrequencySelectionPanelData(
                definition=definition,
                time_days=time_days,
                trajectories=tuple(
                    tuple(float(value) for value in row) for row in plotted
                ),
                total_trajectory_count=int(matrix.shape[0]),
                selection_coefficients=tuple(
                    float(value)
                    for value in filtered_mutations["selection_coefficient"]
                ),
                total_mutation_count=int(len(mutation_table)),
            )
        )
    return FrequencySelectionFigureData(
        minimum_mutation_count=minimum_mutation_count,
        panels=tuple(panels),
    )


def collect_selection_curve_data(
    plotting_data_dir: Path | str,
) -> tuple[ProteinCurveData, ...]:
    """Collect the four canonical smoothed selection/SASA series."""

    np, pd = _load_data_dependencies()
    source = _data_path(
        plotting_data_dir, "derived/structure/sasa_selection.csv"
    )
    table = pd.read_csv(source)
    _require_columns(
        table,
        (
            "subtype",
            "segment",
            "position",
            "sasa_mean_smoothed",
            "selection_coefficient_smoothed",
        ),
        source=source,
    )
    curves: list[ProteinCurveData] = []
    for definition in PROTEIN_DEFINITIONS:
        subset = table.loc[
            (table["subtype"] == definition.subtype)
            & (table["segment"] == definition.segment)
        ].sort_values("position")
        if subset.empty:
            raise SelectionFigureDataError(
                f"{source} has no rows for {definition.subtype}/{definition.segment}"
            )
        positions = subset["position"].to_numpy(dtype=int)
        selection = subset["selection_coefficient_smoothed"].to_numpy(
            dtype=float
        )
        sasa = subset["sasa_mean_smoothed"].to_numpy(dtype=float)
        if len(set(positions.tolist())) != len(positions):
            raise SelectionFigureDataError(
                f"{source} has duplicate positions for "
                f"{definition.subtype}/{definition.segment}"
            )
        if not np.all(np.diff(positions) > 0):
            raise SelectionFigureDataError(
                f"{source} positions are not strictly increasing for "
                f"{definition.subtype}/{definition.segment}"
            )
        if not np.isfinite(selection).all() or not np.isfinite(sasa).all():
            raise SelectionFigureDataError(
                f"{source} contains non-finite values for "
                f"{definition.subtype}/{definition.segment}"
            )
        curves.append(
            ProteinCurveData(
                definition=definition,
                positions=tuple(int(value) for value in positions),
                selection_coefficients_smoothed=tuple(
                    float(value) for value in selection
                ),
                sasa_smoothed=tuple(float(value) for value in sasa),
            )
        )
    return tuple(curves)


def _style_open_axis(axis, profile, *, bottom: bool = True, right: bool = False) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(right)
    axis.spines["bottom"].set_visible(bottom)
    axis.spines["left"].set_visible(not right)
    for name in ("left", "right", "bottom"):
        axis.spines[name].set_color(profile.palette.ink)
        axis.spines[name].set_linewidth(profile.legacy_figures.axwidth)
    axis.tick_params(
        axis="both",
        which="major",
        direction="out",
        colors=profile.palette.ink,
        bottom=bottom,
        left=not right,
        top=False,
        right=right,
        length=profile.strokes.tick_length_pt,
        width=profile.legacy_figures.tick_width,
        pad=profile.strokes.tick_pad_pt,
    )
    axis.tick_params(
        axis="both",
        which="minor",
        direction="out",
        colors=profile.palette.ink,
        bottom=bottom,
        left=not right,
        top=False,
        right=right,
        length=profile.strokes.tick_length_pt - 1.25,
        width=profile.legacy_figures.tick_width,
    )
    font_family = profile.typography.font_families[0]
    for tick_label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        tick_label.set_fontfamily(font_family)
        tick_label.set_fontsize(profile.typography.tick_label_pt)
        tick_label.set_color(profile.palette.ink)


def _set_axis_label(axis, which: str, text: str, profile) -> None:
    properties = {
        "family": profile.typography.font_families[0],
        "fontsize": profile.typography.axis_label_pt,
        "color": profile.palette.ink,
    }
    if which == "x":
        axis.set_xlabel(text, **properties)
    elif which == "y":
        axis.set_ylabel(text, **properties)
    else:  # pragma: no cover - private helper contract
        raise ValueError(which)


def _save_figure(figure, output_path: Path | str | None, profile) -> str | None:
    if output_path is None:
        return None
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    options: dict[str, object] = {
        "dpi": profile.output.dpi,
        "transparent": profile.output.transparent,
        "format": destination.suffix.lstrip(".") or profile.output.format,
    }
    if profile.output.bbox_inches is not None:
        options["bbox_inches"] = profile.output.bbox_inches
    figure.savefig(destination, **options)
    return str(destination)


def render_frequency_selection_figure(
    plotting_data_dir: Path | str,
    *,
    figure_spec: FigureSpec,
    minimum_mutation_count: int = MIN_MUTATION_COUNT,
    output_path: Path | str | None = None,
) -> SelectionFigureRenderResult:
    """Render Supplementary Figure S5 from canonical inputs."""

    data = collect_frequency_selection_data(
        plotting_data_dir,
        minimum_mutation_count=minimum_mutation_count,
    )
    plt, _, np, _ = _load_plotting_dependencies()
    panel_audits: list[FrequencySelectionPanelAudit] = []

    with figure_style_context(figure_spec) as (resolved_spec, profile):
        figure = plt.figure(figsize=resolved_spec.canvas.size_inches)
        figure.set_size_inches(resolved_spec.canvas.size_inches, forward=False)
        trajectory_grid = figure.add_gridspec(
            4, 1, left=0.07, right=0.65, top=0.95, bottom=0.10, hspace=0.40
        )
        selection_grid = figure.add_gridspec(
            4, 1, left=0.73, right=0.95, top=0.95, bottom=0.10, hspace=0.40
        )
        trajectory_axes = tuple(
            figure.add_subplot(trajectory_grid[index, 0]) for index in range(4)
        )
        selection_axes = tuple(
            figure.add_subplot(selection_grid[index, 0]) for index in range(4)
        )

        date_tick_days = tuple(index * 2 * 365 for index in range(6))
        date_tick_labels = tuple(
            (FREQUENCY_TIME_ORIGIN + timedelta(days=value)).strftime("%b %Y")
            for value in date_tick_days
        )
        selection_ticks = (-0.005, 0.0, 0.005, 0.01, 0.015)
        selection_tick_labels = (
            r"$-0.5$",
            r"$0$",
            r"$0.5$",
            r"$1$",
            r"$1.5$",
        )
        font_family = profile.typography.font_families[0]

        for index, (panel, trajectory_axis, selection_axis) in enumerate(
            zip(data.panels, trajectory_axes, selection_axes, strict=True)
        ):
            color = profile.palette.protein_colors[panel.definition.label]
            for trajectory in panel.trajectories:
                trajectory_axis.plot(
                    panel.time_days,
                    trajectory,
                    color=color,
                    linewidth=profile.legacy_figures.sizeline * 1.5,
                    alpha=0.7,
                )
            trajectory_axis.set_xlim(0, FREQUENCY_X_LIMIT_DAYS)
            trajectory_axis.set_xticks(
                (*date_tick_days, FREQUENCY_X_LIMIT_DAYS)
            )
            trajectory_axis.set_xticklabels(
                (*date_tick_labels, "") if index == 3 else ("",) * 7
            )
            trajectory_axis.set_ylim(0.0, 1.05)
            trajectory_axis.set_yticks((0.0, 1.0))
            trajectory_axis.set_yticks((0.25, 0.5, 0.75), minor=True)
            _style_open_axis(trajectory_axis, profile)
            trajectory_axis.spines["left"].set_bounds(0.0, 1.0)
            trajectory_axis.spines["bottom"].set_bounds(
                0, FREQUENCY_X_LIMIT_DAYS
            )

            coefficients = np.asarray(panel.selection_coefficients, dtype=float)
            weights = np.ones(coefficients.shape, dtype=float) / len(coefficients)
            selection_axis.hist(
                coefficients,
                bins=np.asarray(SELECTION_HISTOGRAM_BINS),
                weights=weights,
                color=color,
                edgecolor="none",
                linewidth=profile.legacy_figures.sizeline / 2.0,
                alpha=1.0,
                rwidth=0.8,
            )
            selection_axis.set_yscale("log")
            selection_axis.set_xlim(-0.005, 0.015)
            selection_axis.set_xticks(selection_ticks)
            selection_axis.set_xticklabels(
                selection_tick_labels if index == 3 else ("",) * 5
            )
            selection_axis.set_ylim(0.001, 1.0)
            selection_axis.set_yticks((0.001, 0.01, 0.1, 1.0))
            _style_open_axis(selection_axis, profile)
            selection_axis.spines["left"].set_bounds(0.001, 1.0)
            selection_axis.spines["bottom"].set_bounds(-0.005, 0.015)

            annotation_y = 0.95 - (0.95 - 0.10 + 0.08) * (index / 4.0)
            figure.text(
                0.955,
                annotation_y,
                panel.definition.label,
                color=color,
                family=font_family,
                fontsize=profile.typography.axis_label_pt,
                fontweight="bold",
                ha="right",
                va="center",
                clip_on=False,
            )
            panel_audits.append(
                FrequencySelectionPanelAudit(
                    label=panel.definition.label,
                    subtype=panel.definition.subtype,
                    segment=panel.definition.segment,
                    time_shift_days=panel.definition.time_shift_days,
                    time_point_count=len(panel.time_days),
                    total_trajectory_count=panel.total_trajectory_count,
                    plotted_trajectory_count=len(panel.trajectories),
                    total_mutation_count=panel.total_mutation_count,
                    plotted_mutation_count=len(panel.selection_coefficients),
                    minimum_selection_coefficient=min(
                        panel.selection_coefficients
                    ),
                    maximum_selection_coefficient=max(
                        panel.selection_coefficients
                    ),
                )
            )

        _set_axis_label(
            selection_axes[-1],
            "x",
            "Inferred selection coefficient (%)",
            profile,
        )
        common_text = {
            "family": font_family,
            "fontsize": profile.typography.axis_label_pt,
            "color": profile.palette.ink,
            "ha": "center",
            "va": "center",
        }
        figure.text(0.03, 0.525, "Allele frequency", rotation=90, **common_text)
        figure.text(0.67, 0.525, "Frequency", rotation=90, **common_text)
        panel_text = {
            "family": font_family,
            "fontsize": profile.typography.panel_label_pt,
            "fontweight": "bold",
            "color": "black",
            "ha": "center",
            "va": "center",
            "clip_on": False,
        }
        figure.text(0.04, 0.97, "A", **panel_text)
        figure.text(0.68, 0.97, "B", **panel_text)
        saved_path = _save_figure(figure, output_path, profile)

    audit = FrequencySelectionFigureAudit(
        figure_key=figure_spec.key,
        intended_output_name=figure_spec.output_name,
        profile=figure_spec.profile,
        canvas_name=figure_spec.canvas.name,
        canvas_size_inches=figure_spec.canvas.size_inches,
        minimum_mutation_count=data.minimum_mutation_count,
        allele_frequency_display_threshold=ALLELE_FREQUENCY_DISPLAY_THRESHOLD,
        initial_wild_type_filter_threshold=INITIAL_WILD_TYPE_FILTER_THRESHOLD,
        panels=tuple(panel_audits),
        saved_path=saved_path,
    )
    return SelectionFigureRenderResult(
        figure=figure,
        axes=tuple(
            axis
            for pair in zip(trajectory_axes, selection_axes, strict=True)
            for axis in pair
        ),
        audit=audit,
    )


def _p_value_order_label(p_value: float) -> str:
    if not isfinite(p_value) or p_value < 0.0:
        raise SelectionFigureDataError(f"invalid functional-site p-value: {p_value}")
    exponent = -300 if p_value == 0.0 else floor(log10(p_value)) + 1
    return rf"$p < 10^{{{exponent:d}}}$"


def render_selection_structure_space_figure(
    plotting_data_dir: Path | str,
    *,
    figure_spec: FigureSpec,
    minimum_mutation_count: int = MIN_MUTATION_COUNT,
    output_path: Path | str | None = None,
) -> SelectionFigureRenderResult:
    """Render main Figure 2 from canonical mutation, structure, and SASA data."""

    if minimum_mutation_count < 0:
        raise ValueError("minimum_mutation_count must be non-negative")
    curves = collect_selection_curve_data(plotting_data_dir)
    curve_by_label = {curve.definition.label: curve for curve in curves}
    h3_definition = PROTEIN_DEFINITIONS[0]
    h1_definition = PROTEIN_DEFINITIONS[2]
    h3_source, _, h3_mutations = _mutation_table(
        plotting_data_dir,
        h3_definition,
        minimum_mutation_count=minimum_mutation_count,
    )
    h1_source, _, h1_mutations = _mutation_table(
        plotting_data_dir,
        h1_definition,
        minimum_mutation_count=minimum_mutation_count,
    )
    _require_columns(h3_mutations, ("rbs", "koel"), source=h3_source)
    _require_columns(
        h1_mutations,
        ("rbs", "canton_globular_head"),
        source=h1_source,
    )

    plt, mpimg, np, stats = _load_plotting_dependencies()
    functional_audits: list[FunctionalSiteAudit] = []
    curve_audits: list[SelectionCurveAudit] = []

    structure_paths = (
        _data_path(
            plotting_data_dir, "inputs/structure/h3n2_ha_surface.png"
        ),
        _data_path(
            plotting_data_dir, "inputs/structure/h1n1_ha_surface.png"
        ),
        _data_path(
            plotting_data_dir, "inputs/structure/selection_colorbar.png"
        ),
    )

    with figure_style_context(figure_spec) as (resolved_spec, profile):
        figure = plt.figure(figsize=resolved_spec.canvas.size_inches)
        figure.set_size_inches(resolved_spec.canvas.size_inches, forward=False)
        top = 0.94
        bottom = 0.075
        left = 0.07
        right = 0.98
        space_y = 0.30
        gap_y = 0.12
        gap_x = 0.08
        extra_gap_x = 0.02
        canvas_width, canvas_height = resolved_spec.canvas.size_inches
        ha_ratio = 1026 / 1803
        ha_extra_y = 0.05
        ha_height = (top + ha_extra_y - bottom) - space_y - gap_y
        ha_width = ha_height * ha_ratio * (canvas_height / canvas_width)
        histogram_width = (
            right - left - (2 * ha_width + 5 * gap_x / 2)
        ) / 2
        sequence_width = (right - left - 1.2 * gap_x - extra_gap_x) / 2

        boxes = {
            "colorbar": (
                left - gap_x / 2 + 0.02,
                top - ha_height + 0.05,
                0.008,
                ha_height - 0.10,
            ),
            "h3_structure": (left, top - ha_height, ha_width, ha_height),
            "h3_hist": (
                left + ha_width + gap_x,
                top - ha_height + ha_extra_y,
                histogram_width,
                ha_height - ha_extra_y,
            ),
            "h1_structure": (
                left + ha_width + 1.5 * gap_x + histogram_width,
                top - ha_height,
                ha_width,
                ha_height,
            ),
            "h1_hist": (
                left + 2 * ha_width + 2.5 * gap_x + histogram_width,
                top - ha_height + ha_extra_y,
                right
                - (left + 2 * ha_width + 2.5 * gap_x + histogram_width),
                ha_height - ha_extra_y,
            ),
            "ha_sequence": (
                left + extra_gap_x,
                bottom,
                sequence_width,
                top - ha_height - gap_y - bottom,
            ),
            "na_sequence": (
                left + sequence_width + 1.2 * gap_x + extra_gap_x,
                bottom,
                right
                - (left + sequence_width + 1.2 * gap_x + extra_gap_x),
                top - ha_height - gap_y - bottom,
            ),
        }

        h3_structure_axis = figure.add_axes(boxes["h3_structure"])
        h1_structure_axis = figure.add_axes(boxes["h1_structure"])
        colorbar_axis = figure.add_axes(boxes["colorbar"])
        h3_hist_grid = figure.add_gridspec(
            2,
            1,
            left=boxes["h3_hist"][0],
            right=boxes["h3_hist"][0] + boxes["h3_hist"][2],
            bottom=boxes["h3_hist"][1],
            top=boxes["h3_hist"][1] + boxes["h3_hist"][3],
            hspace=0.30,
        )
        h1_hist_grid = figure.add_gridspec(
            2,
            1,
            left=boxes["h1_hist"][0],
            right=boxes["h1_hist"][0] + boxes["h1_hist"][2],
            bottom=boxes["h1_hist"][1],
            top=boxes["h1_hist"][1] + boxes["h1_hist"][3],
            hspace=0.30,
        )
        h3_hist_axes = tuple(
            figure.add_subplot(h3_hist_grid[index, 0]) for index in range(2)
        )
        h1_hist_axes = tuple(
            figure.add_subplot(h1_hist_grid[index, 0]) for index in range(2)
        )
        ha_sequence_axis = figure.add_axes(boxes["ha_sequence"])
        na_sequence_axis = figure.add_axes(boxes["na_sequence"])

        h3_structure_axis.imshow(mpimg.imread(structure_paths[0]))
        h1_structure_axis.imshow(mpimg.imread(structure_paths[1]))
        h3_structure_axis.axis("off")
        h1_structure_axis.axis("off")
        colorbar_axis.imshow(
            np.rot90(mpimg.imread(structure_paths[2])), aspect="auto"
        )
        colorbar_axis.axis("off")

        font_family = profile.typography.font_families[0]
        label_properties = {
            "family": font_family,
            "fontsize": profile.typography.axis_label_pt,
            "color": profile.palette.ink,
        }
        panel_properties = {
            "family": font_family,
            "fontsize": profile.typography.panel_label_pt,
            "fontweight": "bold",
            "color": "black",
            "ha": "center",
            "va": "center",
            "clip_on": False,
        }
        figure.text(left - 0.03, top + 0.02, "A", **panel_properties)
        figure.text(
            boxes["h1_structure"][0], top + 0.02, "B", **panel_properties
        )
        figure.text(
            left - 0.03,
            boxes["ha_sequence"][1] + boxes["ha_sequence"][3] + 0.02,
            "C",
            **panel_properties,
        )
        figure.text(
            boxes["na_sequence"][0] - 0.03,
            boxes["na_sequence"][1] + boxes["na_sequence"][3] + 0.02,
            "D",
            **panel_properties,
        )
        figure.text(
            boxes["colorbar"][0] - 0.02,
            boxes["colorbar"][1] + boxes["colorbar"][3] / 2,
            "Selection coefficient",
            rotation=90,
            ha="center",
            va="center",
            **label_properties,
        )
        figure.text(
            boxes["colorbar"][0] + boxes["colorbar"][2] / 2,
            boxes["colorbar"][1] + boxes["colorbar"][3] + 0.025,
            r"$+$",
            ha="center",
            va="center",
            **label_properties,
        )
        figure.text(
            boxes["colorbar"][0] + boxes["colorbar"][2] / 2,
            boxes["colorbar"][1] - 0.025,
            r"$-$",
            ha="center",
            va="center",
            **label_properties,
        )

        histogram_edges = np.asarray(SELECTION_HISTOGRAM_BINS)
        histogram_ticks = (-0.005, 0.0, 0.005, 0.01, 0.015)
        histogram_tick_labels = (
            r"$-0.5$",
            r"$0$",
            r"$0.5$",
            r"$1$",
            r"$1.5$",
        )

        def draw_site_histograms(
            table,
            axes,
            *,
            definition: ProteinDefinition,
            subset_specs: tuple[tuple[str, str], ...],
        ) -> None:
            color = profile.palette.protein_colors[definition.label]
            for index, (axis, (subset_id, subset_label)) in enumerate(
                zip(axes, subset_specs, strict=True)
            ):
                subset_values = table.loc[
                    table[subset_id], "selection_coefficient"
                ].to_numpy(dtype=float)
                other_values = table.loc[
                    ~table[subset_id], "selection_coefficient"
                ].to_numpy(dtype=float)
                if len(subset_values) == 0 or len(other_values) == 0:
                    raise SelectionFigureDataError(
                        f"{definition.label}/{subset_id} has an empty comparison group"
                    )
                axis.hist(
                    other_values,
                    bins=histogram_edges,
                    weights=np.ones_like(other_values) / len(other_values),
                    color=profile.palette.muted,
                    edgecolor="none",
                    alpha=0.7,
                    rwidth=0.8,
                )
                axis.hist(
                    subset_values,
                    bins=histogram_edges,
                    weights=np.ones_like(subset_values) / len(subset_values),
                    color=color,
                    edgecolor="none",
                    alpha=0.7,
                    rwidth=0.8,
                )
                axis.set_yscale("log")
                axis.set_xlim(-0.005, 0.015)
                axis.set_xticks(histogram_ticks)
                axis.set_xticklabels(
                    histogram_tick_labels if index == 1 else ("",) * 5
                )
                axis.set_ylim(0.001, 1.0)
                axis.set_yticks((0.001, 0.01, 0.1, 1.0))
                _style_open_axis(axis, profile)
                axis.spines["left"].set_bounds(0.001, 1.0)
                axis.spines["bottom"].set_bounds(-0.005, 0.015)
                if index == 1:
                    _set_axis_label(
                        axis,
                        "x",
                        "Inferred selection coefficient (%)",
                        profile,
                    )

                test = stats.ttest_ind(subset_values, other_values)
                p_value = float(test.pvalue)
                p_label = _p_value_order_label(p_value)
                axis_box = boxes[
                    "h3_hist" if definition.label == "H3" else "h1_hist"
                ]
                if index == 0:
                    annotation_y = axis_box[1] + axis_box[3] - 0.01
                else:
                    annotation_y = axis_box[1] + axis_box[3] / 2 - 0.02
                annotation_x = axis_box[0] + axis_box[2] + 0.005
                line_spacing = 0.035
                figure.text(
                    annotation_x,
                    annotation_y,
                    subset_label,
                    color=color,
                    family=font_family,
                    fontsize=profile.typography.axis_label_pt,
                    fontweight="bold",
                    ha="right",
                    va="center",
                    clip_on=False,
                )
                figure.text(
                    annotation_x,
                    annotation_y - line_spacing,
                    "other",
                    color=profile.palette.muted,
                    family=font_family,
                    fontsize=profile.typography.axis_label_pt,
                    fontweight="bold",
                    ha="right",
                    va="center",
                    clip_on=False,
                )
                figure.text(
                    annotation_x,
                    annotation_y - 2 * line_spacing,
                    p_label,
                    color=profile.palette.ink,
                    family=font_family,
                    fontsize=profile.typography.axis_label_pt,
                    ha="right",
                    va="center",
                    clip_on=False,
                )
                functional_audits.append(
                    FunctionalSiteAudit(
                        label=definition.label,
                        subtype=definition.subtype,
                        subset_id=subset_id,
                        subset_label=subset_label,
                        subset_count=len(subset_values),
                        other_count=len(other_values),
                        subset_mean_selection=float(np.mean(subset_values)),
                        other_mean_selection=float(np.mean(other_values)),
                        t_statistic=float(test.statistic),
                        p_value=p_value,
                        p_value_label=p_label,
                    )
                )

        draw_site_histograms(
            h3_mutations,
            h3_hist_axes,
            definition=h3_definition,
            subset_specs=(("rbs", "RBS"), ("koel", "Ag (Koel)")),
        )
        draw_site_histograms(
            h1_mutations,
            h1_hist_axes,
            definition=h1_definition,
            subset_specs=(
                ("rbs", "RBS"),
                ("canton_globular_head", "Ag (Caton)"),
            ),
        )
        figure.text(
            boxes["h3_hist"][0] - 0.06,
            boxes["h3_hist"][1] + boxes["h3_hist"][3] / 2,
            "Frequency",
            rotation=90,
            ha="center",
            va="center",
            **label_properties,
        )
        figure.text(
            boxes["h1_hist"][0] - 0.06,
            boxes["h1_hist"][1] + boxes["h1_hist"][3] / 2,
            "Frequency",
            rotation=90,
            ha="center",
            va="center",
            **label_properties,
        )

        callout_x = (np.asarray((0.35, 0.50, 0.65)) - 2.5) * 0.01
        callout_y = 10 ** np.asarray((0.0, 0.5, 0.5))
        h3_hist_axes[0].plot(
            callout_x,
            callout_y,
            color=profile.palette.ink,
            linewidth=profile.legacy_figures.axwidth,
            clip_on=False,
        )
        h3_hist_axes[0].text(
            callout_x[-1] + 0.0005,
            callout_y[-1],
            "RBS",
            family=font_family,
            fontsize=profile.typography.axis_label_pt,
            color=profile.palette.ink,
            ha="left",
            va="center",
            clip_on=False,
        )

        def draw_sequence_axis(
            axis,
            first_label: str,
            second_label: str,
            *,
            tick_maximum_y: float,
            limit_maximum_y: float,
            ylabel: str,
        ) -> None:
            first = curve_by_label[first_label]
            second = curve_by_label[second_label]
            first_color = profile.palette.protein_colors[first_label]
            second_color = profile.palette.protein_colors[second_label]
            axis.plot(
                first.positions,
                first.selection_coefficients_smoothed,
                color=first_color,
                linewidth=profile.legacy_figures.sizeline * 2.0,
                alpha=0.7,
            )
            axis.plot(
                second.positions,
                second.selection_coefficients_smoothed,
                color=second_color,
                linewidth=profile.legacy_figures.sizeline * 2.0,
                alpha=0.7,
            )
            axis.set_xlim(0, max(first.positions) + 1)
            axis.set_xticks(())
            axis.set_ylim(0, limit_maximum_y)
            axis.set_yticks((0, tick_maximum_y))
            if tick_maximum_y == HA_SELECTION_TICK_MAX:
                axis.set_yticks((0.1, 0.2, 0.3), minor=True)
            else:
                axis.set_yticks((0.05, 0.1, 0.15), minor=True)
            _style_open_axis(axis, profile, bottom=False)
            axis.spines["left"].set_bounds(0, tick_maximum_y)
            _set_axis_label(axis, "x", "Position along sequence", profile)
            _set_axis_label(axis, "y", ylabel, profile)
            axis.text(
                0.025,
                0.86 if tick_maximum_y == HA_SELECTION_TICK_MAX else 0.95,
                first_label,
                transform=axis.transAxes,
                color=first_color,
                family=font_family,
                fontsize=profile.typography.axis_label_pt,
                fontweight="bold",
                ha="left",
                va="center",
            )
            axis.text(
                0.025,
                0.73 if tick_maximum_y == HA_SELECTION_TICK_MAX else 0.82,
                second_label,
                transform=axis.transAxes,
                color=second_color,
                family=font_family,
                fontsize=profile.typography.axis_label_pt,
                fontweight="bold",
                ha="left",
                va="center",
            )
            for curve in (first, second):
                curve_audits.append(
                    SelectionCurveAudit(
                        label=curve.definition.label,
                        subtype=curve.definition.subtype,
                        segment=curve.definition.segment,
                        point_count=len(curve.positions),
                        minimum_position=min(curve.positions),
                        maximum_position=max(curve.positions),
                        minimum_selection=min(
                            curve.selection_coefficients_smoothed
                        ),
                        maximum_selection=max(
                            curve.selection_coefficients_smoothed
                        ),
                    )
                )

        draw_sequence_axis(
            ha_sequence_axis,
            "H3",
            "H1",
            tick_maximum_y=HA_SELECTION_TICK_MAX,
            limit_maximum_y=HA_SELECTION_LIMIT_MAX,
            ylabel="Relative strength\nof selection (HA)",
        )
        draw_sequence_axis(
            na_sequence_axis,
            "N2",
            "N1",
            tick_maximum_y=NA_SELECTION_TICK_MAX,
            limit_maximum_y=NA_SELECTION_LIMIT_MAX,
            ylabel="Relative strength\nof selection (NA)",
        )
        ha_sequence_axis.plot(
            (21, 341),
            (HA_DOMAIN_SPAN_Y, HA_DOMAIN_SPAN_Y),
            color=profile.palette.ink,
            linewidth=profile.legacy_figures.axwidth,
        )
        ha_sequence_axis.plot(
            (349, 562),
            (HA_DOMAIN_SPAN_Y, HA_DOMAIN_SPAN_Y),
            color=profile.palette.ink,
            linewidth=profile.legacy_figures.axwidth,
        )
        ha_sequence_axis.text(
            (17 + 345) / 2,
            HA_DOMAIN_LABEL_Y,
            "HA1",
            family=font_family,
            fontsize=profile.typography.axis_label_pt,
            color=profile.palette.ink,
            ha="center",
            va="center",
        )
        ha_sequence_axis.text(
            (345 + 566) / 2,
            HA_DOMAIN_LABEL_Y,
            "HA2",
            family=font_family,
            fontsize=profile.typography.axis_label_pt,
            color=profile.palette.ink,
            ha="center",
            va="center",
        )
        saved_path = _save_figure(figure, output_path, profile)

    audit = SelectionStructureFigureAudit(
        figure_key=figure_spec.key,
        intended_output_name=figure_spec.output_name,
        profile=figure_spec.profile,
        canvas_name=figure_spec.canvas.name,
        canvas_size_inches=figure_spec.canvas.size_inches,
        minimum_mutation_count=minimum_mutation_count,
        functional_sites=tuple(functional_audits),
        selection_curves=tuple(curve_audits),
        structure_assets=tuple(str(path) for path in structure_paths),
        saved_path=saved_path,
    )
    return SelectionFigureRenderResult(
        figure=figure,
        axes=(
            h3_structure_axis,
            *h3_hist_axes,
            h1_structure_axis,
            *h1_hist_axes,
            colorbar_axis,
            ha_sequence_axis,
            na_sequence_axis,
        ),
        audit=audit,
    )


def _sasa_correlations(curves, stats, np) -> tuple[SelectionSasaCorrelationAudit, ...]:
    results: list[SelectionSasaCorrelationAudit] = []
    for curve in curves:
        positions = np.asarray(curve.positions, dtype=int)
        selection = np.asarray(
            curve.selection_coefficients_smoothed, dtype=float
        )
        sasa = np.asarray(curve.sasa_smoothed, dtype=float)
        domains: tuple[tuple[str, object], ...]
        if curve.definition.segment == "ha":
            domains = (
                ("full", np.ones(positions.shape, dtype=bool)),
                (
                    "HA1",
                    (positions >= HA1_POSITION_RANGE[0])
                    & (positions <= HA1_POSITION_RANGE[1]),
                ),
                (
                    "HA2",
                    (positions >= HA2_POSITION_RANGE[0])
                    & (positions <= HA2_POSITION_RANGE[1]),
                ),
            )
        else:
            domains = (("full", np.ones(positions.shape, dtype=bool)),)
        for domain, domain_mask in domains:
            mask = domain_mask & (sasa > 0)
            x_values = selection[mask]
            y_values = sasa[mask]
            if len(x_values) < 2:
                raise SelectionFigureDataError(
                    f"too few positive-SASA pairs for {curve.definition.label}/{domain}"
                )
            pearson = stats.pearsonr(x_values, y_values)
            spearman = stats.spearmanr(x_values, y_values)
            results.append(
                SelectionSasaCorrelationAudit(
                    label=curve.definition.label,
                    subtype=curve.definition.subtype,
                    segment=curve.definition.segment,
                    domain=domain,
                    paired_count=len(x_values),
                    pearson_r=float(pearson.statistic),
                    pearson_p_value=float(pearson.pvalue),
                    spearman_r=float(spearman.statistic),
                    spearman_p_value=float(spearman.pvalue),
                )
            )
    return tuple(results)


def render_selection_sasa_figure(
    plotting_data_dir: Path | str,
    *,
    figure_spec: FigureSpec,
    output_path: Path | str | None = None,
) -> SelectionFigureRenderResult:
    """Render Supplementary Figure S6 from the consolidated SASA table."""

    curves = collect_selection_curve_data(plotting_data_dir)
    curve_by_label = {curve.definition.label: curve for curve in curves}
    plt, _, np, stats = _load_plotting_dependencies()
    correlations = _sasa_correlations(curves, stats, np)
    panel_audits: list[SelectionSasaPanelAudit] = []

    with figure_style_context(figure_spec) as (resolved_spec, profile):
        figure = plt.figure(figsize=resolved_spec.canvas.size_inches)
        figure.set_size_inches(resolved_spec.canvas.size_inches, forward=False)
        left_grid = figure.add_gridspec(
            2, 1, left=0.075, right=0.4225, top=0.96, bottom=0.11
        )
        right_grid = figure.add_gridspec(
            2, 1, left=0.5825, right=0.93, top=0.96, bottom=0.11
        )
        selection_axes_by_label = {
            "H3": figure.add_subplot(left_grid[0, 0]),
            "H1": figure.add_subplot(left_grid[1, 0]),
            "N2": figure.add_subplot(right_grid[0, 0]),
            "N1": figure.add_subplot(right_grid[1, 0]),
        }
        sasa_axes_by_label = {
            label: axis.twinx()
            for label, axis in selection_axes_by_label.items()
        }
        font_family = profile.typography.font_families[0]

        for label in ("H3", "H1", "N2", "N1"):
            curve = curve_by_label[label]
            selection_axis = selection_axes_by_label[label]
            sasa_axis = sasa_axes_by_label[label]
            is_ha = curve.definition.segment == "ha"
            is_bottom = label in {"H1", "N1"}
            x_max = 600 if is_ha else 500
            selection_tick_max = (
                HA_SELECTION_TICK_MAX if is_ha else NA_SELECTION_TICK_MAX
            )
            selection_limit_max = (
                HA_SELECTION_LIMIT_MAX if is_ha else NA_SELECTION_LIMIT_MAX
            )
            color = profile.palette.protein_colors[label]
            positions = np.asarray(curve.positions, dtype=int)
            selection = np.asarray(
                curve.selection_coefficients_smoothed, dtype=float
            )
            sasa = np.asarray(curve.sasa_smoothed, dtype=float)

            selection_axis.plot(
                positions,
                selection,
                color=color,
                linewidth=profile.legacy_figures.sizeline * 2.0,
                alpha=0.7,
            )
            positive_sasa = sasa > 0
            sasa_axis.plot(
                positions[positive_sasa],
                sasa[positive_sasa],
                color=profile.palette.muted,
                linewidth=profile.legacy_figures.sizeline * 2.0,
                linestyle="--",
                alpha=0.7,
            )
            selection_axis.set_xlim(0, x_max)
            ticks = tuple(range(0, x_max + 1, 100))
            selection_axis.set_xticks(ticks if is_bottom else (0, x_max))
            if not is_bottom:
                selection_axis.set_xticklabels(("", ""))
            selection_axis.set_ylim(0, selection_limit_max)
            selection_axis.set_yticks((0, selection_tick_max))
            if is_ha:
                selection_axis.set_yticks((0.1, 0.2, 0.3), minor=True)
            else:
                selection_axis.set_yticks((0.05, 0.1, 0.15), minor=True)
            _style_open_axis(selection_axis, profile)
            selection_axis.spines["left"].set_bounds(0, selection_tick_max)
            selection_axis.spines["bottom"].set_bounds(0, x_max)
            _set_axis_label(
                selection_axis,
                "y",
                "Relative strength of selection",
                profile,
            )
            if is_bottom:
                _set_axis_label(
                    selection_axis,
                    "x",
                    "Position along sequence",
                    profile,
                )

            # The twinned axes share their x locator.  Do not set x ticks here:
            # doing so would erase the primary axis's 100-position ticks.
            sasa_axis.set_ylim(0, SASA_LIMIT_MAX)
            sasa_axis.set_yticks((0, 20, 40, 60, SASA_TICK_MAX))
            _style_open_axis(sasa_axis, profile, bottom=False, right=True)
            sasa_axis.spines["right"].set_bounds(0, SASA_TICK_MAX)
            _set_axis_label(
                sasa_axis,
                "y",
                "Solvent accessible surface area",
                profile,
            )

            selection_axis.text(
                0.03,
                0.86,
                f"{label} selection",
                transform=selection_axis.transAxes,
                color=color,
                family=font_family,
                fontsize=profile.typography.axis_label_pt,
                fontweight="bold",
                ha="left",
                va="center",
            )
            selection_axis.text(
                0.03,
                0.76,
                f"{label} SASA",
                transform=selection_axis.transAxes,
                color=profile.palette.muted,
                family=font_family,
                fontsize=profile.typography.axis_label_pt,
                fontweight="bold",
                ha="left",
                va="center",
            )
            panel_audits.append(
                SelectionSasaPanelAudit(
                    label=label,
                    subtype=curve.definition.subtype,
                    segment=curve.definition.segment,
                    point_count=len(curve.positions),
                    positive_sasa_count=int(np.sum(positive_sasa)),
                    maximum_selection=max(
                        curve.selection_coefficients_smoothed
                    ),
                    maximum_sasa=max(curve.sasa_smoothed),
                )
            )

        for label in ("H3", "H1"):
            axis = selection_axes_by_label[label]
            axis.plot(
                (21, 341),
                (HA_DOMAIN_SPAN_Y, HA_DOMAIN_SPAN_Y),
                color=profile.palette.ink,
                linewidth=profile.legacy_figures.axwidth,
            )
            axis.plot(
                (349, 562),
                (HA_DOMAIN_SPAN_Y, HA_DOMAIN_SPAN_Y),
                color=profile.palette.ink,
                linewidth=profile.legacy_figures.axwidth,
            )
        selection_axes_by_label["H3"].text(
            (17 + 345) / 2,
            HA_DOMAIN_LABEL_Y,
            "HA1",
            family=font_family,
            fontsize=profile.typography.axis_label_pt,
            color=profile.palette.ink,
            ha="center",
            va="center",
        )
        selection_axes_by_label["H3"].text(
            (345 + 566) / 2,
            HA_DOMAIN_LABEL_Y,
            "HA2",
            family=font_family,
            fontsize=profile.typography.axis_label_pt,
            color=profile.palette.ink,
            ha="center",
            va="center",
        )
        saved_path = _save_figure(figure, output_path, profile)

    audit = SelectionSasaFigureAudit(
        figure_key=figure_spec.key,
        intended_output_name=figure_spec.output_name,
        profile=figure_spec.profile,
        canvas_name=figure_spec.canvas.name,
        canvas_size_inches=figure_spec.canvas.size_inches,
        panels=tuple(panel_audits),
        correlations=correlations,
        saved_path=saved_path,
    )
    visual_order = ("H3", "N2", "H1", "N1")
    return SelectionFigureRenderResult(
        figure=figure,
        axes=tuple(
            axis
            for label in visual_order
            for axis in (
                selection_axes_by_label[label],
                sasa_axes_by_label[label],
            )
        ),
        audit=audit,
    )


__all__ = [
    "ALLELE_FREQUENCY_DISPLAY_THRESHOLD",
    "FREQUENCY_SAMPLE_INTERVAL_DAYS",
    "FREQUENCY_TIME_ORIGIN",
    "HA_DOMAIN_LABEL_Y",
    "HA_DOMAIN_SPAN_Y",
    "HA_SELECTION_LIMIT_MAX",
    "HA_SELECTION_TICK_MAX",
    "HA1_POSITION_RANGE",
    "HA2_POSITION_RANGE",
    "INITIAL_WILD_TYPE_FILTER_THRESHOLD",
    "MIN_MUTATION_COUNT",
    "NA_SELECTION_LIMIT_MAX",
    "NA_SELECTION_TICK_MAX",
    "PROTEIN_DEFINITIONS",
    "SASA_LIMIT_MAX",
    "SASA_TICK_MAX",
    "FunctionalSiteAudit",
    "FrequencySelectionFigureAudit",
    "FrequencySelectionFigureData",
    "FrequencySelectionPanelAudit",
    "FrequencySelectionPanelData",
    "ProteinCurveData",
    "ProteinDefinition",
    "SelectionCurveAudit",
    "SelectionFigureDataError",
    "SelectionFigureRenderResult",
    "SelectionSasaCorrelationAudit",
    "SelectionSasaFigureAudit",
    "SelectionSasaPanelAudit",
    "SelectionStructureFigureAudit",
    "collect_frequency_selection_data",
    "collect_selection_curve_data",
    "render_frequency_selection_figure",
    "render_selection_sasa_figure",
    "render_selection_structure_space_figure",
]
