"""Canonical renderers for background-fitness manuscript figures.

The manuscript comparison counts each mutation once within a subtype/segment
panel.  Preparation first applies the figure's inclusive duration cutoff, then
selects the earliest eligible ``start_time`` observation for each
position/state identity.  The renderers consume those deterministic
mutation-level records without further
scientific filtering.

Plotting dependencies are imported lazily.  This keeps the event and registry
metadata importable in inventory-only environments while retaining explicit,
immutable scientific audit records for every rendered comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from statistics import fmean

from .background_plot_data import BackgroundPanelData, prepare_background_plot_data
from .registry import FigureSpec
from .render import figure_style_context


BACKGROUND_SELECTION_FIGURE_KEY = "paper.ds"
BACKGROUND_INITIAL_ADVANTAGE_FIGURE_KEY = "paper.dw_bg"
BACKGROUND_GROWTH_FIGURE_KEY = "paper.dw_rate"
BACKGROUND_CLADE_FIGURE_KEY = "paper.dw_clade"

BACKGROUND_METRIC_SELECTION = "selection_coefficient"
BACKGROUND_METRIC_INITIAL_ADVANTAGE = (
    "initial_background_minus_population_fitness"
)
BACKGROUND_METRIC_ANNUALIZED_GROWTH = "annualized_background_fitness_change"

BACKGROUND_DATA_RELEASE = "background_fitness_v2"
HISTOGRAM_BINS = tuple(
    -0.02 + index * 0.00075 for index in range(61)
)
HISTOGRAM_X_LIMIT = (-0.02, 0.025)
HISTOGRAM_Y_LIMIT = (0.0, 0.40)


@dataclass(frozen=True, slots=True)
class BackgroundMetricSpec:
    """Scientific and display definition for one background quantity."""

    field: str
    label: str
    summary_label: str
    units: str
    minimum_duration_days: int | None
    y_limits: tuple[float, float]
    y_ticks: tuple[float, ...]
    y_tick_labels: tuple[str, ...]


SELECTION_METRIC = BackgroundMetricSpec(
    field=BACKGROUND_METRIC_SELECTION,
    label="Inferred selection coefficient (%)",
    summary_label="Selection\ncoefficient (%)",
    units="percentage points",
    minimum_duration_days=None,
    y_limits=(-0.006, 0.011),
    y_ticks=(-0.005, 0.0, 0.005, 0.01),
    y_tick_labels=("-0.5", "0", "0.5", "1"),
)
INITIAL_ADVANTAGE_METRIC = BackgroundMetricSpec(
    field=BACKGROUND_METRIC_INITIAL_ADVANTAGE,
    label="Difference between initial background fitness and population average (%)",
    summary_label="Background\nfitness\nadvantage (%)",
    units="percentage points",
    minimum_duration_days=90,
    y_limits=(-0.016, 0.016),
    y_ticks=(-0.015, 0.0, 0.015),
    y_tick_labels=("-1.5", "0", "1.5"),
)
ANNUALIZED_GROWTH_METRIC = BackgroundMetricSpec(
    field=BACKGROUND_METRIC_ANNUALIZED_GROWTH,
    label="Change in background fitness per year (%)",
    summary_label="Background\nfitness growth\n(%/year)",
    units="% per year",
    minimum_duration_days=90,
    y_limits=(-0.022, 0.03),
    y_ticks=(-0.02, -0.01, 0.0, 0.01, 0.02, 0.03),
    y_tick_labels=("-2", "-1", "0", "1", "2", "3"),
)


@dataclass(frozen=True, slots=True)
class BackgroundPanelAudit:
    """One fixed/lost Welch comparison in plotted units."""

    metric: str
    metric_label: str
    units: str
    label: str
    subtype: str
    segment: str
    minimum_duration_days: int | None
    fixed_count: int
    lost_count: int
    fixed_mean: float
    lost_mean: float
    welch_t_statistic: float
    welch_p_value: float


@dataclass(frozen=True, slots=True)
class BackgroundFigureAudit:
    """Immutable provenance and statistics for one rendered figure."""

    figure_key: str
    intended_output_name: str
    profile: str
    canvas_name: str
    canvas_size_inches: tuple[float, float]
    layout: str
    background_source: str
    data_release: str
    event_count_before_filter: int
    retained_event_count: int
    collapsed_observation_count: int
    excluded_by_duration_count: int
    panels: tuple[BackgroundPanelAudit, ...]
    clade_source: str | None
    selected_clades: tuple[str, ...]
    clade_row_counts: tuple[tuple[str, int], ...]
    saved_path: str | None

    @property
    def mutation_count_before_filter(self) -> int:
        """Number of mutation identities before duration filtering."""

        return self.event_count_before_filter

    @property
    def retained_mutation_count(self) -> int:
        """Number of mutation identities retained for plotting."""

        return self.retained_event_count


@dataclass(frozen=True, slots=True)
class BackgroundRenderResult:
    """Rendered Matplotlib objects plus immutable scientific audit metadata."""

    figure: object
    summary_axes: tuple[object, ...]
    clade_axes: tuple[object, ...]
    audit: BackgroundFigureAudit

    @property
    def axes(self) -> tuple[object, ...]:
        return (*self.summary_axes, *self.clade_axes)


def _plotting_dependencies():
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Matplotlib, NumPy, and SciPy are required to render background "
            "figures."
        ) from error
    return plt, mdates, np, stats


def _panel_values(
    panel: BackgroundPanelData,
    metric: BackgroundMetricSpec,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return fixed and lost values in the fractional units stored in the data."""

    fixed = tuple(float(value) for value in getattr(panel.fixed, metric.field))
    lost = tuple(float(value) for value in getattr(panel.lost, metric.field))
    if not fixed or not lost:
        raise ValueError(
            f"{panel.label} requires at least one fixed and one lost mutation for "
            f"{metric.field}"
        )
    if not all(isfinite(value) for value in (*fixed, *lost)):
        raise ValueError(f"{panel.label} {metric.field} contains non-finite values")
    return fixed, lost


def _panel_audit(
    panel: BackgroundPanelData,
    metric: BackgroundMetricSpec,
    *,
    stats: object,
    minimum_duration_days: int | None,
) -> BackgroundPanelAudit:
    fixed_raw, lost_raw = _panel_values(panel, metric)
    fixed = tuple(100.0 * value for value in fixed_raw)
    lost = tuple(100.0 * value for value in lost_raw)
    test = stats.ttest_ind(fixed_raw, lost_raw, equal_var=False)
    return BackgroundPanelAudit(
        metric=metric.field,
        metric_label=metric.label,
        units=metric.units,
        label=panel.label,
        subtype=panel.subtype,
        segment=panel.segment,
        minimum_duration_days=minimum_duration_days,
        fixed_count=len(fixed),
        lost_count=len(lost),
        fixed_mean=fmean(fixed),
        lost_mean=fmean(lost),
        welch_t_statistic=float(test.statistic),
        welch_p_value=float(test.pvalue),
    )


def _save_figure(
    figure: object,
    output_path: Path | str | None,
    *,
    profile: object,
) -> str | None:
    if output_path is None:
        return None
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_options: dict[str, object] = {
        "dpi": profile.output.dpi,
        "transparent": profile.output.transparent,
        "format": destination.suffix.lstrip(".") or profile.output.format,
    }
    if profile.output.bbox_inches is not None:
        save_options["bbox_inches"] = profile.output.bbox_inches
    figure.savefig(destination, **save_options)
    return str(destination)


def _style_open_axis(axis: object, profile: object) -> None:
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(profile.palette.ink)
        axis.spines[side].set_linewidth(profile.legacy_figures.axwidth)
    axis.tick_params(
        axis="both",
        which="major",
        direction="out",
        top=False,
        right=False,
        colors=profile.palette.ink,
        length=profile.strokes.tick_length_pt,
        width=profile.legacy_figures.tick_width,
        pad=profile.strokes.tick_pad_pt,
    )
    axis.tick_params(
        axis="both",
        which="minor",
        direction="out",
        top=False,
        right=False,
        colors=profile.palette.ink,
        length=max(0.5, profile.strokes.tick_length_pt - 1.25),
        width=profile.legacy_figures.tick_width,
    )


def _histogram_axes(figure: object) -> tuple[object, ...]:
    grid = figure.add_gridspec(
        2,
        2,
        left=0.12,
        right=0.95,
        top=0.95,
        bottom=0.12,
    )
    return tuple(
        figure.add_subplot(grid[row, column])
        for row in range(2)
        for column in range(2)
    )


def _render_histogram_panels(
    *,
    figure: object,
    axes: tuple[object, ...],
    panels: tuple[BackgroundPanelData, ...],
    metric: BackgroundMetricSpec,
    profile: object,
    np: object,
    stats: object,
) -> tuple[BackgroundPanelAudit, ...]:
    font_family = profile.typography.font_families[0]
    panel_audits: list[BackgroundPanelAudit] = []
    major_x_ticks = (-0.02, -0.01, 0.0, 0.01, 0.02)
    minor_x_ticks = (-0.015, -0.005, 0.005, 0.015, 0.025)
    major_y_ticks = (0.0, 0.1, 0.2, 0.3, 0.4)
    minor_y_ticks = (0.05, 0.15, 0.25, 0.35)

    for index, (axis, panel) in enumerate(zip(axes, panels, strict=True)):
        fixed, lost = _panel_values(panel, metric)
        fixed_color = profile.palette.protein_colors[panel.label]
        histogram_options = {
            "bins": HISTOGRAM_BINS,
            "histtype": "bar",
            "rwidth": 0.8,
            "linewidth": profile.legacy_figures.sizeline / 2.0,
            "alpha": 0.7,
            "edgecolor": "none",
        }
        axis.hist(
            lost,
            weights=np.full(len(lost), 1.0 / len(lost)),
            color=profile.palette.neutral,
            **histogram_options,
        )
        axis.hist(
            fixed,
            weights=np.full(len(fixed), 1.0 / len(fixed)),
            color=fixed_color,
            **histogram_options,
        )

        axis.set_xlim(*HISTOGRAM_X_LIMIT)
        axis.set_ylim(*HISTOGRAM_Y_LIMIT)
        axis.set_xticks(major_x_ticks)
        axis.set_xticks(minor_x_ticks, minor=True)
        axis.set_yticks(major_y_ticks)
        axis.set_yticks(minor_y_ticks, minor=True)
        axis.set_xticklabels(("-2", "-1", "0", "1", "2"))
        if index < 2:
            axis.tick_params(axis="x", labelbottom=False)
        if index % 2:
            axis.tick_params(axis="y", labelleft=False)
        _style_open_axis(axis, profile)

        text = {
            "family": font_family,
            "fontsize": profile.typography.axis_label_pt,
            "va": "center",
            "clip_on": False,
        }
        audit = _panel_audit(
            panel,
            metric,
            stats=stats,
            minimum_duration_days=metric.minimum_duration_days,
        )
        panel_audits.append(audit)
        axis.text(
            -0.018,
            0.38,
            panel.label,
            color=profile.palette.ink,
            ha="left",
            **text,
        )
        axis.text(
            -0.018,
            0.35,
            f"p={audit.welch_p_value:.1e}",
            color=profile.palette.ink,
            ha="left",
            **text,
        )
        axis.text(
            0.023,
            0.38,
            "fixed",
            color=fixed_color,
            fontweight="bold",
            ha="right",
            **text,
        )
        axis.text(
            0.023,
            0.35,
            "lost",
            color=profile.palette.neutral,
            fontweight="bold",
            ha="right",
            **text,
        )

    label_options = {
        "family": font_family,
        "fontsize": profile.typography.axis_label_pt,
        "color": profile.palette.ink,
        "ha": "center",
        "va": "center",
    }
    figure.text(0.04, 0.535, "Frequency", rotation=90, **label_options)
    figure.text(0.535, 0.04, metric.label, **label_options)
    return tuple(panel_audits)


def _render_background_histogram(
    background_csv: Path | str,
    *,
    metric: BackgroundMetricSpec,
    figure_spec: FigureSpec | str,
    output_path: Path | str | None,
) -> BackgroundRenderResult:
    plt, _, np, stats = _plotting_dependencies()
    source = Path(background_csv).expanduser().resolve()
    prepared = prepare_background_plot_data(
        source,
        minimum_duration_days=metric.minimum_duration_days,
    )

    with figure_style_context(figure_spec) as (resolved_spec, profile):
        figure = plt.figure(figsize=resolved_spec.canvas.size_inches)
        # Interactive backends may quantize the initial size to screen pixels.
        # Reset it without resizing a GUI window so vector output keeps the
        # registered physical canvas exactly.
        figure.set_size_inches(resolved_spec.canvas.size_inches, forward=False)
        axes = _histogram_axes(figure)
        panel_audits = _render_histogram_panels(
            figure=figure,
            axes=axes,
            panels=prepared.panels,
            metric=metric,
            profile=profile,
            np=np,
            stats=stats,
        )
        saved_path = _save_figure(figure, output_path, profile=profile)

    extraction = prepared.audit
    audit = BackgroundFigureAudit(
        figure_key=resolved_spec.key,
        intended_output_name=resolved_spec.output_name,
        profile=resolved_spec.profile,
        canvas_name=resolved_spec.canvas.name,
        canvas_size_inches=resolved_spec.canvas.size_inches,
        layout="two_by_two_fixed_lost_histograms",
        background_source=str(source),
        data_release=BACKGROUND_DATA_RELEASE,
        event_count_before_filter=extraction.event_count_before_filter,
        retained_event_count=extraction.retained_event_count,
        collapsed_observation_count=extraction.collapsed_observation_count,
        excluded_by_duration_count=extraction.excluded_by_duration_count,
        panels=panel_audits,
        clade_source=None,
        selected_clades=(),
        clade_row_counts=(),
        saved_path=saved_path,
    )
    return BackgroundRenderResult(
        figure=figure,
        summary_axes=axes,
        clade_axes=(),
        audit=audit,
    )


def render_background_selection_figure(
    background_csv: Path | str,
    *,
    figure_spec: FigureSpec | str = BACKGROUND_SELECTION_FIGURE_KEY,
    output_path: Path | str | None = None,
) -> BackgroundRenderResult:
    """Render Supplementary Figure S13 without a duration filter."""

    return _render_background_histogram(
        background_csv,
        metric=SELECTION_METRIC,
        figure_spec=figure_spec,
        output_path=output_path,
    )


def render_background_initial_advantage_figure(
    background_csv: Path | str,
    *,
    figure_spec: FigureSpec | str = BACKGROUND_INITIAL_ADVANTAGE_FIGURE_KEY,
    output_path: Path | str | None = None,
) -> BackgroundRenderResult:
    """Render Supplementary Figure S14 with the inclusive 90-day filter."""

    return _render_background_histogram(
        background_csv,
        metric=INITIAL_ADVANTAGE_METRIC,
        figure_spec=figure_spec,
        output_path=output_path,
    )


def render_background_growth_figure(
    background_csv: Path | str,
    *,
    figure_spec: FigureSpec | str = BACKGROUND_GROWTH_FIGURE_KEY,
    output_path: Path | str | None = None,
) -> BackgroundRenderResult:
    """Render Supplementary Figure S15 with annualized 90-day-filtered data."""

    return _render_background_histogram(
        background_csv,
        metric=ANNUALIZED_GROWTH_METRIC,
        figure_spec=figure_spec,
        output_path=output_path,
    )


def _deterministic_offsets(count: int, *, width: float = 0.10) -> tuple[float, ...]:
    """Return stable, low-discrepancy jitter without using random state."""

    def van_der_corput(index: int) -> float:
        value = 0.0
        denominator = 1.0
        while index:
            index, remainder = divmod(index, 2)
            denominator *= 2.0
            value += remainder / denominator
        return value

    return tuple(
        width * (2.0 * van_der_corput(index) - 1.0)
        for index in range(1, count + 1)
    )


def _summary_axes(figure: object) -> tuple[tuple[object, ...], ...]:
    left, right, top, bottom = 0.14, 0.50, 0.90, 0.03
    row_gap = 0.06
    row_height = (top - bottom - 2.0 * row_gap) / 3.0
    rows: list[tuple[object, ...]] = []
    for row in range(3):
        row_top = top - row * (row_height + row_gap)
        grid = figure.add_gridspec(
            1,
            4,
            left=left,
            right=right,
            top=row_top,
            bottom=row_top - row_height,
        )
        rows.append(tuple(figure.add_subplot(grid[0, column]) for column in range(4)))
    return tuple(rows)


def _significance_stars(p_value: float) -> str:
    if not isfinite(p_value) or p_value >= 0.05:
        return ""
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    return "*"


def _render_summary_row(
    *,
    axes: tuple[object, ...],
    panels: tuple[BackgroundPanelData, ...],
    metric: BackgroundMetricSpec,
    profile: object,
    np: object,
    stats: object,
    show_group_labels: bool,
    minimum_duration_days: int | None,
) -> tuple[BackgroundPanelAudit, ...]:
    font_family = profile.typography.font_families[0]
    audits: list[BackgroundPanelAudit] = []
    for index, (axis, panel) in enumerate(zip(axes, panels, strict=True)):
        fixed, lost = _panel_values(panel, metric)
        fixed_color = profile.palette.protein_colors[panel.label]
        fixed_x = np.asarray(_deterministic_offsets(len(fixed))) + 0.5
        lost_x = np.asarray(_deterministic_offsets(len(lost))) + 1.5
        axis.scatter(
            fixed_x,
            fixed,
            s=profile.legacy_figures.smallsizedot,
            linewidths=0,
            marker="o",
            alpha=0.25,
            color=fixed_color,
        )
        axis.scatter(
            lost_x,
            lost,
            s=profile.legacy_figures.smallsizedot,
            linewidths=0,
            marker="o",
            alpha=0.25,
            color=profile.palette.neutral_light,
        )
        for x_value, values, face_color in (
            (0.5, fixed, fixed_color),
            (1.5, lost, profile.palette.neutral_light),
        ):
            axis.errorbar(
                [x_value],
                [fmean(values)],
                yerr=[float(np.std(values, ddof=0))],
                fmt="o",
                markersize=profile.legacy_figures.error_marker_size,
                markeredgewidth=profile.legacy_figures.axwidth,
                markeredgecolor=profile.palette.ink,
                markerfacecolor=face_color,
                ecolor=profile.palette.ink,
                elinewidth=profile.legacy_figures.axwidth,
                capsize=0,
                zorder=4,
            )

        axis.set_xlim(0.0, 2.0)
        axis.set_ylim(*metric.y_limits)
        axis.set_xticks(())
        axis.set_yticks(metric.y_ticks)
        axis.set_yticklabels(metric.y_tick_labels)
        _style_open_axis(axis, profile)
        # Keep display padding for points and annotations while drawing the
        # visible axis line only between its outer major tick marks.
        axis.spines["left"].set_bounds(
            min(metric.y_ticks),
            max(metric.y_ticks),
        )
        axis.spines["bottom"].set_visible(False)
        if index:
            axis.spines["left"].set_visible(False)
            axis.tick_params(axis="y", left=False, labelleft=False)

        audit = _panel_audit(
            panel,
            metric,
            stats=stats,
            minimum_duration_days=minimum_duration_days,
        )
        audits.append(audit)
        text = {
            "family": font_family,
            "fontsize": profile.typography.axis_label_pt,
            "ha": "center",
            "va": "center",
            "clip_on": False,
        }
        label_y = metric.y_limits[1] + 0.004
        star_y = metric.y_limits[1]
        if show_group_labels:
            axis.text(1.0, label_y, panel.label, color=profile.palette.ink, **text)
        stars = _significance_stars(audit.welch_p_value)
        if stars:
            axis.text(1.0, star_y, stars, color=profile.palette.ink, **text)
        if show_group_labels and index == 0:
            axis.text(
                0.25,
                metric.y_limits[1] + 0.001,
                "fixed",
                color=fixed_color,
                fontweight="bold",
                **text,
            )
            axis.text(
                1.75,
                metric.y_limits[1] - 0.002,
                "lost",
                color=profile.palette.neutral,
                fontweight="bold",
                **text,
            )

    axes[0].set_ylabel(
        metric.summary_label,
        family=font_family,
        fontsize=profile.typography.axis_label_pt,
        color=profile.palette.ink,
    )
    return tuple(audits)


def _read_and_prepare_clades(
    clade_csv: Path,
    *,
    minimum_total_frequency: float,
    start_date: str,
    end_date: str,
    forecast_baseline_date: str,
):
    try:
        import pandas as pd
        from .clades import prepare_clade_frequencies
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("Pandas is required to render the clade panels") from error

    source = pd.read_csv(clade_csv)
    prepared = prepare_clade_frequencies(
        source,
        reference_series="observed_full",
        minimum_total_frequency=minimum_total_frequency,
        reference_start_date=start_date,
        reference_end_date=end_date,
    )
    data = prepared.data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    baseline = pd.Timestamp(forecast_baseline_date)
    past = data[(data["series"] == "observed_history") & (data["date"] > start)].copy()
    full = data[
        (data["series"] == "observed_full")
        & (data["date"] > start)
        & (data["date"] < end)
    ].copy()
    forecast = data[
        (data["series"] == "forecast") & (data["date"] != baseline)
    ].copy()
    if past.empty or full.empty or forecast.empty:
        raise ValueError(
            "Clade history, truth, and forecast series must all be non-empty"
        )
    if forecast["date"].nunique() != 4:
        raise ValueError(
            "The clade layout requires forecast rows at 3, 6, 9, and 12 months"
        )
    return prepared, past, full, forecast


def _clade_axes(
    figure: object,
    *,
    past_duration_days: float,
    forecast_duration_days: float,
) -> tuple[object, ...]:
    top, bottom, left, right = 0.92, 0.32, 0.60, 0.97
    row_gap, column_gap = 0.09, 0.09
    width = (right - left - column_gap) / 2.0
    forecast_height = (top - bottom) * forecast_duration_days / past_duration_days
    boxes = (
        (left, bottom, width, top - bottom),
        (left + width + column_gap, top - forecast_height, width, forecast_height),
        (
            left + width + column_gap,
            top - 2.0 * forecast_height - row_gap,
            width,
            forecast_height,
        ),
        (
            left + width + column_gap,
            top - 3.0 * forecast_height - 2.0 * row_gap,
            width,
            forecast_height,
        ),
    )
    return tuple(figure.add_axes(box) for box in boxes)


def _pivot_clades(frame: object, clades: tuple[str, ...]):
    pivot = frame.pivot_table(
        index="date",
        columns="clade",
        values="frequency",
        aggfunc="sum",
    ).fillna(0.0)
    return pivot.sort_index().reindex(columns=clades, fill_value=0.0)


def _stacked_clades(
    axis: object,
    *,
    pivot: object,
    colors: tuple[object, ...],
    np: object,
) -> None:
    y_values = pivot.index.to_numpy()
    cumulative = np.zeros(len(pivot), dtype=float)
    for color, clade in zip(colors, pivot.columns, strict=True):
        values = pivot[clade].to_numpy(dtype=float)
        axis.fill_betweenx(
            y_values,
            cumulative,
            cumulative + values,
            facecolor=color,
            linewidth=0,
            edgecolor="none",
        )
        cumulative += values


def _style_clade_axis(
    axis: object,
    *,
    profile: object,
    y_ticks: tuple[object, ...],
    y_tick_labels: tuple[str, ...],
    y_limits: tuple[object, object],
    xlabel: str | None,
) -> None:
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(*y_limits)
    axis.set_xticks((0.0, 0.5, 1.0))
    axis.set_xticklabels(("", "", ""))
    axis.set_xticks((0.25, 0.75), minor=True)
    axis.set_yticks(y_ticks)
    axis.set_yticklabels(y_tick_labels)
    if xlabel is not None:
        axis.set_xlabel(xlabel)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(profile.palette.ink)
        spine.set_linewidth(profile.legacy_figures.axwidth)
    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        colors=profile.palette.ink,
        length=profile.strokes.tick_length_pt,
        width=profile.legacy_figures.tick_width,
        pad=profile.strokes.tick_pad_pt,
    )
    axis.tick_params(
        axis="x",
        which="minor",
        direction="in",
        colors=profile.palette.ink,
        length=max(0.5, profile.strokes.tick_length_pt - 1.25),
        width=profile.legacy_figures.tick_width,
    )


def _render_clade_panels(
    *,
    figure: object,
    prepared: object,
    past: object,
    full: object,
    forecast: object,
    profile: object,
    mdates: object,
    np: object,
) -> tuple[object, ...]:
    try:
        import pandas as pd
        import seaborn as sns
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Seaborn is required for the canonical clade palette"
        ) from error

    clades = tuple(prepared.selected_clades)
    colors = tuple(sns.color_palette("ch:s=-.5,r=.6", n_colors=len(clades)))
    past_pivot = _pivot_clades(past, clades)
    full_pivot = _pivot_clades(full, clades)
    baseline_date = past_pivot.index.max()
    baseline = past_pivot.loc[[baseline_date]].copy()
    forecast_with_baseline = _pivot_clades(
        pd.concat(
            [
                baseline.reset_index().melt(
                    id_vars="date", var_name="clade", value_name="frequency"
                ),
                forecast[["date", "clade", "frequency"]],
            ],
            ignore_index=True,
        ),
        clades,
    )
    forecast_dates = forecast_with_baseline.index
    neutral_values = np.tile(
        baseline.iloc[0].to_numpy(dtype=float),
        (len(forecast_dates), 1),
    )
    neutral = forecast_with_baseline.copy()
    neutral.loc[:, :] = neutral_values

    past_duration = float((past_pivot.index.max() - past_pivot.index.min()).days)
    forecast_duration = float((forecast_dates.max() - forecast_dates.min()).days)
    if past_duration <= 0 or forecast_duration <= 0:
        raise ValueError("Clade series must span positive date intervals")
    axes = _clade_axes(
        figure,
        past_duration_days=past_duration,
        forecast_duration_days=forecast_duration,
    )

    for axis, pivot in zip(
        axes,
        (past_pivot, forecast_with_baseline, full_pivot, neutral),
        strict=True,
    ):
        _stacked_clades(axis, pivot=pivot, colors=colors, np=np)

    past_edges = np.linspace(
        mdates.date2num(past_pivot.index.min()),
        mdates.date2num(past_pivot.index.max()) + 1.0,
        5,
    )
    shifted_past_ticks = past_edges[:-1] + (past_edges[1] - past_edges[0]) / 2.0
    past_ticks = tuple(mdates.num2date(value) for value in shifted_past_ticks)
    past_labels = tuple(value.strftime("%b\n%Y") for value in past_ticks)
    _style_clade_axis(
        axes[0],
        profile=profile,
        y_ticks=past_ticks,
        y_tick_labels=past_labels,
        y_limits=(past_pivot.index.max(), past_pivot.index.min()),
        xlabel="Clade frequency",
    )

    forecast_ticks = tuple(forecast_dates)
    forecast_labels = ("", "3m", "6m", "9m", "12m")
    for axis in axes[1:]:
        _style_clade_axis(
            axis,
            profile=profile,
            y_ticks=forecast_ticks,
            y_tick_labels=forecast_labels,
            y_limits=(forecast_dates.max(), forecast_dates.min()),
            xlabel="Clade frequency" if axis is axes[-1] else None,
        )

    font_family = profile.typography.font_families[0]
    title_options = {
        "family": font_family,
        "fontsize": profile.typography.axis_label_pt,
        "color": profile.palette.ink,
        "ha": "center",
        "va": "center",
    }
    titles = (
        "Past evolution",
        "Model forecast",
        "True frequencies",
        "Neutral forecast",
    )
    for axis, title in zip(axes, titles, strict=True):
        position = axis.get_position()
        figure.text(
            (position.x0 + position.x1) / 2.0,
            position.y1 + 0.03,
            title,
            **title_options,
        )

    legend_x, legend_y = 0.60, 0.20
    legend_dy, legend_dx = 0.04, 0.08
    for index, (clade, color) in enumerate(zip(clades, colors, strict=True)):
        column = 0 if index < 4 else 1
        row = index if index < 4 else index - 4
        figure.text(
            legend_x + column * legend_dx,
            legend_y - row * legend_dy,
            clade,
            family=font_family,
            fontsize=profile.typography.axis_label_pt,
            color=color,
            fontweight="bold",
            ha="left",
            va="center",
        )
    return axes


def render_background_clade_figure(
    background_csv: Path | str,
    clade_csv: Path | str,
    *,
    figure_spec: FigureSpec | str = BACKGROUND_CLADE_FIGURE_KEY,
    output_path: Path | str | None = None,
    minimum_total_frequency: float = 1.0,
    start_date: str = "2011-08-01",
    end_date: str = "2015-06-01",
    forecast_baseline_date: str = "2014-06-01",
) -> BackgroundRenderResult:
    """Render main Figure 5 from 90-day mutations and canonical clade data."""

    plt, mdates, np, stats = _plotting_dependencies()
    background_source = Path(background_csv).expanduser().resolve()
    clade_source = Path(clade_csv).expanduser().resolve()
    prepared_background = prepare_background_plot_data(
        background_source,
        minimum_duration_days=90,
    )
    prepared_clades, past, full, forecast = _read_and_prepare_clades(
        clade_source,
        minimum_total_frequency=minimum_total_frequency,
        start_date=start_date,
        end_date=end_date,
        forecast_baseline_date=forecast_baseline_date,
    )

    with figure_style_context(figure_spec) as (resolved_spec, profile):
        figure = plt.figure(figsize=resolved_spec.canvas.size_inches)
        figure.set_size_inches(resolved_spec.canvas.size_inches, forward=False)
        summary_rows = _summary_axes(figure)
        metrics = (
            SELECTION_METRIC,
            INITIAL_ADVANTAGE_METRIC,
            ANNUALIZED_GROWTH_METRIC,
        )
        audits: list[BackgroundPanelAudit] = []
        for row_index, (axes, metric) in enumerate(
            zip(summary_rows, metrics, strict=True)
        ):
            audits.extend(
                _render_summary_row(
                    axes=axes,
                    panels=prepared_background.panels,
                    metric=metric,
                    profile=profile,
                    np=np,
                    stats=stats,
                    show_group_labels=row_index == 0,
                    minimum_duration_days=prepared_background.minimum_duration_days,
                )
            )
        clade_axes = _render_clade_panels(
            figure=figure,
            prepared=prepared_clades,
            past=past,
            full=full,
            forecast=forecast,
            profile=profile,
            mdates=mdates,
            np=np,
        )

        label_options = {
            "family": profile.typography.font_families[0],
            "fontsize": profile.typography.panel_label_pt,
            "fontweight": "bold",
            "color": "black",
            "ha": "center",
            "va": "center",
        }
        figure.text(0.06, 0.935, "A", **label_options)
        figure.text(
            0.06,
            summary_rows[1][0].get_position().y1 + 0.02,
            "B",
            **label_options,
        )
        figure.text(
            0.06,
            summary_rows[2][0].get_position().y1 + 0.02,
            "C",
            **label_options,
        )
        figure.text(0.55, 0.935, "D", **label_options)
        saved_path = _save_figure(figure, output_path, profile=profile)

    extraction = prepared_background.audit
    audit = BackgroundFigureAudit(
        figure_key=resolved_spec.key,
        intended_output_name=resolved_spec.output_name,
        profile=resolved_spec.profile,
        canvas_name=resolved_spec.canvas.name,
        canvas_size_inches=resolved_spec.canvas.size_inches,
        layout="three_by_four_background_summaries_with_clade_forecast",
        background_source=str(background_source),
        data_release=BACKGROUND_DATA_RELEASE,
        event_count_before_filter=extraction.event_count_before_filter,
        retained_event_count=extraction.retained_event_count,
        collapsed_observation_count=extraction.collapsed_observation_count,
        excluded_by_duration_count=extraction.excluded_by_duration_count,
        panels=tuple(audits),
        clade_source=str(clade_source),
        selected_clades=tuple(prepared_clades.selected_clades),
        clade_row_counts=(
            ("observed_history", len(past)),
            ("observed_full", len(full)),
            ("forecast", len(forecast)),
        ),
        saved_path=saved_path,
    )
    return BackgroundRenderResult(
        figure=figure,
        summary_axes=tuple(axis for row in summary_rows for axis in row),
        clade_axes=clade_axes,
        audit=audit,
    )


__all__ = [
    "ANNUALIZED_GROWTH_METRIC",
    "BACKGROUND_CLADE_FIGURE_KEY",
    "BACKGROUND_DATA_RELEASE",
    "BACKGROUND_GROWTH_FIGURE_KEY",
    "BACKGROUND_INITIAL_ADVANTAGE_FIGURE_KEY",
    "BACKGROUND_SELECTION_FIGURE_KEY",
    "HISTOGRAM_BINS",
    "HISTOGRAM_X_LIMIT",
    "HISTOGRAM_Y_LIMIT",
    "INITIAL_ADVANTAGE_METRIC",
    "SELECTION_METRIC",
    "BackgroundFigureAudit",
    "BackgroundMetricSpec",
    "BackgroundPanelAudit",
    "BackgroundRenderResult",
    "render_background_clade_figure",
    "render_background_growth_figure",
    "render_background_initial_advantage_figure",
    "render_background_selection_figure",
]
