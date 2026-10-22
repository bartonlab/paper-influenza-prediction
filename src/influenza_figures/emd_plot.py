"""Canonical, comparator-driven EMD figure recipes.

The data-collection layer in this module is dependency-light. Matplotlib and
SciPy are imported only inside :func:`render_emd_comparison_figure`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from math import floor, isfinite, log10
from pathlib import Path
from statistics import fmean, median
import sys
from types import MappingProxyType
from typing import Mapping, TextIO

from .emd import EmdComparison, EmdCutoff, compare_emd
from .registry import FigureSpec
from .render import figure_style_context


FULL_MODEL_METHOD = "full_model_cross_validated_rare_filtered"
BASELINE_FULL_MODEL_METHOD = "full_model_rare_filtered"
NULL_METHOD = "null"
SLM_METHOD = "single_locus_model_rare_filtered"
MLR_METHOD = "multinomial_logistic_regression_rare_filtered"
INFERENCE_RELATIVE_CUTOFF_DAY = 1095
# Keep the first and last January ticks equally clear of the vertical frame.
DATE_AXIS_PADDING_DAYS = 50

COMPARATOR_DISPLAY_LABELS: Mapping[str, str] = MappingProxyType(
    {
        NULL_METHOD: "neutral model",
        SLM_METHOD: "single-locus model (SLM)",
        MLR_METHOD: "multinomial logistic regression (MLR)",
        BASELINE_FULL_MODEL_METHOD: "baseline GenModel",
    }
)
COMPARATOR_PLOT_LABELS: Mapping[str, str] = MappingProxyType(
    {
        NULL_METHOD: "neutral",
        SLM_METHOD: "SLM",
        MLR_METHOD: "MLR",
        BASELINE_FULL_MODEL_METHOD: "baseline",
    }
)
PROTOCOL_BY_HORIZON: Mapping[int, str] = MappingProxyType(
    {6: "results_grid", 9: "results_grid", 12: "results_grid"}
)
TARGET_METHOD_BY_HORIZON: Mapping[int, str] = MappingProxyType(
    {
        6: FULL_MODEL_METHOD,
        9: FULL_MODEL_METHOD,
        12: FULL_MODEL_METHOD,
    }
)

# These scales reproduce the archived paper recipes where the canonical data
# permit it.  The old neutral plot deliberately used a narrower scale at nine
# months. Horizon-specific adjustments are used only where the definitive
# cross-validation results would otherwise be clipped. SLM is rounded to the
# next 0.25-wide histogram edge for the same no-clipping rule. The corrected
# October 13 release no longer requires the provisional 12-month expansions.
DELTA_LIMIT_BY_COMPARATOR: Mapping[str, float] = MappingProxyType(
    {
        NULL_METHOD: 5.0,
        SLM_METHOD: 5.75,
        MLR_METHOD: 6.25,
        BASELINE_FULL_MODEL_METHOD: 4.0,
    }
)
DELTA_LIMIT_BY_COMPARATOR_AND_HORIZON: Mapping[
    str, Mapping[int, float]
] = MappingProxyType(
    {
        NULL_METHOD: MappingProxyType({6: 5.0, 9: 4.25, 12: 5.0}),
        SLM_METHOD: MappingProxyType({6: 5.75, 9: 5.75, 12: 5.75}),
        MLR_METHOD: MappingProxyType({6: 6.25, 9: 6.25, 12: 6.25}),
        BASELINE_FULL_MODEL_METHOD: MappingProxyType(
            {6: 4.0, 9: 4.0, 12: 4.0}
        ),
    }
)
DELTA_TICKS_BY_COMPARATOR: Mapping[str, tuple[float, ...]] = MappingProxyType(
    {
        NULL_METHOD: (-4.0, -2.0, 0.0, 2.0, 4.0),
        SLM_METHOD: (-4.0, -2.0, 0.0, 2.0, 4.0),
        MLR_METHOD: (-4.0, -2.0, 0.0, 2.0, 4.0),
        BASELINE_FULL_MODEL_METHOD: (-4.0, -2.0, 0.0, 2.0, 4.0),
    }
)
HISTOGRAM_BIN_WIDTH = 0.25
HISTOGRAM_FREQUENCY_LIMIT = 0.25
P_VALUE_SIGNIFICANCE_THRESHOLD = 0.05
P_VALUE_DECIMAL_DISPLAY_THRESHOLD = 0.01
FULL_MODEL_WORSE_MARKER_KEY = "* full model worse"
HISTOGRAM_FREQUENCY_LIMIT_BY_COMPARATOR: Mapping[str, float] = MappingProxyType(
    {
        NULL_METHOD: HISTOGRAM_FREQUENCY_LIMIT,
        SLM_METHOD: 0.40,
        MLR_METHOD: HISTOGRAM_FREQUENCY_LIMIT,
        BASELINE_FULL_MODEL_METHOD: 0.40,
    }
)
HISTOGRAM_FREQUENCY_LIMIT_BY_COMPARATOR_AND_HORIZON: Mapping[
    str, Mapping[int, float]
] = MappingProxyType(
    {
        # The complete six-month release has two null histograms whose tallest
        # 0.25-wide bin is just above the historical 0.25 ceiling.
        NULL_METHOD: MappingProxyType({6: 0.30, 9: 0.25, 12: 0.25}),
        SLM_METHOD: MappingProxyType({6: 0.40, 9: 0.40, 12: 0.40}),
        MLR_METHOD: MappingProxyType({6: 0.25, 9: 0.25, 12: 0.25}),
        BASELINE_FULL_MODEL_METHOD: MappingProxyType(
            {6: 0.40, 9: 0.40, 12: 0.40}
        ),
    }
)
TRAJECTORY_GUIDE_LEVELS = (-4.0, -2.0, 0.0, 2.0, 4.0)
LEGACY_TRAJECTORY_BOUNDS = MappingProxyType(
    {"left": 0.09, "right": 0.65, "top": 0.95, "bottom": 0.15}
)
LEGACY_DISTRIBUTION_BOUNDS = MappingProxyType(
    {"left": 0.73, "right": 0.95, "top": 0.95, "bottom": 0.15}
)
LEGACY_ROW_HSPACE = 0.40
LEGACY_TRAJECTORY_LINE_MULTIPLIER = 2.5
LEGACY_ZERO_GUIDE_ALPHA = 0.70
LEGACY_SECONDARY_GUIDE_ALPHA = 0.20


class EmdFigureRecipeError(ValueError):
    """Raised when a canonical EMD figure recipe is not scientifically defined."""


@dataclass(frozen=True, slots=True)
class EmdPanelDefinition:
    label: str
    subtype: str
    segment: str


PANEL_DEFINITIONS = (
    EmdPanelDefinition("H3", "h3n2", "ha"),
    EmdPanelDefinition("N2", "h3n2", "na"),
    EmdPanelDefinition("H1", "h1n1", "ha"),
    EmdPanelDefinition("N1", "h1n1", "na"),
)


@dataclass(frozen=True, slots=True)
class EmdPanelData:
    definition: EmdPanelDefinition
    comparison: EmdComparison

    @property
    def observation_dates(self) -> tuple[str, ...]:
        return tuple(str(row["observation_date"]) for row in self.comparison.rows)

    @property
    def deltas(self) -> tuple[float, ...]:
        return self.comparison.deltas


@dataclass(frozen=True, slots=True)
class EmdFigureData:
    horizon_months: int
    evaluation_protocol: str
    target_method: str
    comparator_method: str
    relative_cutoff_day: int
    panels: tuple[EmdPanelData, ...]
    cutoffs: tuple[EmdCutoff, ...]

    @property
    def cutoff_report_lines(self) -> tuple[str, ...]:
        """One printable line per distinct subtype cutoff."""

        return tuple(cutoff.describe() for cutoff in self.cutoffs)


@dataclass(frozen=True, slots=True)
class EmdPanelAudit:
    label: str
    subtype: str
    segment: str
    paired_count: int
    unmatched_target_count: int
    unmatched_comparator_count: int
    first_observation_date: str
    last_observation_date: str
    mean_delta_emd: float
    median_delta_emd: float
    minimum_delta_emd: float
    maximum_delta_emd: float
    histogram_weight_sum: float
    t_statistic: float
    p_value: float


@dataclass(frozen=True, slots=True)
class EmdFigureAudit:
    figure_key: str
    intended_output_name: str
    profile: str
    canvas_name: str
    canvas_size_inches: tuple[float, float]
    horizon_months: int
    evaluation_protocol: str
    target_method: str
    comparator_method: str
    relative_cutoff_day: int
    layout: str
    delta_limit_abs: float
    distribution_delta_limit_abs: float
    delta_ticks: tuple[float, ...]
    histogram_bin_width: float
    histogram_frequency_limit: float
    cutoffs: tuple[EmdCutoff, ...]
    panels: tuple[EmdPanelAudit, ...]
    saved_path: str | None

    @property
    def cutoff_report_lines(self) -> tuple[str, ...]:
        """One printable line per distinct subtype cutoff."""

        return tuple(cutoff.describe() for cutoff in self.cutoffs)


@dataclass(frozen=True, slots=True)
class EmdRenderResult:
    """Rendered Matplotlib objects plus immutable scientific audit metadata."""

    figure: object
    trajectory_axes: tuple[object, ...]
    distribution_axes: tuple[object, ...]
    audit: EmdFigureAudit

    @property
    def axes(self) -> tuple[object, ...]:
        """All eight axes in row-major trajectory/distribution order."""

        return tuple(
            axis
            for pair in zip(
                self.trajectory_axes,
                self.distribution_axes,
                strict=True,
            )
            for axis in pair
        )


def _protocol_for_horizon(horizon_months: int) -> str:
    try:
        return PROTOCOL_BY_HORIZON[horizon_months]
    except KeyError as error:
        choices = ", ".join(str(value) for value in sorted(PROTOCOL_BY_HORIZON))
        raise EmdFigureRecipeError(
            f"Unsupported EMD horizon {horizon_months}; choose {choices} months"
        ) from error


def _target_method_for_horizon(horizon_months: int) -> str:
    try:
        return TARGET_METHOD_BY_HORIZON[horizon_months]
    except KeyError as error:
        choices = ", ".join(str(value) for value in sorted(TARGET_METHOD_BY_HORIZON))
        raise EmdFigureRecipeError(
            f"Unsupported EMD horizon {horizon_months}; choose {choices} months"
        ) from error


def _delta_limit_for_recipe(comparator_method: str, horizon_months: int) -> float:
    try:
        return DELTA_LIMIT_BY_COMPARATOR_AND_HORIZON[comparator_method][
            horizon_months
        ]
    except KeyError as error:
        raise EmdFigureRecipeError(
            "No fixed EMD scale is registered for "
            f"{comparator_method!r} at {horizon_months} months"
        ) from error


def _distribution_delta_limit(
    comparator_method: str,
    trajectory_delta_limit: float,
) -> float:
    """Return the archived histogram range without clipping canonical data."""

    if comparator_method == NULL_METHOD:
        return trajectory_delta_limit + HISTOGRAM_BIN_WIDTH
    return trajectory_delta_limit


def _minor_delta_ticks(
    delta_limit: float,
    *,
    include_boundaries: bool = False,
) -> tuple[float, ...]:
    candidates = (-5.0, -3.0, -1.0, 1.0, 3.0, 5.0)
    values = tuple(
        value for value in candidates if -delta_limit <= value <= delta_limit
    )
    if include_boundaries and delta_limit not in values:
        return (-delta_limit, *values, delta_limit)
    return values


def _histogram_frequency_ticks(
    frequency_limit: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if frequency_limit == HISTOGRAM_FREQUENCY_LIMIT:
        return (0.0, 0.1, 0.2), (0.05, 0.15, 0.25)
    if frequency_limit == 0.30:
        return (0.0, 0.1, 0.2, 0.3), (0.05, 0.15, 0.25)
    return (0.0, 0.2, 0.4), (0.1, 0.3)


def _validate_comparator(comparator_method: str) -> None:
    if comparator_method not in COMPARATOR_DISPLAY_LABELS:
        choices = ", ".join(COMPARATOR_DISPLAY_LABELS)
        raise EmdFigureRecipeError(
            "comparator_method must be an explicit canonical method_id; "
            f"choose one of: {choices}"
        )


def _validate_figure_spec_binding(
    figure_spec: FigureSpec,
    *,
    horizon_months: int,
    comparator_method: str,
) -> None:
    """Refuse to save data under a differently parameterized recipe name."""

    if figure_spec.renderer != "render_emd_comparison_figure":
        raise EmdFigureRecipeError(
            f"Figure spec {figure_spec.key!r} is registered for "
            f"{figure_spec.renderer!r}, not the canonical EMD renderer"
        )
    required = {"horizon_months", "comparator_method"}
    missing = required - set(figure_spec.recipe_parameters)
    if missing:
        raise EmdFigureRecipeError(
            f"Figure spec {figure_spec.key!r} does not bind required EMD "
            f"parameters: {', '.join(sorted(missing))}"
        )

    expected_horizon = figure_spec.recipe_parameters["horizon_months"]
    expected_comparator = figure_spec.recipe_parameters["comparator_method"]
    if expected_horizon != horizon_months:
        raise EmdFigureRecipeError(
            f"Figure spec {figure_spec.key!r} binds horizon_months="
            f"{expected_horizon!r}, but the render requested {horizon_months!r}"
        )
    if expected_comparator != comparator_method:
        raise EmdFigureRecipeError(
            f"Figure spec {figure_spec.key!r} binds comparator_method="
            f"{expected_comparator!r}, but the render requested "
            f"{comparator_method!r}"
        )
    expected_target = figure_spec.recipe_parameters.get("target_method")
    resolved_target = _target_method_for_horizon(horizon_months)
    if expected_target is not None and expected_target != resolved_target:
        raise EmdFigureRecipeError(
            f"Figure spec {figure_spec.key!r} binds target_method="
            f"{expected_target!r}, but the {horizon_months}-month recipe resolves "
            f"to {resolved_target!r}"
        )
    expected_sign = figure_spec.recipe_parameters.get("difference_sign")
    if expected_sign is not None and expected_sign != "comparator_minus_target":
        raise EmdFigureRecipeError(
            f"Figure spec {figure_spec.key!r} has unsupported difference_sign="
            f"{expected_sign!r}; canonical EMD deltas are comparator minus target"
        )


def collect_emd_figure_data(
    emd_dir: Path | str,
    *,
    horizon_months: int,
    comparator_method: str,
) -> EmdFigureData:
    """Collect four exact, cutoff-filtered canonical EMD comparisons.

    Comparator display labels are deliberately not accepted as selectors. The
    caller must supply an explicitly supported canonical method ID.
    """

    _validate_comparator(comparator_method)
    evaluation_protocol = _protocol_for_horizon(horizon_months)
    target_method = _target_method_for_horizon(horizon_months)
    if (
        comparator_method == BASELINE_FULL_MODEL_METHOD
        and target_method != FULL_MODEL_METHOD
    ):
        raise EmdFigureRecipeError(
            "The baseline GenModel comparator requires the cross-validation-"
            "selected best GenModel as its target"
        )
    if comparator_method == target_method:
        raise EmdFigureRecipeError(
            f"Target and comparator resolve to the same EMD method: {target_method}"
        )

    panels: list[EmdPanelData] = []
    distinct_cutoffs: dict[tuple[object, ...], EmdCutoff] = {}
    for definition in PANEL_DEFINITIONS:
        comparison = compare_emd(
            emd_dir,
            target_method=target_method,
            comparator_method=comparator_method,
            evaluation_protocol=evaluation_protocol,
            horizon_months=horizon_months,
            subtype=definition.subtype,
            segment=definition.segment,
            minimum_observation_day=INFERENCE_RELATIVE_CUTOFF_DAY,
        )
        if comparison.paired_count == 0:
            raise EmdFigureRecipeError(
                f"No exact EMD pairs for {definition.label}, {horizon_months} months, "
                f"{comparator_method}, {evaluation_protocol}"
            )
        if (
            comparison.unmatched_target_ids
            or comparison.unmatched_comparator_ids
        ):
            raise EmdFigureRecipeError(
                "Incomplete exact EMD pairing for "
                f"{definition.label}, {horizon_months} months, "
                f"{comparator_method}, {evaluation_protocol}: "
                f"unmatched target={len(comparison.unmatched_target_ids)}, "
                "unmatched comparator="
                f"{len(comparison.unmatched_comparator_ids)}"
            )
        if comparison.protocols != (evaluation_protocol,):
            raise EmdFigureRecipeError(
                f"Unexpected protocols for {definition.label}: {comparison.protocols}"
            )
        panels.append(EmdPanelData(definition, comparison))
        for cutoff in comparison.cutoffs:
            key = (
                cutoff.subtype,
                cutoff.initial_inference_date,
                cutoff.relative_day,
                cutoff.cutoff_date,
                cutoff.source,
            )
            distinct_cutoffs.setdefault(key, cutoff)

    return EmdFigureData(
        horizon_months=horizon_months,
        evaluation_protocol=evaluation_protocol,
        target_method=target_method,
        comparator_method=comparator_method,
        relative_cutoff_day=INFERENCE_RELATIVE_CUTOFF_DAY,
        panels=tuple(panels),
        cutoffs=tuple(distinct_cutoffs.values()),
    )


def _load_plotting_dependencies() -> tuple[object, object, object]:
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from scipy import stats
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Matplotlib and SciPy are required to render an EMD figure; install "
            "the project plotting environment first."
        ) from error
    return plt, mdates, stats


def _histogram_edges(delta_limit: float) -> tuple[float, ...]:
    span = 2.0 * delta_limit
    bin_count = round(span / HISTOGRAM_BIN_WIDTH)
    if abs(bin_count * HISTOGRAM_BIN_WIDTH - span) > 1e-12:
        raise EmdFigureRecipeError(
            f"Delta limit {delta_limit} does not align to the "
            f"{HISTOGRAM_BIN_WIDTH}-unit histogram grid"
        )
    return tuple(
        -delta_limit + index * HISTOGRAM_BIN_WIDTH
        for index in range(bin_count + 1)
    )


def _p_value_label(
    p_value: float,
    *,
    full_model_is_worse: bool,
) -> str | None:
    """Return a significant-only p-value label for an EMD panel.

    Canonical deltas are comparator EMD minus full-model EMD, so a negative
    mean delta means the full model performs worse. A trailing asterisk marks
    that direction only when the two-sided test is significant.
    """

    if (
        not isfinite(p_value)
        or p_value < 0.0
        or p_value > P_VALUE_SIGNIFICANCE_THRESHOLD
    ):
        return None
    if p_value > P_VALUE_DECIMAL_DISPLAY_THRESHOLD:
        label = rf"$p = {p_value:.2f}$"
        return f"{label}*" if full_model_is_worse else label
    if p_value == 0.0:
        label = r"$p < 10^{-300}$"
    else:
        exponent = floor(log10(p_value)) + 1
        label = rf"$p < 10^{{{exponent:d}}}$"
    return f"{label}*" if full_model_is_worse else label


def _legacy_date_ticks(
    data: EmdFigureData,
) -> tuple[tuple[date, ...], date, date]:
    """Return the archived eight-tick layout while using real calendar dates."""

    panel = data.panels[0]
    dates = tuple(date.fromisoformat(value) for value in panel.observation_dates)
    first = next(
        date.fromisoformat(cutoff.cutoff_date)
        for cutoff in data.cutoffs
        if cutoff.subtype == panel.definition.subtype
    )
    last = max(dates) + timedelta(days=1)
    span_days = (last - first).days
    ticks = tuple(
        first + timedelta(days=int(index * span_days / 7.0))
        for index in range(8)
    )
    padding = timedelta(days=DATE_AXIS_PADDING_DAYS)
    return ticks, first - padding, last + padding


def render_emd_comparison_figure(
    emd_dir: Path | str,
    *,
    horizon_months: int,
    comparator_method: str,
    figure_spec: FigureSpec,
    output_path: Path | str | None = None,
    report_stream: TextIO | None = None,
) -> EmdRenderResult:
    """Render one canonical EMD comparison as four paired panel rows.

    Each protein has a time trajectory at left and a normalized-frequency
    distribution at right, matching the structural organization of the
    archived EMD figures. Typography, output settings, and physical dimensions
    come from the supplied ``FigureSpec``. If ``output_path`` is omitted, no
    file is written. Distinct cutoff descriptions are always returned in the
    audit and printed to standard output by default; ``report_stream`` may
    redirect that report.
    """

    _validate_figure_spec_binding(
        figure_spec,
        horizon_months=horizon_months,
        comparator_method=comparator_method,
    )
    data = collect_emd_figure_data(
        emd_dir,
        horizon_months=horizon_months,
        comparator_method=comparator_method,
    )
    report_target = sys.stdout if report_stream is None else report_stream
    for line in data.cutoff_report_lines:
        print(line, file=report_target)

    plt, mdates, stats = _load_plotting_dependencies()
    comparator_plot_label = COMPARATOR_PLOT_LABELS[comparator_method]
    delta_limit = _delta_limit_for_recipe(comparator_method, horizon_months)
    distribution_delta_limit = _distribution_delta_limit(
        comparator_method,
        delta_limit,
    )
    delta_ticks = DELTA_TICKS_BY_COMPARATOR[comparator_method]
    histogram_edges = _histogram_edges(distribution_delta_limit)
    histogram_frequency_limit = (
        HISTOGRAM_FREQUENCY_LIMIT_BY_COMPARATOR_AND_HORIZON[
            comparator_method
        ][horizon_months]
    )
    frequency_ticks, frequency_minor_ticks = _histogram_frequency_ticks(
        histogram_frequency_limit
    )
    include_scale_boundaries = comparator_method != NULL_METHOD
    trajectory_minor_ticks = _minor_delta_ticks(
        delta_limit,
        include_boundaries=include_scale_boundaries,
    )
    distribution_minor_ticks = _minor_delta_ticks(
        distribution_delta_limit,
        include_boundaries=include_scale_boundaries,
    )
    date_ticks, date_left, date_right = _legacy_date_ticks(data)
    panel_audits: list[EmdPanelAudit] = []

    with figure_style_context(figure_spec) as (resolved_spec, profile):
        figure = plt.figure(figsize=resolved_spec.canvas.size_inches)
        # Interactive backends may initially quantize figsize to whole screen
        # pixels. Reset the physical size without forwarding it to the window
        # manager so vector output retains the registered paper canvas exactly.
        figure.set_size_inches(resolved_spec.canvas.size_inches, forward=False)
        trajectory_grid = figure.add_gridspec(
            len(data.panels),
            1,
            hspace=LEGACY_ROW_HSPACE,
            **LEGACY_TRAJECTORY_BOUNDS,
        )
        distribution_grid = figure.add_gridspec(
            len(data.panels),
            1,
            hspace=LEGACY_ROW_HSPACE,
            **LEGACY_DISTRIBUTION_BOUNDS,
        )
        trajectory_axes = tuple(
            figure.add_subplot(trajectory_grid[index, 0])
            for index in range(len(data.panels))
        )
        distribution_axes = tuple(
            figure.add_subplot(distribution_grid[index, 0])
            for index in range(len(data.panels))
        )

        font_family = profile.typography.font_families[0]
        data_line_width = (
            profile.legacy_figures.sizeline
            * LEGACY_TRAJECTORY_LINE_MULTIPLIER
        )
        guide_line_width = profile.legacy_figures.sizeline
        histogram_edge_width = profile.legacy_figures.sizeline / 2.0
        minor_tick_length = profile.strokes.tick_length_pt - 1.25

        for index, (trajectory_axis, distribution_axis, panel) in enumerate(
            zip(trajectory_axes, distribution_axes, data.panels, strict=True)
        ):
            dates = tuple(
                date.fromisoformat(value) for value in panel.observation_dates
            )
            deltas = panel.deltas
            test = stats.ttest_1samp(deltas, 0.0, nan_policy="omit")
            t_statistic = float(test.statistic)
            p_value = float(test.pvalue)
            mean_delta = fmean(deltas)
            median_delta = median(deltas)
            color = profile.palette.protein_colors[panel.definition.label]

            trajectory_axis.plot(
                dates,
                deltas,
                color=color,
                linewidth=data_line_width,
                linestyle="-",
                alpha=1.0,
            )
            for guide_level in TRAJECTORY_GUIDE_LEVELS:
                guide_alpha = (
                    LEGACY_ZERO_GUIDE_ALPHA
                    if guide_level == 0.0
                    else LEGACY_SECONDARY_GUIDE_ALPHA
                )
                trajectory_axis.axhline(
                    guide_level,
                    color=profile.palette.muted,
                    linewidth=guide_line_width,
                    linestyle=":",
                    alpha=guide_alpha,
                )
            trajectory_axis.set_xlim(date_left, date_right)
            trajectory_axis.set_xticks(date_ticks)
            trajectory_axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            trajectory_axis.set_ylim(-delta_limit, delta_limit)
            trajectory_axis.set_yticks(delta_ticks)
            trajectory_axis.set_yticks(trajectory_minor_ticks, minor=True)

            weights = tuple(1.0 / len(deltas) for _ in deltas)
            distribution_axis.hist(
                deltas,
                bins=histogram_edges,
                weights=weights,
                color=color,
                edgecolor="none",
                linewidth=histogram_edge_width,
                alpha=1.0,
                rwidth=0.8,
                align="mid",
            )
            distribution_axis.axvline(
                0.0,
                color=profile.palette.muted,
                linewidth=guide_line_width,
                linestyle=":",
                alpha=LEGACY_ZERO_GUIDE_ALPHA,
            )
            distribution_axis.set_xlim(
                -distribution_delta_limit,
                distribution_delta_limit,
            )
            distribution_axis.set_xticks(delta_ticks)
            distribution_axis.set_xticks(distribution_minor_ticks, minor=True)
            distribution_axis.set_ylim(0.0, histogram_frequency_limit)
            distribution_axis.set_yticks(frequency_ticks)
            distribution_axis.set_yticks(frequency_minor_ticks, minor=True)

            for axis in (trajectory_axis, distribution_axis):
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                axis.spines["bottom"].set_visible(True)
                axis.spines["left"].set_visible(True)
                for spine_name in ("bottom", "left"):
                    axis.spines[spine_name].set_color(profile.palette.ink)
                    axis.spines[spine_name].set_linewidth(
                        profile.legacy_figures.axwidth
                    )
                axis.tick_params(
                    axis="both",
                    which="major",
                    direction="out",
                    colors=profile.palette.ink,
                    bottom=True,
                    left=True,
                    top=False,
                    right=False,
                    length=profile.strokes.tick_length_pt,
                    width=profile.legacy_figures.tick_width,
                    pad=profile.strokes.tick_pad_pt,
                )
                axis.tick_params(
                    axis="both",
                    which="minor",
                    direction="out",
                    colors=profile.palette.ink,
                    bottom=True,
                    left=True,
                    top=False,
                    right=False,
                    length=minor_tick_length,
                    width=profile.legacy_figures.tick_width,
                )
                if index < len(data.panels) - 1:
                    axis.tick_params(axis="x", labelbottom=False)
                for tick_label in (
                    *axis.get_xticklabels(),
                    *axis.get_yticklabels(),
                ):
                    tick_label.set_fontfamily(font_family)
                    tick_label.set_fontsize(profile.typography.tick_label_pt)
                    tick_label.set_color(profile.palette.ink)

            trajectory_axis.spines["left"].set_bounds(-delta_limit, delta_limit)
            trajectory_axis.spines["bottom"].set_bounds(
                mdates.date2num(date_ticks[0]),
                mdates.date2num(date_ticks[-1]),
            )
            distribution_axis.spines["left"].set_bounds(
                0.0,
                histogram_frequency_limit,
            )
            distribution_axis.spines["bottom"].set_bounds(
                min((*delta_ticks, *distribution_minor_ticks)),
                max((*delta_ticks, *distribution_minor_ticks)),
            )
            distribution_axis.spines["left"].set_position(("outward", 0.1))
            distribution_axis.spines["bottom"].set_position(("outward", 0.1))

            annotation_base_y = (
                LEGACY_DISTRIBUTION_BOUNDS["top"]
                - (
                    LEGACY_DISTRIBUTION_BOUNDS["top"]
                    - LEGACY_DISTRIBUTION_BOUNDS["bottom"]
                    + 0.06
                )
                * (index / 4.0)
            )
            label_offset = 0.01 if comparator_method == NULL_METHOD else 0.02
            figure.text(
                LEGACY_DISTRIBUTION_BOUNDS["right"] + 0.005,
                annotation_base_y - label_offset,
                panel.definition.label,
                color=color,
                family=font_family,
                fontsize=profile.typography.axis_label_pt,
                fontweight="bold",
                ha="right",
                va="center",
                clip_on=False,
            )
            p_value_label = _p_value_label(
                p_value,
                full_model_is_worse=mean_delta < 0.0,
            )
            if p_value_label is not None:
                figure.text(
                    LEGACY_DISTRIBUTION_BOUNDS["right"] + 0.005,
                    annotation_base_y - label_offset - 0.04,
                    p_value_label,
                    color=profile.palette.ink,
                    family=font_family,
                    fontsize=profile.typography.axis_label_pt,
                    ha="right",
                    va="center",
                    clip_on=False,
                )

            panel_audits.append(
                EmdPanelAudit(
                    label=panel.definition.label,
                    subtype=panel.definition.subtype,
                    segment=panel.definition.segment,
                    paired_count=panel.comparison.paired_count,
                    unmatched_target_count=len(
                        panel.comparison.unmatched_target_ids
                    ),
                    unmatched_comparator_count=len(
                        panel.comparison.unmatched_comparator_ids
                    ),
                    first_observation_date=panel.observation_dates[0],
                    last_observation_date=panel.observation_dates[-1],
                    mean_delta_emd=mean_delta,
                    median_delta_emd=median_delta,
                    minimum_delta_emd=min(deltas),
                    maximum_delta_emd=max(deltas),
                    histogram_weight_sum=sum(weights),
                    t_statistic=t_statistic,
                    p_value=p_value,
                )
            )

        trajectory_axes[-1].set_xlabel(
            "Time prediction made",
            color=profile.palette.ink,
            family=font_family,
            fontsize=profile.typography.axis_label_pt,
        )
        comparator_axis_label = (
            f"Model improvement vs. {comparator_plot_label}\n"
            + r"distance to future $(\Delta EMD)$"
        )
        distribution_axes[-1].set_xlabel(
            comparator_axis_label,
            color=profile.palette.ink,
            family=font_family,
            fontsize=profile.typography.axis_label_pt,
        )

        common_label_properties = {
            "family": font_family,
            "fontsize": profile.typography.axis_label_pt,
            "color": profile.palette.ink,
            "ha": "center",
            "va": "center",
        }
        figure.text(
            LEGACY_TRAJECTORY_BOUNDS["left"] - 0.06,
            (
                LEGACY_TRAJECTORY_BOUNDS["top"]
                + LEGACY_TRAJECTORY_BOUNDS["bottom"]
            )
            / 2.0,
            comparator_axis_label,
            rotation=90,
            **common_label_properties,
        )
        figure.text(
            LEGACY_DISTRIBUTION_BOUNDS["left"] - 0.05,
            (
                LEGACY_DISTRIBUTION_BOUNDS["top"]
                + LEGACY_DISTRIBUTION_BOUNDS["bottom"]
            )
            / 2.0,
            "Frequency",
            rotation=90,
            **common_label_properties,
        )
        panel_label_properties = {
            "family": font_family,
            "fontsize": profile.typography.panel_label_pt,
            "fontweight": "bold",
            "color": "black",
            "ha": "center",
            "va": "center",
            "clip_on": False,
        }
        figure.text(
            LEGACY_TRAJECTORY_BOUNDS["left"] - 0.05,
            LEGACY_TRAJECTORY_BOUNDS["top"] + 0.02,
            "A",
            **panel_label_properties,
        )
        figure.text(
            LEGACY_DISTRIBUTION_BOUNDS["left"] - 0.05,
            LEGACY_DISTRIBUTION_BOUNDS["top"] + 0.02,
            "B",
            **panel_label_properties,
        )
        if any(
            panel.mean_delta_emd < 0.0
            and 0.0 <= panel.p_value <= P_VALUE_SIGNIFICANCE_THRESHOLD
            for panel in panel_audits
        ):
            figure.text(
                LEGACY_DISTRIBUTION_BOUNDS["right"],
                LEGACY_DISTRIBUTION_BOUNDS["top"] + 0.02,
                FULL_MODEL_WORSE_MARKER_KEY,
                color=profile.palette.ink,
                family=font_family,
                fontsize=profile.typography.tick_label_pt - 1.0,
                ha="right",
                va="center",
                clip_on=False,
            )

        saved_path: str | None = None
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            save_options: dict[str, object] = {
                "dpi": profile.output.dpi,
                "transparent": profile.output.transparent,
                "format": destination.suffix.lstrip(".") or profile.output.format,
            }
            if profile.output.bbox_inches is not None:
                save_options["bbox_inches"] = profile.output.bbox_inches
            figure.savefig(destination, **save_options)
            saved_path = str(destination)

    audit = EmdFigureAudit(
        figure_key=figure_spec.key,
        intended_output_name=figure_spec.output_name,
        profile=figure_spec.profile,
        canvas_name=figure_spec.canvas.name,
        canvas_size_inches=figure_spec.canvas.size_inches,
        horizon_months=data.horizon_months,
        evaluation_protocol=data.evaluation_protocol,
        target_method=data.target_method,
        comparator_method=data.comparator_method,
        relative_cutoff_day=data.relative_cutoff_day,
        layout="four_by_two_trajectory_and_distribution",
        delta_limit_abs=delta_limit,
        distribution_delta_limit_abs=distribution_delta_limit,
        delta_ticks=delta_ticks,
        histogram_bin_width=HISTOGRAM_BIN_WIDTH,
        histogram_frequency_limit=histogram_frequency_limit,
        cutoffs=data.cutoffs,
        panels=tuple(panel_audits),
        saved_path=saved_path,
    )
    return EmdRenderResult(
        figure=figure,
        trajectory_axes=trajectory_axes,
        distribution_axes=distribution_axes,
        audit=audit,
    )
