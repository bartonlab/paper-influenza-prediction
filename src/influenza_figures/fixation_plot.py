"""Canonical manuscript figures for mutation fixation forecasts.

The original plotting functions accepted four temporary CSV files, one for
each protein.  This module reads the two canonical long tables directly and
applies the manuscript selectors explicitly.  Data collection is intentionally
dependency-light; Matplotlib is imported only when a figure is rendered.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from statistics import fmean
from typing import Iterable

from .registry import FigureSpec
from .render import figure_style_context


DEFAULT_MODEL_ID = "full_model"
DEFAULT_SOURCE_VARIANT = "all_time"
SLM_MODEL_ID = "slm"
SLM_SOURCE_VARIANT = "paper_date_filtered"
DEFAULT_MINIMUM_OBSERVATION_DAY = 1095
EXPECTED_FREQUENCIES = tuple(index / 10.0 for index in range(1, 10))


class FixationFigureDataError(ValueError):
    """Raised when a canonical fixation-figure selection is incomplete."""


@dataclass(frozen=True, slots=True)
class FixationPanelDefinition:
    label: str
    subtype: str
    segment: str


PANEL_DEFINITIONS = (
    FixationPanelDefinition("H3", "h3n2", "ha"),
    FixationPanelDefinition("N2", "h3n2", "na"),
    FixationPanelDefinition("H1", "h1n1", "ha"),
    FixationPanelDefinition("N1", "h1n1", "na"),
)


@dataclass(frozen=True, slots=True)
class FixationProbabilityPanel:
    definition: FixationPanelDefinition
    frequencies: tuple[float, ...]
    top_25_score: tuple[float, ...]
    top_score: tuple[float, ...]
    null: tuple[float, ...]
    bottom_score: tuple[float, ...]
    bottom_25_score: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FixationProbabilityData:
    source_path: str
    model_id: str
    minimum_observation_day: int
    panels: tuple[FixationProbabilityPanel, ...]


@dataclass(frozen=True, slots=True)
class AucPanel:
    definition: FixationPanelDefinition
    frequencies: tuple[float, ...]
    model: tuple[float, ...]
    null: tuple[float, ...]

    @property
    def differences(self) -> tuple[float, ...]:
        return tuple(
            model_value - null_value
            for model_value, null_value in zip(self.model, self.null, strict=True)
        )


@dataclass(frozen=True, slots=True)
class AucData:
    source_path: str
    model_id: str
    source_variant: str
    minimum_observation_day: int
    panels: tuple[AucPanel, ...]


@dataclass(frozen=True, slots=True)
class FixationPanelAudit:
    label: str
    subtype: str
    segment: str
    fixation_probability_row_count: int
    auc_row_count: int
    frequency_bins: tuple[float, ...]
    mean_model_auc: float
    mean_null_auc: float
    mean_delta_auc: float
    minimum_delta_auc: float
    maximum_delta_auc: float


@dataclass(frozen=True, slots=True)
class FixationFigureAudit:
    figure_key: str
    intended_output_name: str
    profile: str
    canvas_name: str
    canvas_size_inches: tuple[float, float]
    layout: str
    model_id: str
    source_variant: str
    minimum_observation_day: int
    input_paths: tuple[str, ...]
    panels: tuple[FixationPanelAudit, ...]
    saved_path: str | None


@dataclass(frozen=True, slots=True)
class FixationRenderResult:
    """Rendered objects, selected canonical data, and an immutable audit."""

    figure: object
    axes: tuple[object, ...]
    audit: FixationFigureAudit
    auc_data: AucData
    fixation_probability_data: FixationProbabilityData | None = None


_PFIX_REQUIRED_COLUMNS = frozenset(
    {
        "model_id",
        "subtype",
        "segment",
        "minimum_observation_day",
        "frequency",
        "top_score",
        "bottom_score",
        "top_25_score",
        "bottom_25_score",
        "null",
    }
)
_AUC_REQUIRED_COLUMNS = frozenset(
    {
        "model_id",
        "source_variant",
        "subtype",
        "segment",
        "minimum_observation_day",
        "frequency",
        "auc_model",
        "auc_null",
    }
)


def _read_csv(path: Path | str, required_columns: frozenset[str]) -> tuple[Path, list[dict[str, str]]]:
    source = Path(path).expanduser().resolve()
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = frozenset(reader.fieldnames or ())
            missing = sorted(required_columns - columns)
            if missing:
                raise FixationFigureDataError(
                    f"{source} is missing required columns: {', '.join(missing)}"
                )
            return source, [dict(row) for row in reader]
    except FileNotFoundError:
        raise


def _integer(value: str, *, column: str, source: Path, line_number: int) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise FixationFigureDataError(
            f"{source}:{line_number} has invalid {column!r}: {value!r}"
        ) from error
    if not isfinite(numeric) or not numeric.is_integer():
        raise FixationFigureDataError(
            f"{source}:{line_number} has non-integral {column!r}: {value!r}"
        )
    return int(numeric)


def _probability(value: str, *, column: str, source: Path, line_number: int) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise FixationFigureDataError(
            f"{source}:{line_number} has invalid {column!r}: {value!r}"
        ) from error
    if not isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise FixationFigureDataError(
            f"{source}:{line_number} has {column!r} outside [0, 1]: {value!r}"
        )
    return numeric


def _frequency(value: str, *, source: Path, line_number: int) -> float:
    return _probability(
        value,
        column="frequency",
        source=source,
        line_number=line_number,
    )


def _validate_frequency_grid(
    frequencies: tuple[float, ...],
    *,
    source: Path,
    panel: FixationPanelDefinition,
) -> None:
    if frequencies != EXPECTED_FREQUENCIES:
        raise FixationFigureDataError(
            f"{source} has an incomplete or unexpected frequency grid for "
            f"{panel.label}: {frequencies!r}; expected {EXPECTED_FREQUENCIES!r}"
        )


def _selected_rows(
    rows: Iterable[dict[str, str]],
    *,
    source: Path,
    panel: FixationPanelDefinition,
    model_id: str,
    minimum_observation_day: int,
    source_variant: str | None = None,
) -> list[tuple[int, dict[str, str]]]:
    selected: list[tuple[int, dict[str, str]]] = []
    for line_number, row in enumerate(rows, start=2):
        if row["model_id"].strip() != model_id:
            continue
        if row["subtype"].strip().lower() != panel.subtype:
            continue
        if row["segment"].strip().lower() != panel.segment:
            continue
        if source_variant is not None and row["source_variant"].strip() != source_variant:
            continue
        observation_day = _integer(
            row["minimum_observation_day"],
            column="minimum_observation_day",
            source=source,
            line_number=line_number,
        )
        if observation_day == minimum_observation_day:
            selected.append((line_number, row))
    return selected


def collect_fixation_probability_data(
    csv_path: Path | str,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    minimum_observation_day: int = DEFAULT_MINIMUM_OBSERVATION_DAY,
) -> FixationProbabilityData:
    """Select the four manuscript panels from the canonical long table."""

    source, rows = _read_csv(csv_path, _PFIX_REQUIRED_COLUMNS)
    panels: list[FixationProbabilityPanel] = []
    value_columns = (
        "top_25_score",
        "top_score",
        "null",
        "bottom_score",
        "bottom_25_score",
    )

    for definition in PANEL_DEFINITIONS:
        selected = _selected_rows(
            rows,
            source=source,
            panel=definition,
            model_id=model_id,
            minimum_observation_day=minimum_observation_day,
        )
        parsed: list[tuple[float, dict[str, float]]] = []
        for line_number, row in selected:
            frequency = _frequency(
                row["frequency"], source=source, line_number=line_number
            )
            values = {
                column: _probability(
                    row[column],
                    column=column,
                    source=source,
                    line_number=line_number,
                )
                for column in value_columns
            }
            parsed.append((frequency, values))
        parsed.sort(key=lambda item: item[0])
        frequencies = tuple(item[0] for item in parsed)
        if len(frequencies) != len(set(frequencies)):
            raise FixationFigureDataError(
                f"{source} has duplicate frequency rows for {definition.label}, "
                f"model_id={model_id!r}, minimum_observation_day="
                f"{minimum_observation_day}"
            )
        _validate_frequency_grid(frequencies, source=source, panel=definition)
        series = {
            column: tuple(item[1][column] for item in parsed)
            for column in value_columns
        }
        panels.append(
            FixationProbabilityPanel(
                definition=definition,
                frequencies=frequencies,
                top_25_score=series["top_25_score"],
                top_score=series["top_score"],
                null=series["null"],
                bottom_score=series["bottom_score"],
                bottom_25_score=series["bottom_25_score"],
            )
        )

    return FixationProbabilityData(
        source_path=str(source),
        model_id=model_id,
        minimum_observation_day=minimum_observation_day,
        panels=tuple(panels),
    )


def collect_auc_data(
    csv_path: Path | str,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    source_variant: str = DEFAULT_SOURCE_VARIANT,
    minimum_observation_day: int = DEFAULT_MINIMUM_OBSERVATION_DAY,
) -> AucData:
    """Select one unambiguous AUROC recipe from the canonical long table."""

    source, rows = _read_csv(csv_path, _AUC_REQUIRED_COLUMNS)
    panels: list[AucPanel] = []
    for definition in PANEL_DEFINITIONS:
        selected = _selected_rows(
            rows,
            source=source,
            panel=definition,
            model_id=model_id,
            source_variant=source_variant,
            minimum_observation_day=minimum_observation_day,
        )
        parsed: list[tuple[float, float, float]] = []
        for line_number, row in selected:
            parsed.append(
                (
                    _frequency(
                        row["frequency"], source=source, line_number=line_number
                    ),
                    _probability(
                        row["auc_model"],
                        column="auc_model",
                        source=source,
                        line_number=line_number,
                    ),
                    _probability(
                        row["auc_null"],
                        column="auc_null",
                        source=source,
                        line_number=line_number,
                    ),
                )
            )
        parsed.sort(key=lambda item: item[0])
        frequencies = tuple(item[0] for item in parsed)
        if len(frequencies) != len(set(frequencies)):
            raise FixationFigureDataError(
                f"{source} has duplicate frequency rows for {definition.label}, "
                f"model_id={model_id!r}, source_variant={source_variant!r}, "
                f"minimum_observation_day={minimum_observation_day}"
            )
        _validate_frequency_grid(frequencies, source=source, panel=definition)
        panels.append(
            AucPanel(
                definition=definition,
                frequencies=frequencies,
                model=tuple(item[1] for item in parsed),
                null=tuple(item[2] for item in parsed),
            )
        )

    return AucData(
        source_path=str(source),
        model_id=model_id,
        source_variant=source_variant,
        minimum_observation_day=minimum_observation_day,
        panels=tuple(panels),
    )


def _load_matplotlib() -> object:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Matplotlib is required to render fixation figures; install the "
            "project plotting environment first."
        ) from error
    return plt


def _adjusted_lower_limit(lower: float, upper: float, offset: float = 0.02) -> float:
    """Reproduce the small lower-axis gap applied by the archived mplot code."""

    return lower - ((upper - lower) / (1.0 - offset) - (upper - lower))


def _style_open_axis(
    axis: object,
    *,
    profile: object,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["bottom"].set_visible(True)
    axis.spines["left"].set_visible(True)
    axis.spines["bottom"].set_bounds(*x_bounds)
    axis.spines["left"].set_bounds(*y_bounds)
    for name in ("bottom", "left"):
        axis.spines[name].set_color(profile.palette.ink)
        axis.spines[name].set_linewidth(profile.legacy_figures.axwidth)
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
        length=profile.strokes.tick_length_pt - 1.25,
        width=profile.legacy_figures.tick_width,
    )
    for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        label.set_fontfamily(profile.typography.font_families[0])
        label.set_fontsize(profile.typography.tick_label_pt)
        label.set_color(profile.palette.ink)


def _save_figure(figure: object, output_path: Path | str | None, profile: object) -> str | None:
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


def _panel_audits(
    auc_data: AucData,
    fixation_probability_data: FixationProbabilityData | None,
) -> tuple[FixationPanelAudit, ...]:
    audits: list[FixationPanelAudit] = []
    for index, panel in enumerate(auc_data.panels):
        pfix_count = (
            0
            if fixation_probability_data is None
            else len(fixation_probability_data.panels[index].frequencies)
        )
        differences = panel.differences
        audits.append(
            FixationPanelAudit(
                label=panel.definition.label,
                subtype=panel.definition.subtype,
                segment=panel.definition.segment,
                fixation_probability_row_count=pfix_count,
                auc_row_count=len(panel.frequencies),
                frequency_bins=panel.frequencies,
                mean_model_auc=fmean(panel.model),
                mean_null_auc=fmean(panel.null),
                mean_delta_auc=fmean(differences),
                minimum_delta_auc=min(differences),
                maximum_delta_auc=max(differences),
            )
        )
    return tuple(audits)


def render_pfix_dauc_figure(
    fixation_probability_csv: Path | str,
    auc_csv: Path | str,
    *,
    figure_spec: FigureSpec,
    model_id: str = DEFAULT_MODEL_ID,
    source_variant: str = DEFAULT_SOURCE_VARIANT,
    minimum_observation_day: int = DEFAULT_MINIMUM_OBSERVATION_DAY,
    output_path: Path | str | None = None,
) -> FixationRenderResult:
    """Render Main Fig. 3: fixation curves and AUROC improvement.

    The canonical manuscript recipe uses the full model, the ``all_time``
    AUROC source, and a 1,095-day minimum observation time.
    """

    pfix_data = collect_fixation_probability_data(
        fixation_probability_csv,
        model_id=model_id,
        minimum_observation_day=minimum_observation_day,
    )
    auc_data = collect_auc_data(
        auc_csv,
        model_id=model_id,
        source_variant=source_variant,
        minimum_observation_day=minimum_observation_day,
    )
    for pfix_panel, auc_panel in zip(pfix_data.panels, auc_data.panels, strict=True):
        if pfix_panel.definition != auc_panel.definition:
            raise FixationFigureDataError("fixation and AUROC panel orders differ")
        if pfix_panel.frequencies != auc_panel.frequencies:
            raise FixationFigureDataError(
                f"fixation and AUROC frequency grids differ for {pfix_panel.definition.label}"
            )

    plt = _load_matplotlib()
    with figure_style_context(figure_spec) as (resolved_spec, profile):
        figure = plt.figure(figsize=resolved_spec.canvas.size_inches)
        figure.set_size_inches(resolved_spec.canvas.size_inches, forward=False)

        box_left, box_right = 0.11, 0.99
        box_top, box_bottom = 0.97, 0.15
        horizontal_gap, vertical_gap = 0.05, 0.06
        panel_width = (box_right - box_left - 3.0 * horizontal_gap) / 4.0
        panel_height = (
            resolved_spec.canvas.width_in / resolved_spec.canvas.height_in
        ) * panel_width

        fixation_axes: list[object] = []
        auc_axes: list[object] = []
        for index in range(4):
            left = box_left + index * (panel_width + horizontal_gap)
            fixation_axes.append(
                figure.add_axes(
                    [left, box_top - panel_height, panel_width, panel_height]
                )
            )
            auc_axes.append(
                figure.add_axes(
                    [
                        left,
                        box_bottom,
                        panel_width,
                        box_top - panel_height - vertical_gap - box_bottom,
                    ]
                )
            )

        font_family = profile.typography.font_families[0]
        curve_colors = (
            profile.palette.beneficial,
            profile.palette.beneficial_light,
            profile.palette.neutral,
            profile.palette.deleterious_light,
            profile.palette.deleterious,
        )
        curve_labels = (
            "top 25%",
            "top 50%",
            "avg (all)",
            "bottom 50%",
            "bottom 25%",
        )

        for index, (axis, panel) in enumerate(
            zip(fixation_axes, pfix_data.panels, strict=True)
        ):
            curves = (
                panel.top_25_score,
                panel.top_score,
                panel.null,
                panel.bottom_score,
                panel.bottom_25_score,
            )
            for color, values in zip(curve_colors, curves, strict=True):
                axis.plot(
                    panel.frequencies,
                    values,
                    color=color,
                    linewidth=1.2 * profile.legacy_figures.sizeline,
                    linestyle="-",
                    alpha=1.0,
                )
            axis.plot(
                (0.0, 1.0),
                (0.0, 1.0),
                color=profile.palette.ink,
                linewidth=profile.legacy_figures.sizeline,
                linestyle=":",
                alpha=0.3,
            )
            axis.set_xlim(_adjusted_lower_limit(0.0, 1.02), 1.02)
            axis.set_ylim(_adjusted_lower_limit(0.0, 1.02), 1.02)
            axis.set_xticks((0.0, 0.5, 1.0))
            axis.set_xticklabels(("", "", ""))
            axis.set_yticks((0.0, 0.5, 1.0))
            if index != 0:
                axis.set_yticklabels(("", "", ""))
            else:
                axis.set_ylabel(
                    "Fixation probability",
                    color=profile.palette.ink,
                    family=font_family,
                    fontsize=profile.typography.axis_label_pt,
                )
            _style_open_axis(
                axis,
                profile=profile,
                x_bounds=(0.0, 1.0),
                y_bounds=(0.0, 1.0),
            )

            protein_color = profile.palette.protein_colors[panel.definition.label]
            axis.text(
                0.070,
                -0.165,
                panel.definition.label,
                color=protein_color,
                family=font_family,
                fontsize=profile.typography.axis_label_pt,
                fontweight="bold",
                ha="center",
                va="center",
                clip_on=False,
            )
            if index == 1:
                for label_index, (label, color) in enumerate(
                    zip(curve_labels, curve_colors, strict=True)
                ):
                    axis.text(
                        0.67,
                        0.44 - 0.09 * label_index,
                        label,
                        color=color,
                        family=font_family,
                        fontsize=profile.typography.axis_label_pt,
                        fontweight="bold",
                        ha="left",
                        va="center",
                        clip_on=False,
                    )

        scatter_area = 2.0 * profile.legacy_figures.smallsizedot
        for index, (axis, panel) in enumerate(zip(auc_axes, auc_data.panels, strict=True)):
            color = profile.palette.protein_colors[panel.definition.label]
            axis.scatter(
                panel.frequencies,
                panel.differences,
                s=scatter_area,
                linewidths=0.0,
                marker="o",
                facecolors=color,
                edgecolors=color,
            )
            axis.axhline(
                0.0,
                color=profile.palette.ink,
                linewidth=2.0 * profile.legacy_figures.sizeline,
                linestyle=":",
                alpha=0.5,
            )
            axis.set_xlim(_adjusted_lower_limit(0.0, 1.02), 1.02)
            axis.set_ylim(_adjusted_lower_limit(-0.5, 0.5), 0.5)
            axis.set_xticks((0.0, 0.5, 1.0))
            axis.set_xticklabels(("0", "0.5", "1"))
            axis.set_yticks((-0.5, 0.0, 0.5))
            if index != 0:
                axis.set_yticklabels(("", "", ""))
            axis.set_xlabel(
                "Allele frequency",
                color=profile.palette.ink,
                family=font_family,
                fontsize=profile.typography.axis_label_pt,
            )
            _style_open_axis(
                axis,
                profile=profile,
                x_bounds=(0.0, 1.0),
                y_bounds=(-0.5, 0.5),
            )

        common_label = {
            "color": profile.palette.ink,
            "family": font_family,
            "fontsize": profile.typography.axis_label_pt,
            "ha": "center",
            "va": "center",
        }
        figure.text(
            box_left - 0.063,
            (box_top - panel_height - vertical_gap + box_bottom) / 2.0,
            "Improvement\nvs. neutral model\n" + r"$(\Delta \mathrm{AUROC})$",
            rotation=90,
            **common_label,
        )
        panel_label = {
            "color": "black",
            "family": font_family,
            "fontsize": profile.typography.panel_label_pt,
            "fontweight": "bold",
            "ha": "center",
            "va": "center",
            "clip_on": False,
        }
        figure.text(box_left - 0.095, box_top - 0.035, "A", **panel_label)
        figure.text(
            box_left - 0.095,
            box_top - panel_height - vertical_gap + 0.035,
            "B",
            **panel_label,
        )
        saved_path = _save_figure(figure, output_path, profile)

    audit = FixationFigureAudit(
        figure_key=figure_spec.key,
        intended_output_name=figure_spec.output_name,
        profile=figure_spec.profile,
        canvas_name=figure_spec.canvas.name,
        canvas_size_inches=figure_spec.canvas.size_inches,
        layout="four_columns_by_two_rows",
        model_id=model_id,
        source_variant=source_variant,
        minimum_observation_day=minimum_observation_day,
        input_paths=(pfix_data.source_path, auc_data.source_path),
        panels=_panel_audits(auc_data, pfix_data),
        saved_path=saved_path,
    )
    return FixationRenderResult(
        figure=figure,
        axes=tuple((*fixation_axes, *auc_axes)),
        audit=audit,
        auc_data=auc_data,
        fixation_probability_data=pfix_data,
    )


def render_fixation_auc_figure(
    auc_csv: Path | str,
    *,
    figure_spec: FigureSpec,
    model_id: str = DEFAULT_MODEL_ID,
    source_variant: str = DEFAULT_SOURCE_VARIANT,
    minimum_observation_day: int = DEFAULT_MINIMUM_OBSERVATION_DAY,
    model_display_label: str | None = None,
    output_path: Path | str | None = None,
) -> FixationRenderResult:
    """Render the four-panel AUROC plot used for Figs. S7 and S16.

    Use the defaults for the full-model figure (S7).  For the single-site
    figure (S16), pass ``model_id='slm'`` and
    ``source_variant='paper_date_filtered'``.  The human-readable panel label
    defaults to ``model`` for the full model and ``single site`` for SLM.
    """

    data = collect_auc_data(
        auc_csv,
        model_id=model_id,
        source_variant=source_variant,
        minimum_observation_day=minimum_observation_day,
    )
    if model_display_label is None:
        model_display_label = "single site" if model_id == SLM_MODEL_ID else "model"

    plt = _load_matplotlib()
    with figure_style_context(figure_spec) as (resolved_spec, profile):
        figure = plt.figure(figsize=resolved_spec.canvas.size_inches)
        figure.set_size_inches(resolved_spec.canvas.size_inches, forward=False)
        grid = figure.add_gridspec(
            2,
            2,
            left=0.14,
            right=0.97,
            top=0.95,
            bottom=0.08,
        )
        axes = tuple(
            figure.add_subplot(grid[row, column])
            for row in range(2)
            for column in range(2)
        )
        font_family = profile.typography.font_families[0]
        bar_width = 0.09
        edge_width = profile.legacy_figures.axwidth / 2.0

        for index, (axis, panel) in enumerate(zip(axes, data.panels, strict=True)):
            color = profile.palette.protein_colors[panel.definition.label]
            axis.bar(
                panel.frequencies,
                panel.model,
                width=bar_width,
                align="center",
                orientation="vertical",
                color=color,
                edgecolor=profile.palette.ink,
                linewidth=edge_width,
                alpha=0.75,
            )
            axis.bar(
                panel.frequencies,
                panel.null,
                width=bar_width,
                align="center",
                orientation="vertical",
                color=profile.palette.neutral,
                edgecolor=profile.palette.ink,
                linewidth=edge_width,
                alpha=0.75,
            )
            axis.set_xlim(_adjusted_lower_limit(0.0, 1.0), 1.0)
            axis.set_ylim(_adjusted_lower_limit(0.0, 1.0), 1.0)
            axis.set_xticks((0.0, 0.5, 1.0))
            axis.set_xticks(
                (0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9), minor=True
            )
            axis.set_yticks((0.0, 0.5, 1.0))
            axis.set_yticks((0.25, 0.75), minor=True)
            if index in (0, 1):
                axis.set_xticklabels(("", "", ""))
            else:
                axis.set_xticklabels(("0", "0.5", "1"))
            if index in (1, 3):
                axis.set_yticklabels(("", "", ""))
            else:
                axis.set_yticklabels(("0", "0.5", "1"))
            _style_open_axis(
                axis,
                profile=profile,
                x_bounds=(0.0, 1.0),
                y_bounds=(0.0, 1.0),
            )
            text_properties = {
                "family": font_family,
                "fontsize": profile.typography.axis_label_pt,
                "fontweight": "bold",
                "ha": "right",
                "va": "center",
                "clip_on": False,
            }
            axis.text(
                0.95,
                0.97,
                f"{panel.definition.label} {model_display_label}",
                color=color,
                **text_properties,
            )
            axis.text(
                0.95,
                0.90,
                "null",
                color=profile.palette.neutral,
                **text_properties,
            )

        common_label = {
            "color": profile.palette.ink,
            "family": font_family,
            "fontsize": profile.typography.axis_label_pt,
            "ha": "center",
            "va": "center",
        }
        figure.text(
            0.14 - 0.11,
            (0.95 + 0.08) / 2.0,
            "Accuracy of mutation fixation/loss prediction (AUROC)",
            rotation=90,
            **common_label,
        )
        figure.text(
            (0.14 + 0.97) / 2.0,
            0.08 - 0.06,
            "Mutation frequency",
            **common_label,
        )
        saved_path = _save_figure(figure, output_path, profile)

    audit = FixationFigureAudit(
        figure_key=figure_spec.key,
        intended_output_name=figure_spec.output_name,
        profile=figure_spec.profile,
        canvas_name=figure_spec.canvas.name,
        canvas_size_inches=figure_spec.canvas.size_inches,
        layout="two_by_two",
        model_id=model_id,
        source_variant=source_variant,
        minimum_observation_day=minimum_observation_day,
        input_paths=(data.source_path,),
        panels=_panel_audits(data, None),
        saved_path=saved_path,
    )
    return FixationRenderResult(
        figure=figure,
        axes=axes,
        audit=audit,
        auc_data=data,
    )

