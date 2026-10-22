"""Deterministic plotting series for canonical background-fitness mutations.

The legacy background plots grouped files in the visual order H3, N2, H1,
N1, selected the first observation row for each position/state pair, and added
random horizontal jitter.  This module performs only the scientific data
preparation: it consumes the validated mutation-level extraction, partitions
all retained mutations into the established panel order, and returns aligned
fixed/lost series.
Renderers can then choose deterministic display coordinates separately.

Fitness quantities remain in the fractional units stored in the canonical
tables.  The existing plots display those fractions with percentage-formatted
tick labels.  Annualization uses exactly 365 days, matching the legacy plots.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .background import (
    BackgroundMutationExtraction,
    extract_background_mutations,
)


OUTCOME_ORDER = ("fixed", "lost")


@dataclass(frozen=True, slots=True)
class BackgroundPanelSpec:
    """Identity and display label for one established background panel."""

    label: str
    subtype: str
    segment: str


PANEL_SPECS = (
    BackgroundPanelSpec(label="H3", subtype="h3n2", segment="ha"),
    BackgroundPanelSpec(label="N2", subtype="h3n2", segment="na"),
    BackgroundPanelSpec(label="H1", subtype="h1n1", segment="ha"),
    BackgroundPanelSpec(label="N1", subtype="h1n1", segment="na"),
)
PANEL_ORDER = tuple(spec.label for spec in PANEL_SPECS)


class BackgroundPlotDataError(ValueError):
    """Raised when canonical events cannot supply the background plot series."""


@dataclass(frozen=True, slots=True)
class BackgroundOutcomeSeries:
    """Aligned mutation records and scientific values for one outcome."""

    outcome: str
    events: tuple[Mapping[str, object], ...]
    selection_coefficient: tuple[float, ...]
    background_fitness_change: tuple[float, ...]
    annualized_background_fitness_change: tuple[float, ...]
    initial_background_minus_population_fitness: tuple[float, ...]

    @property
    def count(self) -> int:
        return len(self.events)


@dataclass(frozen=True, slots=True)
class BackgroundPanelData:
    """Fixed/lost plotting series for one subtype/segment panel."""

    label: str
    subtype: str
    segment: str
    fixed: BackgroundOutcomeSeries
    lost: BackgroundOutcomeSeries

    @property
    def event_count(self) -> int:
        return self.fixed.count + self.lost.count

    def outcome(self, name: str) -> BackgroundOutcomeSeries:
        """Return one outcome series without relying on tuple positions."""

        normalized = name.lower()
        if normalized not in OUTCOME_ORDER:
            raise KeyError(f"Unknown background outcome: {name!r}")
        return getattr(self, normalized)


@dataclass(frozen=True, slots=True)
class BackgroundPlotAudit:
    """Counts documenting mutation collapse, filtering, and partitioning."""

    input_row_count: int
    event_count_before_filter: int
    retained_event_count: int
    collapsed_observation_count: int
    excluded_by_duration_count: int
    outcome_counts: Mapping[str, int]
    panel_outcome_counts: Mapping[str, Mapping[str, int]]

    @property
    def mutation_count_before_filter(self) -> int:
        """Number of mutation identities before duration filtering."""

        return self.event_count_before_filter

    @property
    def retained_mutation_count(self) -> int:
        """Number of mutation identities retained for plotting."""

        return self.retained_event_count


@dataclass(frozen=True, slots=True)
class BackgroundPlotData:
    """Ordered canonical panel data plus an extraction/partition audit."""

    source: Path
    minimum_duration_days: int | float | None
    panels: tuple[BackgroundPanelData, ...]
    audit: BackgroundPlotAudit

    def panel(self, label: str) -> BackgroundPanelData:
        """Return a panel by its established display label."""

        normalized = label.upper()
        for panel in self.panels:
            if panel.label == normalized:
                return panel
        raise KeyError(f"Unknown background panel: {label!r}")


def _required_finite_value(
    event: Mapping[str, object],
    field: str,
) -> float:
    raw_value = event.get(field)
    if raw_value is None:
        raise BackgroundPlotDataError(
            f"Mutation {_mutation_identity(event)} is missing required {field!r}"
        )
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as error:
        raise BackgroundPlotDataError(
            f"Mutation {_mutation_identity(event)} has non-numeric {field!r}: "
            f"{raw_value!r}"
        ) from error
    if not math.isfinite(value):
        raise BackgroundPlotDataError(
            f"Mutation {_mutation_identity(event)} has non-finite {field!r}: "
            f"{raw_value!r}"
        )
    return value


def _mutation_identity(event: Mapping[str, object]) -> tuple[object, ...]:
    return (
        event.get("subtype"),
        event.get("segment"),
        event.get("position"),
        event.get("state"),
    )


def _outcome_series(
    outcome: str,
    events: tuple[Mapping[str, object], ...],
) -> BackgroundOutcomeSeries:
    selection: list[float] = []
    background_change: list[float] = []
    annualized_change: list[float] = []
    initial_difference: list[float] = []

    for event in events:
        duration = _required_finite_value(event, "duration_days")
        if duration <= 0:
            raise BackgroundPlotDataError(
                f"Mutation {_mutation_identity(event)} must have positive "
                "duration_days "
                "for annualization"
            )
        initial_background = _required_finite_value(
            event, "initial_background_fitness"
        )
        final_background = _required_finite_value(
            event, "final_background_fitness"
        )
        initial_population = _required_finite_value(
            event, "initial_population_fitness"
        )
        # The maintained series deliberately preserves the legacy plotting
        # formula ``w1_bg - w0_bg``.  The separately stored change field is a
        # rounded upstream result and can differ in its last decimal places.
        change = final_background - initial_background

        selection.append(_required_finite_value(event, "selection_coefficient"))
        background_change.append(change)
        annualized_change.append(365.0 * change / duration)
        initial_difference.append(initial_background - initial_population)

    return BackgroundOutcomeSeries(
        outcome=outcome,
        events=events,
        selection_coefficient=tuple(selection),
        background_fitness_change=tuple(background_change),
        annualized_background_fitness_change=tuple(annualized_change),
        initial_background_minus_population_fitness=tuple(initial_difference),
    )


def _partition_mutations(
    extraction: BackgroundMutationExtraction,
) -> tuple[BackgroundPanelData, ...]:
    specs_by_identity = {
        (spec.subtype, spec.segment): spec for spec in PANEL_SPECS
    }
    partitioned: dict[
        tuple[str, str], dict[str, list[Mapping[str, object]]]
    ] = {
        identity: {outcome: [] for outcome in OUTCOME_ORDER}
        for identity in specs_by_identity
    }

    for event in extraction.events:
        identity = (str(event["subtype"]).lower(), str(event["segment"]).lower())
        if identity not in specs_by_identity:
            raise BackgroundPlotDataError(
                f"Mutation {_mutation_identity(event)} does not belong to an "
                "established "
                f"background panel; expected {tuple(specs_by_identity)}"
            )
        outcome = str(event["outcome"]).lower()
        if outcome not in OUTCOME_ORDER:
            raise BackgroundPlotDataError(
                f"Mutation {_mutation_identity(event)} has unsupported outcome "
                f"{outcome!r}; "
                f"expected {OUTCOME_ORDER}"
            )
        partitioned[identity][outcome].append(event)

    panels: list[BackgroundPanelData] = []
    for spec in PANEL_SPECS:
        identity = (spec.subtype, spec.segment)
        series = {
            outcome: _outcome_series(
                outcome,
                tuple(partitioned[identity][outcome]),
            )
            for outcome in OUTCOME_ORDER
        }
        panels.append(
            BackgroundPanelData(
                label=spec.label,
                subtype=spec.subtype,
                segment=spec.segment,
                fixed=series["fixed"],
                lost=series["lost"],
            )
        )
    return tuple(panels)


def prepare_background_plot_data(
    background_csv: Path | str,
    *,
    minimum_duration_days: int | float | None,
) -> BackgroundPlotData:
    """Return deterministic fixed/lost background series in legacy panel order.

    ``minimum_duration_days`` is deliberately required so each plotting recipe
    records whether it uses the paper's 90-day restriction or no restriction.
    Representative selection and filtering are delegated to
    :func:`extract_background_mutations`.  Rows first pass the inclusive
    duration cutoff, and the earliest eligible ``start_time`` is then selected
    for each mutation.

    The returned series contain no display jitter.  Each tuple is aligned with
    its outcome's ``events`` tuple, whose order comes from the canonical
    mutation key rather than input row order.
    """

    extraction = extract_background_mutations(
        background_csv,
        minimum_duration_days=minimum_duration_days,
    )
    panels = _partition_mutations(extraction)
    panel_counts = MappingProxyType(
        {
            panel.label: MappingProxyType(
                {
                    "fixed": panel.fixed.count,
                    "lost": panel.lost.count,
                }
            )
            for panel in panels
        }
    )
    retained_count = sum(panel.event_count for panel in panels)
    if retained_count != extraction.event_count:
        raise BackgroundPlotDataError(
            "Background panel partition did not preserve every retained mutation: "
            f"partitioned={retained_count}, retained={extraction.event_count}"
        )

    return BackgroundPlotData(
        source=extraction.source,
        minimum_duration_days=minimum_duration_days,
        panels=panels,
        audit=BackgroundPlotAudit(
            input_row_count=extraction.input_row_count,
            event_count_before_filter=extraction.event_count_before_filter,
            retained_event_count=extraction.event_count,
            collapsed_observation_count=extraction.collapsed_observation_count,
            excluded_by_duration_count=extraction.excluded_by_duration_count,
            outcome_counts=extraction.outcome_counts,
            panel_outcome_counts=panel_counts,
        ),
    )


__all__ = [
    "OUTCOME_ORDER",
    "PANEL_ORDER",
    "PANEL_SPECS",
    "BackgroundOutcomeSeries",
    "BackgroundPanelData",
    "BackgroundPanelSpec",
    "BackgroundPlotAudit",
    "BackgroundPlotData",
    "BackgroundPlotDataError",
    "prepare_background_plot_data",
]
