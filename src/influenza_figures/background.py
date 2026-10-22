"""Canonical mutation-level extraction for background-fitness plotting data.

The upstream simulation code contains 0.05/0.95 frequency rules for active,
lost, and fixed mutations.  The plotting inputs, however, already contain the
upstream ``outcome``, ``initial_time``, and ``final_time`` assignments.  This
module treats those source values as authoritative and never reconstructs an
event or outcome from an observation's frequency.

Canonical background tables contain repeated observations at different
``start_time`` values.  The manuscript figures count each biological mutation
once within a subtype/segment panel.  :func:`extract_background_mutations`
groups rows by ``(subtype, segment, position, state)``.  When a duration cutoff
is requested, observations are first restricted to those meeting the cutoff;
the earliest eligible ``start_time`` is then chosen deterministically.  This
keeps mutations that have a later sustained appearance without counting their
repeated observations as independent samples.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


MUTATION_KEY_FIELDS = (
    "subtype",
    "segment",
    "position",
    "state",
)
# Backward-compatible public name retained for notebooks and downstream code
# written during the first plotting migration.
EVENT_KEY_FIELDS = MUTATION_KEY_FIELDS

_REQUIRED_FIELDS = {
    "subtype",
    "segment",
    "source_variant",
    "position",
    "state",
    "start_time",
    "initial_time",
    "final_time",
    "outcome",
}

# The first alias in each tuple is the intended canonical name.  The other
# names support the first migration release, whose sparse background schema
# retained two legacy naming choices.  Conflicting populated aliases are an
# error rather than an implicit precedence rule.
_SCIENTIFIC_ALIASES = {
    "initial_background_fitness": (
        "initial_background_fitness",
        "initial_fitness",
    ),
    "initial_population_fitness": ("initial_population_fitness",),
    "final_background_fitness": (
        "final_background_fitness",
        "final_fitness",
    ),
    "background_fitness_change": (
        "background_fitness_change",
        "fitness_change",
    ),
    "selection_coefficient": (
        "selection_coefficient",
        "population_fitness",
    ),
}


class BackgroundEventError(ValueError):
    """Raised when canonical background data cannot define stable mutations."""


@dataclass(frozen=True, slots=True)
class BackgroundMutationExtraction:
    """Immutable mutation extraction plus an audit of collapsed/filtered rows."""

    source: Path
    minimum_duration_days: int | float | None
    input_row_count: int
    event_count_before_filter: int
    event_count: int
    collapsed_observation_count: int
    excluded_by_duration_count: int
    outcome_counts_before_filter: Mapping[str, int]
    outcome_counts: Mapping[str, int]
    events: tuple[Mapping[str, object], ...]

    @property
    def rows(self) -> tuple[Mapping[str, object], ...]:
        """Alias matching other canonical comparison/extraction result objects."""

        return self.events

    @property
    def mutation_count_before_filter(self) -> int:
        """Number of unique mutation identities before duration filtering."""

        return self.event_count_before_filter

    @property
    def mutation_count(self) -> int:
        """Number of mutation identities retained after duration filtering."""

        return self.event_count


# Compatibility alias for the public type name used by the first migration.
BackgroundEventExtraction = BackgroundMutationExtraction


@dataclass(frozen=True, slots=True)
class _Observation:
    subtype: str
    segment: str
    source_variant: str
    position: str
    state: str
    start_time: Decimal
    initial_time: Decimal
    final_time: Decimal
    outcome: str
    scientific_values: Mapping[str, Decimal | None]
    source_row: int

    @property
    def mutation_key(self) -> tuple[object, ...]:
        return (
            self.subtype,
            self.segment,
            self.position,
            self.state,
        )


def _required_text(row: Mapping[str, str], field: str, source_row: int) -> str:
    value = (row.get(field) or "").strip()
    if not value:
        raise BackgroundEventError(f"Missing {field!r} at source row {source_row}")
    return value


def _decimal(value: object, *, field: str, source_row: int) -> Decimal:
    text = str(value).strip()
    if not text:
        raise BackgroundEventError(f"Missing {field!r} at source row {source_row}")
    try:
        number = Decimal(text)
    except InvalidOperation as error:
        raise BackgroundEventError(
            f"Invalid {field!r} value {text!r} at source row {source_row}"
        ) from error
    if not number.is_finite():
        raise BackgroundEventError(
            f"Non-finite {field!r} value {text!r} at source row {source_row}"
        )
    return number


def _optional_decimal(
    value: object, *, field: str, source_row: int
) -> Decimal | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    return _decimal(text, field=field, source_row=source_row)


def _coalesce_scientific_value(
    row: Mapping[str, str],
    *,
    canonical_field: str,
    aliases: tuple[str, ...],
    source_row: int,
) -> Decimal | None:
    populated: list[tuple[str, Decimal]] = []
    for alias in aliases:
        value = _optional_decimal(row.get(alias), field=alias, source_row=source_row)
        if value is not None:
            populated.append((alias, value))
    if not populated:
        return None

    distinct = {value for _, value in populated}
    if len(distinct) != 1:
        details = ", ".join(f"{name}={value}" for name, value in populated)
        raise BackgroundEventError(
            f"Conflicting aliases for {canonical_field!r} at source row "
            f"{source_row}: {details}"
        )
    return populated[0][1]


def _public_number(value: Decimal) -> int | float:
    integral = value.to_integral_value()
    if value == integral:
        return int(integral)
    return float(value)


def _public_optional(value: Decimal | None) -> float | None:
    return None if value is None else float(value)


def _identifier_sort_key(value: str) -> tuple[object, ...]:
    try:
        number = Decimal(value)
    except InvalidOperation:
        return (1, value)
    if not number.is_finite():
        return (1, value)
    return (0, number, value)


def _mutation_sort_key(key: tuple[object, ...]) -> tuple[object, ...]:
    subtype, segment, position, state = key
    return (
        subtype,
        segment,
        _identifier_sort_key(str(position)),
        _identifier_sort_key(str(state)),
    )


def _read_observations(path: Path) -> list[_Observation]:
    if not path.is_file():
        raise BackgroundEventError(f"Missing canonical background table: {path}")

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise BackgroundEventError(
                f"Canonical background table has no header: {path}"
            )
        fields = set(reader.fieldnames)
        missing = sorted(_REQUIRED_FIELDS - fields)
        if missing:
            raise BackgroundEventError(
                f"Canonical background table is missing fields {missing}: {path}"
            )
        missing_alias_groups = [
            canonical
            for canonical, aliases in _SCIENTIFIC_ALIASES.items()
            if not fields.intersection(aliases)
        ]
        if missing_alias_groups:
            raise BackgroundEventError(
                "Canonical background table is missing scientific fields "
                f"{missing_alias_groups}: {path}"
            )

        observations: list[_Observation] = []
        for source_row, row in enumerate(reader, start=2):
            initial_time = _decimal(
                row.get("initial_time"), field="initial_time", source_row=source_row
            )
            final_time = _decimal(
                row.get("final_time"), field="final_time", source_row=source_row
            )
            if final_time < initial_time:
                raise BackgroundEventError(
                    "final_time precedes initial_time at source row "
                    f"{source_row}: {final_time} < {initial_time}"
                )
            scientific_values = {
                canonical: _coalesce_scientific_value(
                    row,
                    canonical_field=canonical,
                    aliases=aliases,
                    source_row=source_row,
                )
                for canonical, aliases in _SCIENTIFIC_ALIASES.items()
            }
            observations.append(
                _Observation(
                    subtype=_required_text(row, "subtype", source_row).lower(),
                    segment=_required_text(row, "segment", source_row).lower(),
                    source_variant=_required_text(row, "source_variant", source_row),
                    position=_required_text(row, "position", source_row),
                    state=_required_text(row, "state", source_row),
                    start_time=_decimal(
                        row.get("start_time"), field="start_time", source_row=source_row
                    ),
                    initial_time=initial_time,
                    final_time=final_time,
                    outcome=_required_text(row, "outcome", source_row).lower(),
                    scientific_values=MappingProxyType(scientific_values),
                    source_row=source_row,
                )
            )
    return observations


def _representative_observation(
    mutation_key: tuple[object, ...], observations: list[_Observation]
) -> _Observation:
    """Return the earliest observation without relying on input row order.

    The archived files were ordered by ``StartTime`` and selected ``iloc[0]``
    for each mutation.  Selecting the minimum canonical ``start_time`` is the
    explicit equivalent.  Ties at that same earliest time are resolved using
    source event time, outcome, variant, and scientific values as stable
    tie-breakers.  The final source-row tie-break can only distinguish records
    that are identical for every plotted quantity.
    """

    def optional_decimal_key(value: Decimal | None) -> tuple[int, Decimal]:
        return (1, Decimal(0)) if value is None else (0, value)

    return min(
        observations,
        key=lambda item: (
            item.start_time,
            item.initial_time,
            item.final_time,
            item.outcome,
            item.source_variant,
            tuple(
                optional_decimal_key(item.scientific_values[field])
                for field in _SCIENTIFIC_ALIASES
            ),
            item.source_row,
        ),
    )


def _outcome_counts(events: list[Mapping[str, object]]) -> Mapping[str, int]:
    counts = Counter(str(event["outcome"]) for event in events)
    return MappingProxyType(dict(sorted(counts.items())))


def _minimum_duration(value: int | float | None) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BackgroundEventError(
            "minimum_duration_days must be a non-negative number"
        )
    if not math.isfinite(value) or value < 0:
        raise BackgroundEventError(
            "minimum_duration_days must be a non-negative number"
        )
    return Decimal(str(value))


def _observation_duration(observation: _Observation) -> Decimal:
    """Return the source-defined duration for one observation."""

    return observation.final_time - observation.initial_time


def _mutation_record(
    mutation_key: tuple[object, ...],
    mutation_observations: list[_Observation],
    eligible_observations: list[_Observation],
) -> Mapping[str, object]:
    """Build one mutation record from its earliest eligible observation.

    ``mutation_observations`` contains the full source history for audit
    fields.  ``eligible_observations`` contains the rows allowed to supply the
    plotted representative and must therefore be non-empty.
    """

    representative = _representative_observation(
        mutation_key,
        eligible_observations,
    )
    duration = _observation_duration(representative)
    start_times = tuple(
        _public_number(value)
        for value in sorted(item.start_time for item in mutation_observations)
    )
    eligible_start_times = tuple(
        _public_number(value)
        for value in sorted(item.start_time for item in eligible_observations)
    )
    source_variants = tuple(
        sorted({item.source_variant for item in mutation_observations})
    )
    return MappingProxyType(
        {
            "subtype": representative.subtype,
            "segment": representative.segment,
            "position": representative.position,
            "state": representative.state,
            "initial_time": _public_number(representative.initial_time),
            "final_time": _public_number(representative.final_time),
            "outcome": representative.outcome,
            "duration_days": _public_number(duration),
            "initial_background_fitness": _public_optional(
                representative.scientific_values["initial_background_fitness"]
            ),
            "initial_population_fitness": _public_optional(
                representative.scientific_values["initial_population_fitness"]
            ),
            "final_background_fitness": _public_optional(
                representative.scientific_values["final_background_fitness"]
            ),
            "background_fitness_change": _public_optional(
                representative.scientific_values["background_fitness_change"]
            ),
            "selection_coefficient": _public_optional(
                representative.scientific_values["selection_coefficient"]
            ),
            "observation_count": len(mutation_observations),
            "eligible_observation_count": len(eligible_observations),
            "representative_start_time": _public_number(
                representative.start_time
            ),
            "first_start_time": start_times[0],
            "last_start_time": start_times[-1],
            "start_times": start_times,
            "eligible_start_times": eligible_start_times,
            "source_variants": source_variants,
        }
    )


def extract_background_mutations(
    background_csv: Path | str,
    *,
    minimum_duration_days: int | float | None = None,
) -> BackgroundMutationExtraction:
    """Collapse observations to one row-order-independent record per mutation.

    The mutation key is ``(subtype, segment, position, state)``.  If
    ``minimum_duration_days`` is supplied, observations are first filtered
    inclusively to ``duration_days >=`` the requested minimum.  Within each
    mutation that still has an eligible observation, the earliest eligible
    ``start_time`` supplies the source-provided event times, outcome, and
    scientific values.  Those values are not recomputed.

    Observation-level frequency and prediction values are not used to choose
    the representative.  The full and eligible start-time histories remain in
    each returned record so the filter-first choice is auditable.
    """

    source = Path(background_csv)
    minimum = _minimum_duration(minimum_duration_days)
    observations = _read_observations(source)

    grouped: dict[tuple[object, ...], list[_Observation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.mutation_key].append(observation)

    all_mutations: list[Mapping[str, object]] = []
    retained: list[Mapping[str, object]] = []
    for mutation_key in sorted(grouped, key=_mutation_sort_key):
        mutation_observations = grouped[mutation_key]
        all_mutations.append(
            _mutation_record(
                mutation_key,
                mutation_observations,
                mutation_observations,
            )
        )

        eligible_observations = (
            mutation_observations
            if minimum is None
            else [
                observation
                for observation in mutation_observations
                if _observation_duration(observation) >= minimum
            ]
        )
        if not eligible_observations:
            continue
        retained.append(
            _mutation_record(
                mutation_key,
                mutation_observations,
                eligible_observations,
            )
        )

    return BackgroundMutationExtraction(
        source=source,
        minimum_duration_days=minimum_duration_days,
        input_row_count=len(observations),
        event_count_before_filter=len(all_mutations),
        event_count=len(retained),
        collapsed_observation_count=len(observations) - len(all_mutations),
        excluded_by_duration_count=len(all_mutations) - len(retained),
        outcome_counts_before_filter=_outcome_counts(all_mutations),
        outcome_counts=_outcome_counts(retained),
        events=tuple(retained),
    )


def extract_background_events(
    background_csv: Path | str,
    *,
    minimum_duration_days: int | float | None = None,
) -> BackgroundMutationExtraction:
    """Compatibility wrapper for :func:`extract_background_mutations`."""

    return extract_background_mutations(
        background_csv,
        minimum_duration_days=minimum_duration_days,
    )


__all__ = [
    "EVENT_KEY_FIELDS",
    "MUTATION_KEY_FIELDS",
    "BackgroundEventError",
    "BackgroundEventExtraction",
    "BackgroundMutationExtraction",
    "extract_background_events",
    "extract_background_mutations",
]
