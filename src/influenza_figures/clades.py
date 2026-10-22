"""Selection and normalization helpers for canonical clade frequencies.

The archived notebook intended to retain clades whose frequency summed to more
than one across a reference observation window.  Filtering must happen before
per-date normalization; otherwise excluded clades remain in the denominator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
    if error.name in {"numpy", "pandas"}:
        raise ModuleNotFoundError(
            "Canonical clade preparation requires the scientific Python "
            "dependencies; install the project environment before importing "
            "influenza_figures.clades.",
            name=error.name,
        ) from error
    raise


class CladePreparationError(ValueError):
    """Raised when clade data cannot be selected or normalized safely."""


@dataclass(frozen=True, slots=True)
class PreparedCladeFrequencies:
    """Filtered, normalized clade data plus an auditable selection record."""

    data: pd.DataFrame
    selected_clades: tuple[str, ...]
    reference_totals: Mapping[str, float]
    normalization_columns: tuple[str, ...]


def _required_columns(
    data: pd.DataFrame,
    columns: Sequence[str],
) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise CladePreparationError(
            f"Clade data are missing required columns: {', '.join(missing)}"
        )


def _validated_copy(
    data: pd.DataFrame,
    *,
    clade_column: str,
    frequency_column: str,
) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise CladePreparationError("Clade data must be a pandas DataFrame")
    _required_columns(data, (clade_column, frequency_column))

    result = data.copy(deep=True)
    if result[clade_column].isna().any():
        raise CladePreparationError(f"{clade_column!r} contains missing values")
    result[clade_column] = result[clade_column].astype(str).str.strip()
    if result[clade_column].eq("").any():
        raise CladePreparationError(f"{clade_column!r} contains empty values")

    try:
        frequencies = pd.to_numeric(result[frequency_column], errors="raise").astype(
            float
        )
    except (TypeError, ValueError) as error:
        raise CladePreparationError(
            f"{frequency_column!r} must contain numeric values"
        ) from error
    values = frequencies.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise CladePreparationError(
            f"{frequency_column!r} must contain only finite values"
        )
    if (values < 0).any():
        raise CladePreparationError(
            f"{frequency_column!r} must not contain negative values"
        )
    result[frequency_column] = frequencies
    return result


def select_clades_by_total_frequency(
    reference: pd.DataFrame,
    *,
    minimum_total_frequency: float = 1.0,
    clade_column: str = "clade",
    frequency_column: str = "frequency",
) -> tuple[str, ...]:
    """Return clades whose reference-window total is strictly above a cutoff.

    The strict comparison matches the archived notebook expression
    ``clade_frequency_sum > 1.0``.  The total is a sum over sampled timepoints,
    rather than a time-weighted integral, so callers should define the reference
    window and sampling grid explicitly before calling this function.
    """

    try:
        threshold = float(minimum_total_frequency)
    except (TypeError, ValueError) as error:
        raise CladePreparationError(
            "minimum_total_frequency must be a finite, non-negative number"
        ) from error
    if not math.isfinite(threshold) or threshold < 0:
        raise CladePreparationError(
            "minimum_total_frequency must be a finite, non-negative number"
        )

    data = _validated_copy(
        reference,
        clade_column=clade_column,
        frequency_column=frequency_column,
    )
    totals = data.groupby(clade_column, sort=True)[frequency_column].sum()
    return tuple(str(clade) for clade in totals[totals > threshold].index)


def filter_and_normalize_clade_frequencies(
    data: pd.DataFrame,
    *,
    selected_clades: Sequence[str],
    normalization_columns: Sequence[str] | None = None,
    clade_column: str = "clade",
    frequency_column: str = "frequency",
    normalization_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Return selected clades normalized within each canonical series and date.

    Rows are filtered before normalization, and the input frame is never
    modified.  By default, normalization groups use every available canonical
    identity column from ``subtype``, ``segment``, ``series``, and ``date``.
    """

    prepared = _validated_copy(
        data,
        clade_column=clade_column,
        frequency_column=frequency_column,
    )
    labels = tuple(dict.fromkeys(str(clade).strip() for clade in selected_clades))
    if not labels or any(not label for label in labels):
        raise CladePreparationError("selected_clades must contain non-empty labels")

    if normalization_columns is None:
        groups = tuple(
            column
            for column in ("subtype", "segment", "series", "date")
            if column in prepared.columns
        )
    else:
        groups = tuple(normalization_columns)
    if not groups:
        raise CladePreparationError("At least one normalization column is required")
    _required_columns(prepared, groups)
    if prepared.loc[:, list(groups)].isna().any().any():
        raise CladePreparationError("Normalization columns contain missing values")

    prepared = prepared.loc[prepared[clade_column].isin(labels)].copy()
    if prepared.empty:
        raise CladePreparationError("No rows match selected_clades")

    group_totals = prepared.groupby(list(groups), sort=False, dropna=False)[
        frequency_column
    ].transform("sum")
    if (group_totals <= 0).any() or not np.isfinite(
        group_totals.to_numpy(dtype=float)
    ).all():
        raise CladePreparationError(
            "Every selected clade group must have a positive finite frequency total"
        )
    prepared[frequency_column] = prepared[frequency_column] / group_totals

    normalized_sums = prepared.groupby(
        list(groups), sort=False, dropna=False
    )[frequency_column].sum()
    if not np.allclose(
        normalized_sums.to_numpy(dtype=float),
        1.0,
        rtol=0.0,
        atol=normalization_tolerance,
    ):
        raise CladePreparationError(
            "Normalized clade frequencies do not sum to one in every group"
        )
    return prepared.reset_index(drop=True)


def prepare_clade_frequencies(
    data: pd.DataFrame,
    *,
    reference_series: str = "observed_full",
    minimum_total_frequency: float = 1.0,
    reference_start_date: str | pd.Timestamp | None = None,
    reference_end_date: str | pd.Timestamp | None = None,
    normalization_columns: Sequence[str] | None = None,
    series_column: str = "series",
    date_column: str = "date",
    clade_column: str = "clade",
    frequency_column: str = "frequency",
    normalization_tolerance: float = 1e-12,
) -> PreparedCladeFrequencies:
    """Select, filter, and normalize one canonical long-form clade table.

    Selection totals are calculated only from ``reference_series`` and from the
    optional open date interval: rows equal to either supplied boundary are
    excluded, matching the archived notebook.  The selected clades are then
    applied to every series before normalization.
    """

    working = _validated_copy(
        data,
        clade_column=clade_column,
        frequency_column=frequency_column,
    )
    _required_columns(working, (series_column, date_column))

    identity_columns = [
        column for column in ("subtype", "segment") if column in working.columns
    ]
    if identity_columns and len(working[identity_columns].drop_duplicates()) != 1:
        raise CladePreparationError(
            "Clade selection requires one subtype/segment combination; "
            "filter the canonical table and call once per combination"
        )

    parsed_dates = pd.to_datetime(working[date_column], errors="coerce")
    if parsed_dates.isna().any():
        raise CladePreparationError(f"{date_column!r} contains invalid dates")

    reference_mask = working[series_column].eq(reference_series)
    if reference_start_date is not None:
        try:
            start = pd.Timestamp(reference_start_date)
        except (TypeError, ValueError) as error:
            raise CladePreparationError("reference_start_date is invalid") from error
        reference_mask &= parsed_dates > start
    if reference_end_date is not None:
        try:
            end = pd.Timestamp(reference_end_date)
        except (TypeError, ValueError) as error:
            raise CladePreparationError("reference_end_date is invalid") from error
        reference_mask &= parsed_dates < end

    reference = working.loc[reference_mask].copy()
    if reference.empty:
        raise CladePreparationError(
            f"No rows remain in reference series {reference_series!r} and date window"
        )
    reference_totals_series = reference.groupby(clade_column, sort=True)[
        frequency_column
    ].sum()
    selected = select_clades_by_total_frequency(
        reference,
        minimum_total_frequency=minimum_total_frequency,
        clade_column=clade_column,
        frequency_column=frequency_column,
    )
    if not selected:
        raise CladePreparationError(
            "No clades exceed the minimum total-frequency threshold"
        )

    normalized = filter_and_normalize_clade_frequencies(
        working,
        selected_clades=selected,
        normalization_columns=normalization_columns,
        clade_column=clade_column,
        frequency_column=frequency_column,
        normalization_tolerance=normalization_tolerance,
    )
    return PreparedCladeFrequencies(
        data=normalized,
        selected_clades=selected,
        reference_totals=MappingProxyType(
            {
                str(clade): float(total)
                for clade, total in reference_totals_series.items()
            }
        ),
        normalization_columns=(
            tuple(normalization_columns)
            if normalization_columns is not None
            else tuple(
                column
                for column in ("subtype", "segment", "series", "date")
                if column in working.columns
            )
        ),
    )


__all__ = [
    "CladePreparationError",
    "PreparedCladeFrequencies",
    "filter_and_normalize_clade_frequencies",
    "prepare_clade_frequencies",
    "select_clades_by_total_frequency",
]
