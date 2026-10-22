"""Exact, provenance-aware pairwise comparisons of canonical EMD results."""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class EmdComparisonError(ValueError):
    """Raised when canonical EMD data cannot support a requested comparison."""


SUBTYPE_INITIAL_INFERENCE_DATES: Mapping[str, str] = MappingProxyType(
    {
        # Results_EMD start_date values are the only day-specific inference
        # origins in the archived inputs. Older full-grid Obs_Time values imply
        # Jan 1 / May 1 origins instead, so callers can override these defaults
        # when reproducing that legacy convention deliberately.
        "h3n2": "2009-01-02",
        "h1n1": "2009-05-02",
    }
)

# The first canonical EMD release expanded MLR incorrectly. Keep this one
# input-only alias so notebooks and scripts written against that brief release
# fail gradually rather than selecting the wrong comparator. Canonical tables,
# returned comparison metadata, recipes, and documentation always use the
# scientifically correct identifier.
DEPRECATED_METHOD_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "multilinear_regression_rare_filtered": (
            "multinomial_logistic_regression_rare_filtered"
        )
    }
)


def _normalize_method_id(method_id: str) -> str:
    canonical = DEPRECATED_METHOD_ALIASES.get(method_id)
    if canonical is None:
        return method_id
    warnings.warn(
        f"EMD method_id {method_id!r} is deprecated; use {canonical!r}. MLR "
        "means multinomial logistic regression.",
        DeprecationWarning,
        stacklevel=3,
    )
    return canonical


@dataclass(frozen=True, slots=True)
class EmdCutoff:
    """One inclusive, subtype-specific EMD observation-date cutoff."""

    subtype: str
    initial_inference_date: str
    relative_day: int
    cutoff_date: str
    inclusive: bool
    source: str

    def describe(self) -> str:
        """Return a concise printable description of this cutoff."""

        relation = "on or after" if self.inclusive else "after"
        return (
            f"{self.subtype.upper()}: {relation} {self.cutoff_date} "
            f"(initial inference {self.initial_inference_date} + "
            f"{self.relative_day} days; {self.source})"
        )


@dataclass(frozen=True, slots=True)
class EmdComparison:
    target_method: str
    comparator_method: str
    result_version: str
    protocols: tuple[str, ...]
    target_count: int
    comparator_count: int
    paired_count: int
    unmatched_target_ids: tuple[str, ...]
    unmatched_comparator_ids: tuple[str, ...]
    cutoffs: tuple[EmdCutoff, ...]
    rows: tuple[Mapping[str, object], ...]

    @property
    def deltas(self) -> tuple[float, ...]:
        """Comparator EMD minus target EMD; positive favors the target."""

        return tuple(float(row["delta_emd"]) for row in self.rows)

    def describe_cutoff(self) -> str:
        """Return the comparison's observation-date cutoff in printable form."""

        if not self.cutoffs:
            return "No observation-date cutoff applied."
        return "Inclusive EMD cutoff — " + "; ".join(
            cutoff.describe() for cutoff in self.cutoffs
        )


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise EmdComparisonError(f"Missing canonical EMD table: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_date(value: str | date, *, context: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise EmdComparisonError(f"Invalid ISO date for {context}: {value!r}") from error


def _normalize_initial_inference_dates(
    values: Mapping[str, str | date] | None,
) -> dict[str, date]:
    normalized: dict[str, date] = {}
    for raw_subtype, raw_date in (values or {}).items():
        subtype = raw_subtype.lower()
        if subtype in normalized:
            raise EmdComparisonError(
                f"Duplicate initial-inference-date override for subtype {subtype}"
            )
        normalized[subtype] = _parse_date(
            raw_date,
            context=f"initial inference date for {subtype}",
        )
    return normalized


def _cutoff_for_evaluation(
    evaluation: Mapping[str, str],
    *,
    relative_day: int,
    overrides: Mapping[str, date],
) -> tuple[date, EmdCutoff]:
    subtype = evaluation["subtype"].lower()
    if subtype in overrides:
        initial_date = overrides[subtype]
        source = "caller override"
    elif evaluation["model_start_date"]:
        initial_date = _parse_date(
            evaluation["model_start_date"],
            context=f"model_start_date for {evaluation['evaluation_id']}",
        )
        source = "evaluation model_start_date"
    elif subtype in SUBTYPE_INITIAL_INFERENCE_DATES:
        initial_date = _parse_date(
            SUBTYPE_INITIAL_INFERENCE_DATES[subtype],
            context=f"repository default for {subtype}",
        )
        source = "repository subtype default"
    else:
        raise EmdComparisonError(
            "Cannot apply an inference-relative cutoff to "
            f"{evaluation['evaluation_id']}: no initial inference date is "
            f"available for subtype {subtype!r}"
        )

    cutoff_date = initial_date + timedelta(days=relative_day)
    return cutoff_date, EmdCutoff(
        subtype=subtype,
        initial_inference_date=initial_date.isoformat(),
        relative_day=relative_day,
        cutoff_date=cutoff_date.isoformat(),
        inclusive=True,
        source=source,
    )


def compare_emd(
    emd_dir: Path | str,
    *,
    target_method: str,
    comparator_method: str,
    evaluation_protocol: str,
    result_version: str = "current",
    horizon_months: int | None = None,
    subtype: str | None = None,
    segment: str | None = None,
    minimum_observation_day: int | None = None,
    initial_inference_dates: Mapping[str, str | date] | None = None,
) -> EmdComparison:
    """Pair two methods using exact canonical evaluation identifiers.

    No date approximation is performed. Unmatched identifiers are returned
    explicitly. ``evaluation_protocol`` is mandatory so summary statistics can
    never silently mix distinct grids.

    When ``minimum_observation_day`` is supplied, it is interpreted as an
    inclusive number of days after the subtype's initial inference date, not as
    a day relative to a universal calendar origin. Per-evaluation
    ``model_start_date`` values are preferred over repository subtype defaults;
    ``initial_inference_dates`` can override either source explicitly. The
    comparison result records every concrete cutoff date used.
    """

    target_method = _normalize_method_id(target_method)
    comparator_method = _normalize_method_id(comparator_method)

    root = Path(emd_dir)
    methods = {row["method_id"] for row in _read(root / "methods.csv")}
    for method in (target_method, comparator_method):
        if method not in methods:
            raise EmdComparisonError(f"Unknown EMD method: {method}")
    if target_method == comparator_method:
        raise EmdComparisonError("Target and comparator methods must differ")

    evaluations = {
        row["evaluation_id"]: row for row in _read(root / "evaluations.csv")
    }
    available_protocols = {row["evaluation_protocol"] for row in evaluations.values()}
    if not evaluation_protocol:
        raise EmdComparisonError("evaluation_protocol must be specified explicitly")
    if evaluation_protocol not in available_protocols:
        raise EmdComparisonError(
            f"Unknown EMD evaluation protocol: {evaluation_protocol}"
        )
    if minimum_observation_day is not None and minimum_observation_day < 0:
        raise EmdComparisonError("minimum_observation_day must be nonnegative")
    inference_date_overrides = _normalize_initial_inference_dates(
        initial_inference_dates
    )
    cutoff_records: dict[tuple[object, ...], EmdCutoff] = {}

    def selected(evaluation: Mapping[str, str]) -> bool:
        checks = (
            (evaluation_protocol, evaluation["evaluation_protocol"]),
            (
                None if horizon_months is None else str(horizon_months),
                evaluation["horizon_months"],
            ),
            (
                None if subtype is None else subtype.lower(),
                evaluation["subtype"].lower(),
            ),
            (
                None if segment is None else segment.lower(),
                evaluation["segment"].lower(),
            ),
        )
        if any(
            expected is not None and expected != actual
            for expected, actual in checks
        ):
            return False
        if minimum_observation_day is not None:
            cutoff_date, cutoff = _cutoff_for_evaluation(
                evaluation,
                relative_day=minimum_observation_day,
                overrides=inference_date_overrides,
            )
            cutoff_records[
                (
                    cutoff.subtype,
                    cutoff.initial_inference_date,
                    cutoff.relative_day,
                    cutoff.cutoff_date,
                    cutoff.source,
                )
            ] = cutoff
            observation_date = _parse_date(
                evaluation["observation_date"],
                context=f"observation_date for {evaluation['evaluation_id']}",
            )
            return observation_date >= cutoff_date
        return True

    selected_ids = {
        evaluation_id
        for evaluation_id, evaluation in evaluations.items()
        if selected(evaluation)
    }
    values: dict[str, dict[str, dict[str, str]]] = {
        target_method: {},
        comparator_method: {},
    }
    for row in _read(root / "values.csv"):
        method = row["method_id"]
        evaluation_id = row["evaluation_id"]
        if (
            method in values
            and row["result_version"] == result_version
            and evaluation_id in selected_ids
        ):
            if evaluation_id in values[method]:
                raise EmdComparisonError(
                    f"Duplicate {method}/{result_version} value for {evaluation_id}"
                )
            values[method][evaluation_id] = row

    target_ids = set(values[target_method])
    comparator_ids = set(values[comparator_method])
    paired_ids = target_ids & comparator_ids
    protocols = tuple(
        sorted(
            {
                evaluations[evaluation_id]["evaluation_protocol"]
                for evaluation_id in paired_ids
            }
        )
    )
    if any(protocol != evaluation_protocol for protocol in protocols):
        raise EmdComparisonError(
            "Internal protocol-selection error: paired rows escaped the explicit "
            f"{evaluation_protocol!r} protocol"
        )

    def sort_key(evaluation_id: str) -> tuple[object, ...]:
        item = evaluations[evaluation_id]
        return (
            item["evaluation_protocol"],
            int(item["horizon_months"]),
            item["subtype"],
            item["segment"],
            item["region"],
            item["model_start_date"],
            item["observation_date"],
            item["prediction_date"],
            evaluation_id,
        )

    rows: list[Mapping[str, object]] = []
    for evaluation_id in sorted(paired_ids, key=sort_key):
        evaluation = evaluations[evaluation_id]
        target = values[target_method][evaluation_id]
        comparator = values[comparator_method][evaluation_id]
        target_emd = float(target["emd"])
        comparator_emd = float(comparator["emd"])
        rows.append(
            MappingProxyType(
                {
                    **evaluation,
                    "target_method": target_method,
                    "target_emd": target_emd,
                    "target_source_id": target["source_id"],
                    "comparator_method": comparator_method,
                    "comparator_emd": comparator_emd,
                    "comparator_source_id": comparator["source_id"],
                    "delta_emd": comparator_emd - target_emd,
                    "result_version": result_version,
                }
            )
        )

    return EmdComparison(
        target_method=target_method,
        comparator_method=comparator_method,
        result_version=result_version,
        protocols=protocols,
        target_count=len(target_ids),
        comparator_count=len(comparator_ids),
        paired_count=len(paired_ids),
        unmatched_target_ids=tuple(sorted(target_ids - comparator_ids)),
        unmatched_comparator_ids=tuple(sorted(comparator_ids - target_ids)),
        cutoffs=tuple(cutoff_records[key] for key in sorted(cutoff_records)),
        rows=tuple(rows),
    )
