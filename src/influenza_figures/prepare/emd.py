"""Build canonical, provenance-preserving EMD plotting tables.

The legacy EMD inputs use several schemas and overload labels such as ``FM``.
This module validates the known relationships between those files and emits a
normalized evaluation table plus a long method/value table.  Legacy inputs are
read-only: this module writes only to the explicitly supplied output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Mapping, Sequence


CANONICAL_RELEASE = "emd_v6"
TRANSFORM_VERSION = "emd_migration_v6"
PROTOCOL_VERSION = "1"
DEFAULT_ARCHIVE_PREFIX = "archive/pre_plot_reorg_2026-08-26/original"
DEFAULT_CV_9M_ARCHIVE_PREFIX = "archive/emd_cv_2026-09-24/original"
DEFAULT_CV_9M_ROOT = (
    Path(__file__).resolve().parents[3]
    / DEFAULT_CV_9M_ARCHIVE_PREFIX
    / "tmp"
)
DEFAULT_PREVIOUS_CORRECTED_CV_9M_ROOT = (
    Path(__file__).resolve().parents[3]
    / "archive/emd_cv_2026-09-30/original/tmp"
)
DEFAULT_PREVIOUS_CORRECTED_CV_6_12_ROOT = (
    Path(__file__).resolve().parents[3]
    / "archive/emd_cv_2026-10-10/original/tmp"
)
DEFAULT_CORRECTED_CV_9M_ARCHIVE_PREFIX = "archive/emd_cv_2026-10-13/original"
DEFAULT_CORRECTED_CV_9M_ROOT = (
    Path(__file__).resolve().parents[3]
    / DEFAULT_CORRECTED_CV_9M_ARCHIVE_PREFIX
    / "tmp"
)
DEFAULT_CORRECTED_CV_6_12_ARCHIVE_PREFIX = "archive/emd_cv_2026-10-13/original"
DEFAULT_CORRECTED_CV_6_12_ROOT = (
    Path(__file__).resolve().parents[3]
    / DEFAULT_CORRECTED_CV_6_12_ARCHIVE_PREFIX
    / "tmp"
)
TIME_ORIGIN = date(2009, 1, 1)
RESULTS_12_TOLERANCE = Decimal("5e-8")

EVALUATION_FIELDS = (
    "evaluation_id",
    "evaluation_protocol",
    "protocol_version",
    "horizon_months",
    "horizon_days",
    "subtype",
    "segment",
    "region",
    "model_start_date",
    "observation_date",
    "prediction_date",
    "observation_day_2009",
    "prediction_day_2009",
)

VALUE_FIELDS = (
    "evaluation_id",
    "method_id",
    "emd",
    "result_version",
    "source_id",
    "source_row",
    "source_column",
    "transform_version",
    "selected_hyperparameters",
    "value_note",
)

METHOD_FIELDS = (
    "method_id",
    "model_family",
    "display_label",
    "linkage_policy",
    "rare_mutation_policy",
    "definition_status",
    "notes",
)

SOURCE_FIELDS = (
    "source_id",
    "source_family",
    "original_path",
    "archived_path",
    "sha256",
    "bytes",
    "rows",
    "schema_id",
    "upstream_version",
    "canonical_release",
    "transform_description",
)

FULL_HEADER = (
    "Obs_Time",
    "Predict_Time",
    "Obs_Date",
    "Predict_Date",
    "LB",
    "Null",
    "FM",
    "FM_unlinked",
    "FM_remove_rare_mut",
)

PLUS_MLR_HEADER = (
    "Obs_Time",
    "Predict_Time",
    "Obs_Date",
    "Predict_Date",
    "LB",
    "Null",
    "FM_remove_rare_mut",
    "MLR_remove_rare_mut",
)

UNION_HEADER = (
    "Obs_Time",
    "Predict_Time",
    "Obs_Date",
    "Predict_Date",
    "LB",
    "Null",
    "FM",
    "FM_unlinked",
    "FM_remove_rare_mut",
    "MLR_remove_rare_mut",
)

ALL_MODELS_HEADER = (
    "Obs_Time",
    "Predict_Time",
    "Obs_Date",
    "Predict_Date",
    "LB",
    "Null",
    "FM",
    "FMSS",
    "SLM",
    "SLFS",
)

RESULTS_12_HEADER = (
    "Obs_Time",
    "Predict_Time",
    "Obs_Date",
    "Predict_Date",
    "LB",
    "Null",
    "GenModel",
    "FMSS",
    "SLM",
    "SLFS",
)

RESULTS_GRID_HEADER = (
    "subtype",
    "protein",
    "region",
    "start_date",
    "obs_date",
    "pred_date",
    "horizon_days",
    "Lower Bound",
    "Null",
    "GenModel",
    "SLM",
    "MLR",
)

RESULTS_CV_HEADER = (
    "subtype",
    "protein",
    "region",
    "start_date",
    "obs_date",
    "pred_date",
    "horizon_days",
    "Lower Bound",
    "Null",
    "SLM",
    "MLR",
    "GenModel (baseline)",
    "best GenModel",
    "CV parameter",
)

LEGACY_HEADER = (
    "Obs_Time",
    "Predict_Time",
    "Obs_Date",
    "Predict_Date",
    "LB_Dist",
    "Null_Dist",
    "Medoid_Dist",
)

LEGACY_SINCE_2012_HEADER = (
    "Obs_Time",
    "Predict_Time",
    "Obs_Date",
    "Predict_Date",
    "LB",
    "Null",
    "FM",
    "FMSS",
    "SLM",
    "SLFS",
    "FM_unlinked",
    "FM_remove_rare_mut",
)

METHODS: tuple[dict[str, str], ...] = (
    {
        "method_id": "lower_bound",
        "model_family": "bound",
        "display_label": "Lower bound",
        "linkage_policy": "not_applicable",
        "rare_mutation_policy": "not_applicable",
        "definition_status": "confirmed",
        "notes": "Lower bound on EMD.",
    },
    {
        "method_id": "null",
        "model_family": "null",
        "display_label": "Null model",
        "linkage_policy": "not_applicable",
        "rare_mutation_policy": "not_applicable",
        "definition_status": "confirmed",
        "notes": "Neutral/null forecast comparator.",
    },
    {
        "method_id": "full_model_raw",
        "model_family": "full_model",
        "display_label": "Full model",
        "linkage_policy": "linked",
        "rare_mutation_policy": "included",
        "definition_status": "confirmed",
        "notes": "Unfiltered full-model result (FM in the FullModel family).",
    },
    {
        "method_id": "full_model_unlinked",
        "model_family": "full_model",
        "display_label": "Full model, unlinked",
        "linkage_policy": "unlinked",
        "rare_mutation_policy": "included",
        "definition_status": "confirmed",
        "notes": "Full-model diagnostic with sites treated as unlinked.",
    },
    {
        "method_id": "full_model_rare_filtered",
        "model_family": "full_model",
        "display_label": "Full model (baseline)",
        "linkage_policy": "linked",
        "rare_mutation_policy": "removed",
        "definition_status": "confirmed",
        "notes": (
            "Rare-mutation-filtered full model before cross-validation selection; "
            "GenModel in legacy Results files and GenModel (baseline) in the "
            "corrected cross-validation releases."
        ),
    },
    {
        "method_id": "full_model_cross_validated_rare_filtered",
        "model_family": "full_model",
        "display_label": "Full model (cross-validated)",
        "linkage_policy": "linked",
        "rare_mutation_policy": "removed",
        "definition_status": "confirmed",
        "notes": (
            "Rare-mutation-filtered full model with hyperparameters selected by "
            "cross-validation; best GenModel in the corrected releases."
        ),
    },
    {
        "method_id": "single_locus_model_rare_filtered",
        "model_family": "single_locus_model",
        "display_label": "Single-locus model (SLM)",
        "linkage_policy": "single_locus",
        "rare_mutation_policy": "removed",
        "definition_status": "confirmed",
        "notes": "SLM in the Results and AllModels families.",
    },
    {
        "method_id": "multinomial_logistic_regression_rare_filtered",
        "model_family": "multinomial_logistic_regression",
        "display_label": "Multinomial logistic regression (MLR)",
        "linkage_policy": "not_documented",
        "rare_mutation_policy": "removed",
        "definition_status": "confirmed",
        "notes": "MLR or MLR_remove_rare_mut; rare mutations are filtered.",
    },
    {
        "method_id": "full_model_single_site_simulation",
        "model_family": "full_model",
        "display_label": "Full model / single-site simulation (FMSS)",
        "linkage_policy": "not_documented",
        "rare_mutation_policy": "not_documented",
        "definition_status": "confirmed_legacy_diagnostic",
        "notes": "Legacy diagnostic: full model evaluated with single-site simulation.",
    },
    {
        "method_id": "single_locus_full_sites_simulation",
        "model_family": "single_locus_model",
        "display_label": "Single-locus model / full-sites simulation (SLFS)",
        "linkage_policy": "not_documented",
        "rare_mutation_policy": "not_documented",
        "definition_status": "confirmed_legacy_diagnostic",
        "notes": "Legacy diagnostic: single-locus model evaluated with full-sites simulation.",
    },
)

METHOD_ORDER = {row["method_id"]: index for index, row in enumerate(METHODS)}

RESULT_CODE_TO_IDENTITY = {
    "H1": ("h1n1", "ha"),
    "N1": ("h1n1", "na"),
    "H3": ("h3n2", "ha"),
    "N2": ("h3n2", "na"),
}

EXPECTED_FAMILY_COUNTS = {
    "full_model_full": 12,
    "full_model_plus_mlr": 12,
    "full_model_union_9m": 4,
    "all_models_filtered_12m": 4,
    "results_12m": 4,
    "results_6m": 4,
    "results_9m_pre_cv": 4,
    "results_9m_cross_validated_uncorrected": 4,
    "results_9m_cross_validated_corrected": 4,
    "results_6m_cross_validated_corrected": 4,
    "results_12m_cross_validated_corrected": 4,
    "legacy_full_model": 12,
    "legacy_since_2012": 4,
}

EXPECTED_OUTPUT_COUNTS = {
    "evaluations": 2591,
    "values": 16578,
    "methods": 10,
    "sources": 76,
}

class MigrationError(RuntimeError):
    """Raised when an input violates an audited EMD relationship."""


@dataclass(frozen=True)
class CsvTable:
    header: tuple[str, ...]
    rows: tuple[tuple[int, dict[str, str]], ...]


@dataclass(frozen=True)
class MigrationSummary:
    evaluations: int
    values: int
    methods: int
    sources: int
    output_sha256: Mapping[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "evaluations": self.evaluations,
            "values": self.values,
            "methods": self.methods,
            "sources": self.sources,
            "output_sha256": dict(sorted(self.output_sha256.items())),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_csv(path: Path) -> CsvTable:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        rows: list[tuple[int, dict[str, str]]] = []
        for source_row, row in enumerate(reader, start=2):
            if None in row:
                raise MigrationError(f"Malformed row {source_row} in {path}")
            normalized = {key: value if value is not None else "" for key, value in row.items()}
            if not any(value.strip() for value in normalized.values()):
                continue
            rows.append((source_row, normalized))
    return CsvTable(header=header, rows=tuple(rows))


def _require_header(path: Path, table: CsvTable, expected: Sequence[str]) -> None:
    if table.header != tuple(expected):
        raise MigrationError(
            f"Unexpected schema for {path}: {table.header!r}; expected {tuple(expected)!r}"
        )


def _parse_date(value: str) -> date:
    stripped = value.strip()
    for pattern in ("%Y-%m-%d", "%m/%d/%y", "%m/%d/%Y"):
        try:
            return datetime.strptime(stripped, pattern).date()
        except ValueError:
            pass
    raise MigrationError(f"Unsupported date value: {value!r}")


def _decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value.strip())
    except (InvalidOperation, AttributeError) as exc:
        raise MigrationError(f"Non-numeric EMD value: {value!r}") from exc
    if not parsed.is_finite():
        raise MigrationError(f"Non-finite EMD value: {value!r}")
    return parsed


def _canonical_number(value: str) -> str:
    parsed = _decimal(value)
    text = format(parsed, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _same_number(left: str, right: str, tolerance: Decimal = Decimal(0)) -> bool:
    return abs(_decimal(left) - _decimal(right)) <= tolerance


def _row_key(row: Mapping[str, str]) -> tuple[Decimal, Decimal, str, str]:
    return (
        _decimal(row["Obs_Time"]),
        _decimal(row["Predict_Time"]),
        _parse_date(row["Obs_Date"]).isoformat(),
        _parse_date(row["Predict_Date"]).isoformat(),
    )


def _index_rows(table: CsvTable) -> dict[tuple[Decimal, Decimal, str, str], tuple[int, dict[str, str]]]:
    indexed: dict[tuple[Decimal, Decimal, str, str], tuple[int, dict[str, str]]] = {}
    for source_row, row in table.rows:
        key = _row_key(row)
        if key in indexed:
            raise MigrationError(f"Duplicate EMD evaluation key {key!r}")
        indexed[key] = (source_row, row)
    return indexed


def _assert_projection(
    left_path: Path,
    left: CsvTable,
    right_path: Path,
    right: CsvTable,
    column_map: Mapping[str, str],
    *,
    require_equal_keys: bool = True,
    tolerance: Decimal = Decimal(0),
) -> None:
    left_rows = _index_rows(left)
    right_rows = _index_rows(right)
    if require_equal_keys and left_rows.keys() != right_rows.keys():
        raise MigrationError(f"Evaluation keys differ between {left_path} and {right_path}")
    missing = left_rows.keys() - right_rows.keys()
    if missing:
        raise MigrationError(f"{left_path} contains keys absent from {right_path}: {sorted(missing)[:3]}")
    for key, (left_line, left_row) in left_rows.items():
        right_line, right_row = right_rows[key]
        for left_column, right_column in column_map.items():
            left_value = left_row[left_column]
            right_value = right_row[right_column]
            if left_column.endswith("Date") or right_column.endswith("Date"):
                equal = _parse_date(left_value) == _parse_date(right_value)
            else:
                equal = _same_number(left_value, right_value, tolerance)
            if not equal:
                raise MigrationError(
                    f"Value mismatch {left_path}:{left_line}:{left_column}={left_value!r} "
                    f"vs {right_path}:{right_line}:{right_column}={right_value!r}"
                )


_FULL_RE = re.compile(
    r"^EMD_(?P<horizon>6|9|12)m_FullModel_(?P<subtype>H1N1|H3N2)_(?P<segment>HA|NA)_full\.csv$"
)
_PLUS_RE = re.compile(
    r"^EMD_(?P<horizon>6|9|12)m_FullModel_(?P<subtype>H1N1|H3N2)_(?P<segment>HA|NA)_full_plus_MLR\.csv$"
)
_UNION_RE = re.compile(
    r"^EMD_9m_FullModel_(?P<subtype>H1N1|H3N2)_(?P<segment>HA|NA)_full_dates_plus_MLR\.csv$"
)
_ALL_MODELS_RE = re.compile(
    r"^EMD_12m_AllModels_(?P<subtype>H1N1|H3N2)_(?P<segment>HA|NA)_filt\.csv$"
)
_RESULTS_RE = re.compile(r"^Results_EMD_(?P<horizon>6|9|12)m_(?P<code>H1|H3|N1|N2)\.csv$")
_CV_9M_RE = re.compile(
    r"^(?P<code>h1|h3|n1|n2)_9m_hyperparam_selection_summary\.csv$"
)
_CORRECTED_CV_RE = re.compile(
    r"^(?P<code>h1|h3|n1|n2)_(?P<horizon>6|9|12)m_"
    r"hyperparam_selection_summary_corrected\.csv$"
)
_CV_PARAMETER_RE = re.compile(
    r"^gamma=(?P<gamma>\d+), f=(?P<frequency>1e-[234])(?: \(baseline\))?$"
)
_LEGACY_RE = re.compile(
    r"^EMD_(?P<horizon>6|9|12)m_FullModel_(?P<subtype>H1N1|H3N2)_(?P<segment>HA|NA)\.csv$"
)
_LEGACY_SINCE_RE = re.compile(
    r"^EMD_12m_FullModel_(?P<subtype>H1N1|H3N2)_(?P<segment>HA|NA)_since_2012\.csv$"
)


def _identity(match: re.Match[str]) -> tuple[int, str, str]:
    horizon = int(match.groupdict().get("horizon") or 9)
    return horizon, match.group("subtype").lower(), match.group("segment").lower()


def _classify_source(relative_path: str) -> tuple[str, str, tuple[str, ...], str, str]:
    path = Path(relative_path)
    name = path.name
    in_old = path.parent.as_posix() == "data/emd/old"

    if in_old and _LEGACY_SINCE_RE.fullmatch(name):
        return (
            "legacy_since_2012",
            "emd.legacy_since_2012.v1",
            LEGACY_SINCE_2012_HEADER,
            "legacy_since_2012",
            "Archive-only historical estimates; do not promote to current values.",
        )
    if in_old and _LEGACY_RE.fullmatch(name):
        return (
            "legacy_full_model",
            "emd.legacy_full_model.v1",
            LEGACY_HEADER,
            "legacy_current",
            "Validation-only exact projection of the current FullModel family.",
        )
    if _UNION_RE.fullmatch(name):
        return (
            "full_model_union_9m",
            "emd.full_union.v1",
            UNION_HEADER,
            "legacy_current",
            "Validation-only exact union of 9-month FullModel and MLR files.",
        )
    if _PLUS_RE.fullmatch(name):
        return (
            "full_model_plus_mlr",
            "emd.full_plus_mlr.v1",
            PLUS_MLR_HEADER,
            "legacy_current",
            "Authoritative rare-filtered MLR; shared values validate FullModel inputs.",
        )
    if _FULL_RE.fullmatch(name):
        return (
            "full_model_full",
            "emd.full.v1",
            FULL_HEADER,
            "legacy_current",
            "Authoritative full-grid evaluations and full-model values.",
        )
    if _ALL_MODELS_RE.fullmatch(name):
        return (
            "all_models_filtered_12m",
            "emd.all_models_filtered.v1",
            ALL_MODELS_HEADER,
            "legacy_current",
            "Authoritative 12-month SLM and legacy diagnostic values on the full grid.",
        )
    results_match = _RESULTS_RE.fullmatch(name)
    if results_match and results_match.group("horizon") == "12":
        return (
            "results_12m",
            "emd.results_12m.v1",
            RESULTS_12_HEADER,
            "legacy_current",
            "Validation-only rounded projection of 12-month AllModels values.",
        )
    if results_match and results_match.group("horizon") == "6":
        return (
            "results_6m",
            "emd.results_grid.v1",
            RESULTS_GRID_HEADER,
            "pre_cross_validation",
            (
                "Superseded 6-month comparison-grid release retained for "
                "coverage validation; it supplies no current canonical values."
            ),
        )
    if results_match and results_match.group("horizon") == "9":
        return (
            "results_9m_pre_cv",
            "emd.results_grid.v1",
            RESULTS_GRID_HEADER,
            "pre_cross_validation",
            (
                "Superseded 9-month comparison-grid release retained for provenance; "
                "it supplies no current canonical values."
            ),
        )
    raise MigrationError(f"Unrecognized EMD input path: {relative_path}")


def _evaluation_id(fields: Mapping[str, object]) -> str:
    natural_key = (
        str(fields["evaluation_protocol"]),
        str(fields["protocol_version"]),
        str(fields["horizon_months"]),
        str(fields["horizon_days"]),
        str(fields["subtype"]),
        str(fields["segment"]),
        str(fields["region"]),
        str(fields["model_start_date"]),
        str(fields["observation_date"]),
        str(fields["prediction_date"]),
    )
    digest = hashlib.sha256("\x1f".join(natural_key).encode("utf-8")).hexdigest()[:24]
    return f"eval_{digest}"


def _make_evaluation(
    *,
    protocol: str,
    horizon_months: int,
    subtype: str,
    segment: str,
    region: str,
    model_start_date: str,
    observation_date: str,
    prediction_date: str,
    horizon_days: int | None = None,
) -> dict[str, object]:
    observed = _parse_date(observation_date)
    predicted = _parse_date(prediction_date)
    calculated_horizon = (predicted - observed).days
    if horizon_days is not None and abs(horizon_days - calculated_horizon) > 2:
        raise MigrationError(
            f"horizon_days={horizon_days} disagrees with dates {observed} to {predicted} "
            f"({calculated_horizon} days)"
        )
    canonical_horizon = calculated_horizon if horizon_days is None else horizon_days
    fields: dict[str, object] = {
        "evaluation_protocol": protocol,
        "protocol_version": PROTOCOL_VERSION,
        "horizon_months": horizon_months,
        "horizon_days": canonical_horizon,
        "subtype": subtype,
        "segment": segment,
        "region": region,
        "model_start_date": _parse_date(model_start_date).isoformat() if model_start_date else "",
        "observation_date": observed.isoformat(),
        "prediction_date": predicted.isoformat(),
        "observation_day_2009": (observed - TIME_ORIGIN).days,
        "prediction_day_2009": (predicted - TIME_ORIGIN).days,
    }
    fields["evaluation_id"] = _evaluation_id(fields)
    return fields


def _add_evaluation(
    evaluations: dict[str, dict[str, object]], evaluation: dict[str, object]
) -> str:
    evaluation_id = str(evaluation["evaluation_id"])
    previous = evaluations.get(evaluation_id)
    if previous is not None and previous != evaluation:
        raise MigrationError(f"Evaluation ID collision: {evaluation_id}")
    evaluations[evaluation_id] = evaluation
    return evaluation_id


def _add_value(
    values: dict[tuple[str, str, str], dict[str, object]],
    *,
    evaluation_id: str,
    method_id: str,
    emd: str,
    source_id: str,
    source_row: int,
    source_column: str,
    selected_hyperparameters: str = "",
    value_note: str = "",
) -> None:
    if method_id not in METHOD_ORDER:
        raise MigrationError(f"Unknown method ID: {method_id}")
    key = (evaluation_id, method_id, "current")
    if key in values:
        raise MigrationError(f"Duplicate canonical EMD value: {key}")
    values[key] = {
        "evaluation_id": evaluation_id,
        "method_id": method_id,
        "emd": _canonical_number(emd),
        "result_version": "current",
        "source_id": source_id,
        "source_row": source_row,
        "source_column": source_column,
        "transform_version": TRANSFORM_VERSION,
        "selected_hyperparameters": selected_hyperparameters,
        "value_note": value_note,
    }


def _write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def migrate_emd(
    repo_root: Path | str,
    output_dir: Path | str,
    *,
    cv_9m_root: Path | str = DEFAULT_CV_9M_ROOT,
    corrected_cv_9m_root: Path | str = DEFAULT_CORRECTED_CV_9M_ROOT,
    corrected_cv_6_12_root: Path | str = DEFAULT_CORRECTED_CV_6_12_ROOT,
    previous_corrected_cv_9m_root: Path | str = (
        DEFAULT_PREVIOUS_CORRECTED_CV_9M_ROOT
    ),
    previous_corrected_cv_6_12_root: Path | str = (
        DEFAULT_PREVIOUS_CORRECTED_CV_6_12_ROOT
    ),
    archive_prefix: str = DEFAULT_ARCHIVE_PREFIX,
    cv_9m_archive_prefix: str = DEFAULT_CV_9M_ARCHIVE_PREFIX,
    corrected_cv_9m_archive_prefix: str = DEFAULT_CORRECTED_CV_9M_ARCHIVE_PREFIX,
    corrected_cv_6_12_archive_prefix: str = (
        DEFAULT_CORRECTED_CV_6_12_ARCHIVE_PREFIX
    ),
    overwrite: bool = False,
) -> MigrationSummary:
    """Validate archived EMD releases and write four canonical CSV tables."""

    root = Path(repo_root).resolve()
    cv_root = Path(cv_9m_root).resolve()
    corrected_cv_9m_root = Path(corrected_cv_9m_root).resolve()
    corrected_cv_6_12_root = Path(corrected_cv_6_12_root).resolve()
    previous_corrected_cv_9m_root = Path(
        previous_corrected_cv_9m_root
    ).resolve()
    previous_corrected_cv_6_12_root = Path(
        previous_corrected_cv_6_12_root
    ).resolve()
    destination = Path(output_dir).resolve()
    emd_root = root / "data" / "emd"
    if not emd_root.is_dir():
        raise MigrationError(f"Missing EMD input directory: {emd_root}")
    if not cv_root.is_dir():
        raise MigrationError(f"Missing cross-validated 9-month input directory: {cv_root}")
    if not corrected_cv_9m_root.is_dir():
        raise MigrationError(
            "Missing corrected cross-validated 9-month input directory: "
            f"{corrected_cv_9m_root}"
        )
    if not corrected_cv_6_12_root.is_dir():
        raise MigrationError(
            "Missing corrected cross-validated 6/12-month input directory: "
            f"{corrected_cv_6_12_root}"
        )
    if not previous_corrected_cv_9m_root.is_dir():
        raise MigrationError(
            "Missing previous corrected cross-validated 9-month input directory: "
            f"{previous_corrected_cv_9m_root}"
        )
    if not previous_corrected_cv_6_12_root.is_dir():
        raise MigrationError(
            "Missing previous corrected cross-validated 6/12-month input directory: "
            f"{previous_corrected_cv_6_12_root}"
        )

    input_paths = sorted(emd_root.rglob("*.csv"))
    if len(input_paths) != 60:
        raise MigrationError(f"Expected 60 EMD CSV inputs, found {len(input_paths)}")
    cv_input_paths = sorted(cv_root.glob("*.csv"))
    if len(cv_input_paths) != 4:
        raise MigrationError(
            f"Expected 4 cross-validated 9-month CSV inputs, found {len(cv_input_paths)}"
        )
    def corrected_paths(
        directory: Path,
        horizons: set[int],
        *,
        release_label: str,
    ) -> list[Path]:
        paths: list[Path] = []
        for path in sorted(directory.glob("*.csv")):
            match = _CORRECTED_CV_RE.fullmatch(path.name)
            if match is None:
                raise MigrationError(
                    f"Unrecognized {release_label} cross-validation input: {path}"
                )
            if int(match.group("horizon")) in horizons:
                paths.append(path)
        return paths

    corrected_cv_9m_input_paths = corrected_paths(
        corrected_cv_9m_root,
        {9},
        release_label="authoritative",
    )
    if len(corrected_cv_9m_input_paths) != 4:
        raise MigrationError(
            "Expected 4 corrected cross-validated 9-month CSV inputs, found "
            f"{len(corrected_cv_9m_input_paths)}"
        )
    corrected_cv_6_12_input_paths = corrected_paths(
        corrected_cv_6_12_root,
        {6, 12},
        release_label="authoritative",
    )
    if len(corrected_cv_6_12_input_paths) != 8:
        raise MigrationError(
            "Expected 8 corrected cross-validated 6/12-month CSV inputs, found "
            f"{len(corrected_cv_6_12_input_paths)}"
        )
    previous_corrected_cv_input_paths = corrected_paths(
        previous_corrected_cv_9m_root,
        {9},
        release_label="previous",
    ) + corrected_paths(
        previous_corrected_cv_6_12_root,
        {6, 12},
        release_label="previous",
    )
    if len(previous_corrected_cv_input_paths) != 12:
        raise MigrationError(
            "Expected 12 previous corrected cross-validation CSV inputs, found "
            f"{len(previous_corrected_cv_input_paths)}"
        )

    table_cache: dict[Path, CsvTable] = {}

    def table(path: Path) -> CsvTable:
        if path not in table_cache:
            table_cache[path] = _load_csv(path)
        return table_cache[path]

    family_paths: dict[str, list[Path]] = {name: [] for name in EXPECTED_FAMILY_COUNTS}
    source_rows: list[dict[str, object]] = []
    source_ids: dict[Path, str] = {}
    seen_source_ids: set[str] = set()

    for path in input_paths:
        relative = path.relative_to(root).as_posix()
        family, schema_id, expected_header, upstream_version, description = _classify_source(relative)
        loaded = table(path)
        _require_header(path, loaded, expected_header)
        family_paths[family].append(path)
        digest = _sha256(path)
        source_id = f"src_{digest[:16]}"
        if source_id in seen_source_ids:
            raise MigrationError(f"Source-ID collision for {path}")
        seen_source_ids.add(source_id)
        source_ids[path] = source_id
        source_rows.append(
            {
                "source_id": source_id,
                "source_family": family,
                "original_path": relative,
                "archived_path": f"{archive_prefix.rstrip('/')}/{relative}",
                "sha256": digest,
                "bytes": path.stat().st_size,
                "rows": len(loaded.rows),
                "schema_id": schema_id,
                "upstream_version": upstream_version,
                "canonical_release": CANONICAL_RELEASE,
                "transform_description": description,
            }
        )

    for path in cv_input_paths:
        match = _CV_9M_RE.fullmatch(path.name)
        if match is None:
            raise MigrationError(f"Unrecognized cross-validated 9-month input: {path}")
        loaded = table(path)
        _require_header(path, loaded, RESULTS_CV_HEADER)
        family = "results_9m_cross_validated_uncorrected"
        family_paths[family].append(path)
        digest = _sha256(path)
        source_id = f"src_{digest[:16]}"
        if source_id in seen_source_ids:
            raise MigrationError(f"Source-ID collision for {path}")
        seen_source_ids.add(source_id)
        source_ids[path] = source_id
        original_path = f"tmp/{path.name}"
        source_rows.append(
            {
                "source_id": source_id,
                "source_family": family,
                "original_path": original_path,
                "archived_path": (
                    f"{cv_9m_archive_prefix.rstrip('/')}/{original_path}"
                ),
                "sha256": digest,
                "bytes": path.stat().st_size,
                "rows": len(loaded.rows),
                "schema_id": "emd.results_cv_9m.v1",
                "upstream_version": "cross_validated_null_uncorrected",
                "canonical_release": CANONICAL_RELEASE,
                "transform_description": (
                    "Superseded 9-month cross-validation run with incorrect null "
                    "EMDs; retained for provenance and supplies no current values."
                ),
            }
        )

    for path in corrected_cv_9m_input_paths:
        match = _CORRECTED_CV_RE.fullmatch(path.name)
        if match is None or match.group("horizon") != "9":
            raise MigrationError(
                f"Unrecognized corrected cross-validated 9-month input: {path}"
            )
        loaded = table(path)
        _require_header(path, loaded, RESULTS_CV_HEADER)
        family = "results_9m_cross_validated_corrected"
        family_paths[family].append(path)
        digest = _sha256(path)
        source_id = f"src_{digest[:16]}"
        if source_id in seen_source_ids:
            raise MigrationError(f"Source-ID collision for {path}")
        seen_source_ids.add(source_id)
        source_ids[path] = source_id
        original_path = f"tmp/{path.name}"
        source_rows.append(
            {
                "source_id": source_id,
                "source_family": family,
                "original_path": original_path,
                "archived_path": (
                    f"{corrected_cv_9m_archive_prefix.rstrip('/')}/{original_path}"
                ),
                "sha256": digest,
                "bytes": path.stat().st_size,
                "rows": len(loaded.rows),
                "schema_id": "emd.results_cv_9m_corrected.v1",
                "upstream_version": (
                    "cross_validated_lower_bound_horizon_pairing_corrected"
                ),
                "canonical_release": CANONICAL_RELEASE,
                "transform_description": (
                    "Authoritative complete 9-month run after correcting the "
                    "lower-bound horizon pairing used for hyperparameter "
                    "selection; all row values retain same-file provenance."
                ),
            }
        )

    for path in corrected_cv_6_12_input_paths:
        match = _CORRECTED_CV_RE.fullmatch(path.name)
        if match is None or match.group("horizon") not in {"6", "12"}:
            raise MigrationError(
                f"Unrecognized corrected cross-validated 6/12-month input: {path}"
            )
        horizon = int(match.group("horizon"))
        loaded = table(path)
        _require_header(path, loaded, RESULTS_CV_HEADER)
        family = f"results_{horizon}m_cross_validated_corrected"
        family_paths[family].append(path)
        digest = _sha256(path)
        source_id = f"src_{digest[:16]}"
        if source_id in seen_source_ids:
            raise MigrationError(f"Source-ID collision for {path}")
        seen_source_ids.add(source_id)
        source_ids[path] = source_id
        original_path = f"tmp/{path.name}"
        source_rows.append(
            {
                "source_id": source_id,
                "source_family": family,
                "original_path": original_path,
                "archived_path": (
                    f"{corrected_cv_6_12_archive_prefix.rstrip('/')}/"
                    f"{original_path}"
                ),
                "sha256": digest,
                "bytes": path.stat().st_size,
                "rows": len(loaded.rows),
                "schema_id": f"emd.results_cv_{horizon}m_corrected.v1",
                "upstream_version": (
                    "cross_validated_lower_bound_horizon_pairing_corrected"
                ),
                "canonical_release": CANONICAL_RELEASE,
                "transform_description": (
                    f"Authoritative complete {horizon}-month run after correcting "
                    "the lower-bound horizon pairing used for hyperparameter "
                    "selection; all row values retain same-file provenance."
                ),
            }
        )

    actual_family_counts = {name: len(paths) for name, paths in family_paths.items()}
    if actual_family_counts != EXPECTED_FAMILY_COUNTS:
        raise MigrationError(
            f"Unexpected EMD family counts: {actual_family_counts}; "
            f"expected {EXPECTED_FAMILY_COUNTS}"
        )

    evaluations: dict[str, dict[str, object]] = {}
    values: dict[tuple[str, str, str], dict[str, object]] = {}
    full_tables: dict[tuple[int, str, str], Path] = {}
    full_evaluations: dict[tuple[int, str, str, str, str], str] = {}

    # The FullModel family owns full-grid evaluations and all non-MLR values.
    for path in sorted(family_paths["full_model_full"]):
        match = _FULL_RE.fullmatch(path.name)
        if match is None:
            raise MigrationError(f"Could not parse FullModel identity: {path}")
        horizon, subtype, segment = _identity(match)
        full_tables[(horizon, subtype, segment)] = path
        for source_row, row in table(path).rows:
            evaluation = _make_evaluation(
                protocol="full_grid",
                horizon_months=horizon,
                subtype=subtype,
                segment=segment,
                region="",
                model_start_date="",
                observation_date=row["Obs_Date"],
                prediction_date=row["Predict_Date"],
            )
            evaluation_id = _add_evaluation(evaluations, evaluation)
            natural = (
                horizon,
                subtype,
                segment,
                str(evaluation["observation_date"]),
                str(evaluation["prediction_date"]),
            )
            if natural in full_evaluations:
                raise MigrationError(f"Duplicate full-grid natural key: {natural}")
            full_evaluations[natural] = evaluation_id
            for source_column, method_id in (
                ("LB", "lower_bound"),
                ("Null", "null"),
                ("FM", "full_model_raw"),
                ("FM_unlinked", "full_model_unlinked"),
                ("FM_remove_rare_mut", "full_model_rare_filtered"),
            ):
                _add_value(
                    values,
                    evaluation_id=evaluation_id,
                    method_id=method_id,
                    emd=row[source_column],
                    source_id=source_ids[path],
                    source_row=source_row,
                    source_column=source_column,
                )

    # MLR files must agree exactly with their FullModel partner on shared values.
    plus_tables: dict[tuple[int, str, str], Path] = {}
    for path in sorted(family_paths["full_model_plus_mlr"]):
        match = _PLUS_RE.fullmatch(path.name)
        if match is None:
            raise MigrationError(f"Could not parse MLR identity: {path}")
        horizon, subtype, segment = _identity(match)
        full_path = full_tables[(horizon, subtype, segment)]
        plus_tables[(horizon, subtype, segment)] = path
        _assert_projection(
            path,
            table(path),
            full_path,
            table(full_path),
            {
                "Obs_Time": "Obs_Time",
                "Predict_Time": "Predict_Time",
                "Obs_Date": "Obs_Date",
                "Predict_Date": "Predict_Date",
                "LB": "LB",
                "Null": "Null",
                "FM_remove_rare_mut": "FM_remove_rare_mut",
            },
        )
        for source_row, row in table(path).rows:
            observation_date = _parse_date(row["Obs_Date"]).isoformat()
            prediction_date = _parse_date(row["Predict_Date"]).isoformat()
            evaluation_id = full_evaluations[
                (horizon, subtype, segment, observation_date, prediction_date)
            ]
            _add_value(
                values,
                evaluation_id=evaluation_id,
                method_id="multinomial_logistic_regression_rare_filtered",
                emd=row["MLR_remove_rare_mut"],
                source_id=source_ids[path],
                source_row=source_row,
                source_column="MLR_remove_rare_mut",
            )

    # The 9-month union files are redundant but useful exact validation inputs.
    for path in sorted(family_paths["full_model_union_9m"]):
        match = _UNION_RE.fullmatch(path.name)
        if match is None:
            raise MigrationError(f"Could not parse 9-month union identity: {path}")
        _, subtype, segment = _identity(match)
        full_path = full_tables[(9, subtype, segment)]
        plus_path = plus_tables[(9, subtype, segment)]
        _assert_projection(
            path,
            table(path),
            full_path,
            table(full_path),
            {column: column for column in FULL_HEADER},
        )
        _assert_projection(
            path,
            table(path),
            plus_path,
            table(plus_path),
            {column: column for column in PLUS_MLR_HEADER},
        )

    # AllModels owns the full-precision 12-month SLM and legacy diagnostics.
    all_models_tables: dict[tuple[str, str], Path] = {}
    for path in sorted(family_paths["all_models_filtered_12m"]):
        match = _ALL_MODELS_RE.fullmatch(path.name)
        if match is None:
            raise MigrationError(f"Could not parse AllModels identity: {path}")
        _, subtype, segment = _identity(match)
        all_models_tables[(subtype, segment)] = path
        full_path = full_tables[(12, subtype, segment)]
        _assert_projection(
            path,
            table(path),
            full_path,
            table(full_path),
            {
                "Obs_Time": "Obs_Time",
                "Predict_Time": "Predict_Time",
                "Obs_Date": "Obs_Date",
                "Predict_Date": "Predict_Date",
                "LB": "LB",
                "Null": "Null",
                "FM": "FM_remove_rare_mut",
            },
            require_equal_keys=False,
        )
        for source_row, row in table(path).rows:
            observation_date = _parse_date(row["Obs_Date"]).isoformat()
            prediction_date = _parse_date(row["Predict_Date"]).isoformat()
            evaluation_id = full_evaluations[
                (12, subtype, segment, observation_date, prediction_date)
            ]
            for source_column, method_id, note in (
                ("SLM", "single_locus_model_rare_filtered", ""),
                (
                    "FMSS",
                    "full_model_single_site_simulation",
                    "Legacy diagnostic method.",
                ),
                (
                    "SLFS",
                    "single_locus_full_sites_simulation",
                    "Legacy diagnostic method.",
                ),
            ):
                _add_value(
                    values,
                    evaluation_id=evaluation_id,
                    method_id=method_id,
                    emd=row[source_column],
                    source_id=source_ids[path],
                    source_row=source_row,
                    source_column=source_column,
                    value_note=note,
                )

    # Results 12m is a lower-precision copy and is validation-only.
    for path in sorted(family_paths["results_12m"]):
        match = _RESULTS_RE.fullmatch(path.name)
        if match is None:
            raise MigrationError(f"Could not parse Results identity: {path}")
        subtype, segment = RESULT_CODE_TO_IDENTITY[match.group("code")]
        all_models_path = all_models_tables[(subtype, segment)]
        _assert_projection(
            path,
            table(path),
            all_models_path,
            table(all_models_path),
            {
                "Obs_Time": "Obs_Time",
                "Predict_Time": "Predict_Time",
                "Obs_Date": "Obs_Date",
                "Predict_Date": "Predict_Date",
                "LB": "LB",
                "Null": "Null",
                "GenModel": "FM",
                "FMSS": "FMSS",
                "SLM": "SLM",
                "SLFS": "SLFS",
            },
            tolerance=RESULTS_12_TOLERANCE,
        )

    def results_grid_evaluation(
        path: Path,
        source_row: int,
        row: Mapping[str, str],
        *,
        horizon: int,
        subtype: str,
        segment: str,
    ) -> tuple[dict[str, object], str]:
        """Validate one Results row and return its canonical evaluation."""

        row_subtype = row["subtype"].strip().lower()
        if row_subtype != subtype:
            raise MigrationError(
                f"Subtype mismatch in {path}:{source_row}: {row_subtype!r} != {subtype!r}"
            )
        source_segment = row["protein"].strip().lower()
        identity_imputed = False
        if not source_segment and segment == "na":
            source_segment = "na"
            identity_imputed = True
        if source_segment != segment:
            raise MigrationError(
                f"Segment mismatch in {path}:{source_row}: {source_segment!r} != {segment!r}"
            )
        horizon_days = int(row["horizon_days"])
        expected_horizon_days = {6: 180, 9: 270, 12: 360}[horizon]
        if horizon_days != expected_horizon_days:
            raise MigrationError(
                f"Unexpected horizon in {path}:{source_row}: {horizon_days}"
            )
        evaluation = _make_evaluation(
            protocol="results_grid",
            horizon_months=horizon,
            horizon_days=horizon_days,
            subtype=subtype,
            segment=segment,
            region=row["region"].strip().lower(),
            model_start_date=row["start_date"],
            observation_date=row["obs_date"],
            prediction_date=row["pred_date"],
        )
        note = "Segment imputed as na from filename." if identity_imputed else ""
        return evaluation, note

    # Each corrected cross-validation release is complete and authoritative,
    # not a column patch. All six values on a row come from the same source so
    # baseline, selected model, and comparator provenance remain coherent.
    cv_evaluation_ids: dict[tuple[int, str, str], set[str]] = {}
    corrected_cv_rows: dict[
        tuple[int, str, str], dict[str, tuple[Path, int, Mapping[str, str]]]
    ] = {}
    corrected_cv_grids: dict[
        tuple[int, str, str], tuple[tuple[str, str, str, str, str], ...]
    ] = {}
    corrected_families = {
        6: "results_6m_cross_validated_corrected",
        9: "results_9m_cross_validated_corrected",
        12: "results_12m_cross_validated_corrected",
    }
    for horizon, family in corrected_families.items():
        for path in sorted(family_paths[family]):
            match = _CORRECTED_CV_RE.fullmatch(path.name)
            if match is None or int(match.group("horizon")) != horizon:
                raise MigrationError(
                    f"Could not parse corrected cross-validation identity: {path}"
                )
            subtype, segment = RESULT_CODE_TO_IDENTITY[
                match.group("code").upper()
            ]
            identity = (horizon, subtype, segment)
            if identity in cv_evaluation_ids:
                raise MigrationError(
                    f"Duplicate corrected cross-validation panel: {identity}"
                )
            loaded = table(path)
            if len(loaded.rows) != 97:
                raise MigrationError(
                    f"Expected 97 corrected cross-validation rows in {path}, "
                    f"found {len(loaded.rows)}"
                )
            panel_ids: set[str] = set()
            panel_rows: dict[str, tuple[Path, int, Mapping[str, str]]] = {}
            panel_grid: list[tuple[str, str, str, str, str]] = []
            cv_evaluation_ids[identity] = panel_ids
            corrected_cv_rows[identity] = panel_rows
            previous_observation_date: date | None = None
            for source_row, row in loaded.rows:
                evaluation, note = results_grid_evaluation(
                    path,
                    source_row,
                    row,
                    horizon=horizon,
                    subtype=subtype,
                    segment=segment,
                )
                if evaluation["region"] != "all":
                    raise MigrationError(
                        f"Unexpected region in {path}:{source_row}: "
                        f"{evaluation['region']!r}"
                    )
                observed = _parse_date(str(evaluation["observation_date"]))
                if (
                    previous_observation_date is not None
                    and observed <= previous_observation_date
                ):
                    raise MigrationError(
                        "Observation dates are not strictly increasing in "
                        f"{path}:{source_row}"
                    )
                previous_observation_date = observed
                panel_grid.append(
                    (
                        subtype,
                        segment,
                        str(evaluation["region"]),
                        str(evaluation["model_start_date"]),
                        str(evaluation["observation_date"]),
                    )
                )

                cv_parameter = row["CV parameter"].strip()
                if _CV_PARAMETER_RE.fullmatch(cv_parameter) is None:
                    raise MigrationError(
                        f"Unexpected CV parameter in {path}:{source_row}: "
                        f"{cv_parameter!r}"
                    )
                numeric_columns = (
                    "Lower Bound",
                    "Null",
                    "SLM",
                    "MLR",
                    "GenModel (baseline)",
                    "best GenModel",
                )
                numeric_values = {
                    column: _decimal(row[column]) for column in numeric_columns
                }
                if any(value < 0 for value in numeric_values.values()):
                    raise MigrationError(f"Negative EMD value in {path}:{source_row}")
                lower_bound = numeric_values["Lower Bound"]
                for column in numeric_columns[1:]:
                    if lower_bound > numeric_values[column]:
                        raise MigrationError(
                            "Lower Bound/model EMD inversion in "
                            f"{path}:{source_row}: Lower Bound="
                            f"{_canonical_number(row['Lower Bound'])!r} exceeds "
                            f"{column}={_canonical_number(row[column])!r}"
                        )

                evaluation_id = _add_evaluation(evaluations, evaluation)
                if evaluation_id in panel_ids:
                    raise MigrationError(
                        "Duplicate corrected cross-validation evaluation in "
                        f"{path}:{source_row}"
                    )
                panel_ids.add(evaluation_id)
                panel_rows[evaluation_id] = (path, source_row, row)
                for source_column, method_id in (
                    ("Lower Bound", "lower_bound"),
                    ("Null", "null"),
                    ("SLM", "single_locus_model_rare_filtered"),
                    ("MLR", "multinomial_logistic_regression_rare_filtered"),
                    ("GenModel (baseline)", "full_model_rare_filtered"),
                    (
                        "best GenModel",
                        "full_model_cross_validated_rare_filtered",
                    ),
                ):
                    selected_hyperparameters = (
                        cv_parameter if source_column == "best GenModel" else ""
                    )
                    value_note = note
                    if source_column == "Null":
                        if horizon == 9:
                            value_note = (
                                "Corrected null EMD; supersedes the 2026-09-24 "
                                "cross-validation release."
                            )
                        else:
                            value_note = (
                                "Corrected null EMD from the definitive "
                                "cross-validation release."
                            )
                    elif source_column == "GenModel (baseline)":
                        value_note = "Baseline full-model rerun."
                    elif source_column == "best GenModel":
                        value_note = (
                            "Primary full-model value selected by cross-validation "
                            "after correcting lower-bound horizon pairing."
                        )
                    _add_value(
                        values,
                        evaluation_id=evaluation_id,
                        method_id=method_id,
                        emd=row[source_column],
                        source_id=source_ids[path],
                        source_row=source_row,
                        source_column=source_column,
                        selected_hyperparameters=selected_hyperparameters,
                        value_note=value_note,
                    )
            corrected_cv_grids[identity] = tuple(panel_grid)

    expected_panels = {
        (horizon, subtype, segment)
        for horizon in corrected_families
        for subtype, segment in set(RESULT_CODE_TO_IDENTITY.values())
    }
    if set(corrected_cv_grids) != expected_panels:
        raise MigrationError(
            "Corrected cross-validation panel set differs from the expected "
            f"three-horizon grid: {sorted(corrected_cv_grids)!r}"
        )
    for subtype, segment in set(RESULT_CODE_TO_IDENTITY.values()):
        reference = corrected_cv_grids[(9, subtype, segment)]
        for horizon in (6, 12):
            if corrected_cv_grids[(horizon, subtype, segment)] != reference:
                raise MigrationError(
                    "Corrected cross-validation observation grids differ for "
                    f"{subtype}/{segment}: 9 months vs {horizon} months"
                )

    # The 2026-10-13 release corrects the lower-bound horizon pairing used for
    # cross-validation selection. Relative to the previous authoritative files,
    # evaluation metadata and every supplied EMD except the selected best model
    # must remain byte-identical. A selected-model change must be accompanied by
    # a changed CV-parameter label, and vice versa.
    previous_release_selection_changes = 0
    previous_release_panels: set[tuple[int, str, str]] = set()
    invariant_columns = tuple(
        column
        for column in RESULTS_CV_HEADER
        if column not in {"best GenModel", "CV parameter"}
    )
    for path in sorted(previous_corrected_cv_input_paths):
        match = _CORRECTED_CV_RE.fullmatch(path.name)
        if match is None:
            raise MigrationError(
                f"Could not parse previous corrected cross-validation identity: {path}"
            )
        horizon = int(match.group("horizon"))
        subtype, segment = RESULT_CODE_TO_IDENTITY[match.group("code").upper()]
        identity = (horizon, subtype, segment)
        if identity in previous_release_panels:
            raise MigrationError(
                f"Duplicate previous corrected cross-validation panel: {identity}"
            )
        previous_release_panels.add(identity)
        loaded = table(path)
        _require_header(path, loaded, RESULTS_CV_HEADER)
        if len(loaded.rows) != 97:
            raise MigrationError(
                f"Expected 97 previous corrected cross-validation rows in {path}, "
                f"found {len(loaded.rows)}"
            )
        current_rows = corrected_cv_rows[identity]
        observed_ids: set[str] = set()
        for source_row, row in loaded.rows:
            evaluation, _ = results_grid_evaluation(
                path,
                source_row,
                row,
                horizon=horizon,
                subtype=subtype,
                segment=segment,
            )
            evaluation_id = str(evaluation["evaluation_id"])
            if evaluation_id not in current_rows:
                raise MigrationError(
                    "Previous corrected cross-validation evaluation absent from "
                    f"the authoritative release: {path}:{source_row}"
                )
            observed_ids.add(evaluation_id)
            current_path, current_row_number, current_row = current_rows[
                evaluation_id
            ]
            for column in invariant_columns:
                if row[column] != current_row[column]:
                    raise MigrationError(
                        "Unexpected horizon-pairing correction change for "
                        f"{column}: {path}:{source_row} vs "
                        f"{current_path}:{current_row_number}"
                    )
            best_changed = not _same_number(
                row["best GenModel"], current_row["best GenModel"]
            )
            parameter_changed = row["CV parameter"] != current_row["CV parameter"]
            if best_changed != parameter_changed:
                raise MigrationError(
                    "Best GenModel and CV parameter did not change together in "
                    f"the horizon-pairing correction: {path}:{source_row} vs "
                    f"{current_path}:{current_row_number}"
                )
            if best_changed:
                previous_release_selection_changes += 1
        if observed_ids != set(current_rows):
            raise MigrationError(
                "Authoritative release has a different evaluation set from the "
                f"previous corrected release for {subtype}/{segment}/{horizon}m"
            )
    if previous_release_panels != expected_panels:
        raise MigrationError(
            "Previous corrected cross-validation panel set differs from the "
            f"expected three-horizon grid: {sorted(previous_release_panels)!r}"
        )
    if previous_release_selection_changes == 0:
        raise MigrationError(
            "Horizon-pairing correction does not change any selected GenModel values"
        )

    # Validate the superseded cross-validation release against the correction.
    # Evaluation metadata and every non-null result must be unchanged; only the
    # erroneous Null column is allowed to differ.
    corrected_null_differences = 0
    for path in sorted(family_paths["results_9m_cross_validated_uncorrected"]):
        match = _CV_9M_RE.fullmatch(path.name)
        if match is None:
            raise MigrationError(f"Could not parse cross-validation identity: {path}")
        subtype, segment = RESULT_CODE_TO_IDENTITY[match.group("code").upper()]
        corrected_rows = corrected_cv_rows[(9, subtype, segment)]
        observed_ids: set[str] = set()
        for source_row, row in table(path).rows:
            evaluation, _ = results_grid_evaluation(
                path,
                source_row,
                row,
                horizon=9,
                subtype=subtype,
                segment=segment,
            )
            evaluation_id = str(evaluation["evaluation_id"])
            if evaluation_id not in corrected_rows:
                raise MigrationError(
                    "Superseded cross-validation evaluation absent from corrected "
                    f"release: {path}:{source_row}"
                )
            observed_ids.add(evaluation_id)
            corrected_path, corrected_row_number, corrected_row = corrected_rows[
                evaluation_id
            ]
            for column in (
                "Lower Bound",
                "SLM",
                "MLR",
                "GenModel (baseline)",
                "best GenModel",
            ):
                if not _same_number(row[column], corrected_row[column]):
                    raise MigrationError(
                        f"Unexpected corrected value change for {column}: "
                        f"{path}:{source_row} vs "
                        f"{corrected_path}:{corrected_row_number}"
                    )
            if row["CV parameter"] != corrected_row["CV parameter"]:
                raise MigrationError(
                    "CV parameter changed in the null correction: "
                    f"{path}:{source_row} vs "
                    f"{corrected_path}:{corrected_row_number}"
                )
            if not _same_number(row["Null"], corrected_row["Null"]):
                corrected_null_differences += 1
        if observed_ids != set(corrected_rows):
            raise MigrationError(
                f"Corrected release has a different evaluation set for {subtype}/{segment}"
            )
    if corrected_null_differences == 0:
        raise MigrationError(
            "Corrected cross-validation release does not change any Null EMD values"
        )

    # Every pre-CV six- and nine-month evaluation must be represented in the
    # corresponding definitive cross-validation run. Values are intentionally
    # not compared: the reruns changed all numeric columns and the older sources
    # are provenance-only.
    for horizon, family in ((6, "results_6m"), (9, "results_9m_pre_cv")):
        for path in sorted(family_paths[family]):
            match = _RESULTS_RE.fullmatch(path.name)
            if match is None or int(match.group("horizon")) != horizon:
                raise MigrationError(f"Could not parse Results identity: {path}")
            subtype, segment = RESULT_CODE_TO_IDENTITY[match.group("code")]
            definitive_ids = cv_evaluation_ids[(horizon, subtype, segment)]
            for source_row, row in table(path).rows:
                evaluation, _ = results_grid_evaluation(
                    path,
                    source_row,
                    row,
                    horizon=horizon,
                    subtype=subtype,
                    segment=segment,
                )
                if str(evaluation["evaluation_id"]) not in definitive_ids:
                    raise MigrationError(
                        "Pre-CV evaluation absent from definitive release: "
                        f"{path}:{source_row}"
                    )

    # Legacy base files must remain exact renamed projections of modern full data.
    for path in sorted(family_paths["legacy_full_model"]):
        match = _LEGACY_RE.fullmatch(path.name)
        if match is None:
            raise MigrationError(f"Could not parse legacy identity: {path}")
        horizon, subtype, segment = _identity(match)
        full_path = full_tables[(horizon, subtype, segment)]
        _assert_projection(
            path,
            table(path),
            full_path,
            table(full_path),
            {
                "Obs_Time": "Obs_Time",
                "Predict_Time": "Predict_Time",
                "Obs_Date": "Obs_Date",
                "Predict_Date": "Predict_Date",
                "LB_Dist": "LB",
                "Null_Dist": "Null",
                "Medoid_Dist": "FM",
            },
        )

    # Historical since-2012 files are intentionally not compared for equality:
    # audited model values differ and remain archive-only.
    for path in family_paths["legacy_since_2012"]:
        _index_rows(table(path))

    evaluation_rows = sorted(
        evaluations.values(),
        key=lambda row: (
            row["evaluation_protocol"],
            row["horizon_months"],
            row["subtype"],
            row["segment"],
            row["region"],
            row["model_start_date"],
            row["observation_date"],
            row["prediction_date"],
            row["evaluation_id"],
        ),
    )
    evaluation_rank = {
        str(row["evaluation_id"]): index for index, row in enumerate(evaluation_rows)
    }
    value_rows = sorted(
        values.values(),
        key=lambda row: (
            evaluation_rank[str(row["evaluation_id"])],
            METHOD_ORDER[str(row["method_id"])],
            row["result_version"],
        ),
    )
    method_rows = list(METHODS)
    source_rows.sort(key=lambda row: str(row["original_path"]))

    actual_output_counts = {
        "evaluations": len(evaluation_rows),
        "values": len(value_rows),
        "methods": len(method_rows),
        "sources": len(source_rows),
    }
    if actual_output_counts != EXPECTED_OUTPUT_COUNTS:
        raise MigrationError(
            f"Unexpected canonical counts: {actual_output_counts}; "
            f"expected {EXPECTED_OUTPUT_COUNTS}"
        )

    output_specs = (
        ("evaluations.csv", EVALUATION_FIELDS, evaluation_rows),
        ("values.csv", VALUE_FIELDS, value_rows),
        ("methods.csv", METHOD_FIELDS, method_rows),
        ("sources.csv", SOURCE_FIELDS, source_rows),
    )
    destination.mkdir(parents=True, exist_ok=True)
    output_paths = [destination / name for name, _, _ in output_specs]
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise MigrationError(
            "Refusing to overwrite canonical outputs without overwrite=True: "
            + ", ".join(str(path) for path in existing)
        )
    for path, fields, rows in (
        (destination / name, fields, rows) for name, fields, rows in output_specs
    ):
        _write_csv(path, fields, rows)

    output_hashes = {path.name: _sha256(path) for path in output_paths}
    return MigrationSummary(
        evaluations=len(evaluation_rows),
        values=len(value_rows),
        methods=len(method_rows),
        sources=len(source_rows),
        output_sha256=output_hashes,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root containing data/emd (default: inferred from this module).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination for evaluations.csv, values.csv, methods.csv, and sources.csv.",
    )
    parser.add_argument(
        "--cv-9m-root",
        type=Path,
        default=DEFAULT_CV_9M_ROOT,
        help=(
            "Directory containing the four archived 9-month cross-validation "
            "summary CSVs."
        ),
    )
    parser.add_argument(
        "--corrected-cv-9m-root",
        type=Path,
        default=DEFAULT_CORRECTED_CV_9M_ROOT,
        help=(
            "Directory containing the authoritative 2026-10-13 corrected "
            "cross-validation summary CSVs; only its four 9-month files are used."
        ),
    )
    parser.add_argument(
        "--corrected-cv-6-12-root",
        type=Path,
        default=DEFAULT_CORRECTED_CV_6_12_ROOT,
        help=(
            "Directory containing the authoritative 2026-10-13 corrected "
            "cross-validation summary CSVs; only its eight 6- and 12-month "
            "files are used."
        ),
    )
    parser.add_argument(
        "--previous-corrected-cv-9m-root",
        type=Path,
        default=DEFAULT_PREVIOUS_CORRECTED_CV_9M_ROOT,
        help=(
            "Validation-only directory containing the previous four corrected "
            "9-month cross-validation summary CSVs."
        ),
    )
    parser.add_argument(
        "--previous-corrected-cv-6-12-root",
        type=Path,
        default=DEFAULT_PREVIOUS_CORRECTED_CV_6_12_ROOT,
        help=(
            "Validation-only directory containing the previous eight corrected "
            "6- and 12-month cross-validation summary CSVs."
        ),
    )
    parser.add_argument(
        "--archive-prefix",
        default=DEFAULT_ARCHIVE_PREFIX,
        help="Archive prefix recorded in sources.csv.",
    )
    parser.add_argument(
        "--cv-9m-archive-prefix",
        default=DEFAULT_CV_9M_ARCHIVE_PREFIX,
        help="Archive prefix recorded for the 9-month cross-validation sources.",
    )
    parser.add_argument(
        "--corrected-cv-9m-archive-prefix",
        default=DEFAULT_CORRECTED_CV_9M_ARCHIVE_PREFIX,
        help=(
            "Archive prefix recorded for the authoritative corrected 9-month "
            "cross-validation sources."
        ),
    )
    parser.add_argument(
        "--corrected-cv-6-12-archive-prefix",
        default=DEFAULT_CORRECTED_CV_6_12_ARCHIVE_PREFIX,
        help=(
            "Archive prefix recorded for the authoritative corrected 6- and "
            "12-month cross-validation sources."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing canonical outputs in the destination directory.",
    )
    args = parser.parse_args(argv)
    summary = migrate_emd(
        args.repo_root,
        args.output_dir,
        cv_9m_root=args.cv_9m_root,
        corrected_cv_9m_root=args.corrected_cv_9m_root,
        corrected_cv_6_12_root=args.corrected_cv_6_12_root,
        previous_corrected_cv_9m_root=args.previous_corrected_cv_9m_root,
        previous_corrected_cv_6_12_root=(
            args.previous_corrected_cv_6_12_root
        ),
        archive_prefix=args.archive_prefix,
        cv_9m_archive_prefix=args.cv_9m_archive_prefix,
        corrected_cv_9m_archive_prefix=args.corrected_cv_9m_archive_prefix,
        corrected_cv_6_12_archive_prefix=(
            args.corrected_cv_6_12_archive_prefix
        ),
        overwrite=args.overwrite,
    )
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
