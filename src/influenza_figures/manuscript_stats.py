"""Recompute manuscript-facing statistics without rendering figures.

Historically, several plotting functions printed scientific summaries while
they drew a figure.  That made the numbers easy to miss and difficult to reuse
when editing the manuscript.  This module keeps the same analysis choices
visible, gives every result a manuscript figure label, and returns ordinary
records that can be displayed in ``figures.ipynb`` or written to Markdown.

The source CSV files are read only.  No statistics are copied from the draft
manuscript and no plotting code is invoked.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from statistics import fmean, median
from typing import Iterable, Mapping, Sequence

from .background_plot_data import prepare_background_plot_data
from .emd_plot import MLR_METHOD, NULL_METHOD, SLM_METHOD, collect_emd_figure_data


PANEL_ORDER = (
    ("H3", "h3n2", "ha"),
    ("N2", "h3n2", "na"),
    ("H1", "h1n1", "ha"),
    ("N1", "h1n1", "na"),
)

EMD_MANUSCRIPT_RECIPES = (
    ("Figure 4", "null", NULL_METHOD, 9),
    ("Figure S8", "null", NULL_METHOD, 6),
    ("Figure S9", "null", NULL_METHOD, 12),
    ("Figure S10", "MLR", MLR_METHOD, 6),
    ("Figure S11", "MLR", MLR_METHOD, 9),
    ("Figure S12", "MLR", MLR_METHOD, 12),
    ("Figure S17", "SLM", SLM_METHOD, 6),
    ("Figure S18", "SLM", SLM_METHOD, 9),
    ("Figure S19", "SLM", SLM_METHOD, 12),
)


@dataclass(frozen=True, slots=True)
class ManuscriptStatistic:
    """One manuscript-facing result with its analysis and provenance."""

    figure_label: str
    panel: str
    family: str
    subtype: str | None
    segment: str | None
    comparison: str
    test: str
    n_1: int | None
    n_2: int | None
    estimate_name: str
    estimate: float | None
    statistic_name: str | None
    statistic: float | None
    p_value: float | None
    units: str
    data_release: str
    source: str
    notes: str = ""

    def as_record(self) -> dict[str, object]:
        """Return a stable, table-friendly representation."""

        return asdict(self)


def _dependencies() -> tuple[object, object]:
    try:
        import numpy as np
        from scipy import stats
    except ModuleNotFoundError as error:  # pragma: no cover - environment specific
        raise RuntimeError(
            "NumPy and SciPy are required to recompute manuscript statistics; "
            "install the plotting environment first."
        ) from error
    return np, stats


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _display_source(path: Path, plotting_data_dir: Path) -> str:
    try:
        return str(path.relative_to(plotting_data_dir.parent.parent))
    except ValueError:
        return str(path)


def collect_emd_statistics(plotting_data_dir: Path | str) -> tuple[ManuscriptStatistic, ...]:
    """Return the paired one-sample tests used by manuscript EMD figures."""

    _, stats = _dependencies()
    plotting_data = Path(plotting_data_dir)
    emd_dir = plotting_data / "inputs" / "emd"
    rows: list[ManuscriptStatistic] = []

    for figure_label, comparator_label, comparator_method, horizon in EMD_MANUSCRIPT_RECIPES:
        figure_data = collect_emd_figure_data(
            emd_dir,
            horizon_months=horizon,
            comparator_method=comparator_method,
        )
        cutoff_summary = "; ".join(figure_data.cutoff_report_lines)
        for panel in figure_data.panels:
            deltas = panel.deltas
            test = stats.ttest_1samp(deltas, 0.0, nan_policy="omit")
            rows.append(
                ManuscriptStatistic(
                    figure_label=figure_label,
                    panel=panel.definition.label,
                    family="EMD",
                    subtype=panel.definition.subtype,
                    segment=panel.definition.segment,
                    comparison=f"{comparator_label} EMD - full-model EMD",
                    test="paired one-sample t-test of delta EMD against zero",
                    n_1=len(deltas),
                    n_2=None,
                    estimate_name="mean delta EMD",
                    estimate=fmean(deltas),
                    statistic_name="t",
                    statistic=float(test.statistic),
                    p_value=float(test.pvalue),
                    units="EMD",
                    data_release="emd_v6",
                    source=_display_source(emd_dir / "values.csv", plotting_data),
                    notes=(
                        f"median delta EMD={median(deltas):.12g}; horizon={horizon} months; "
                        f"positive values favor the full model; {cutoff_summary}"
                    ),
                )
            )
    return tuple(rows)


def _selection_region_statistics(plotting_data: Path) -> list[ManuscriptStatistic]:
    """Reproduce the four functional-region tests in main Figure 2."""

    np, stats = _dependencies()
    specifications = (
        ("A / RBS", "h3n2", "rbs", "RBS", "h3n2_ha_mutations.csv"),
        ("A / antigenic", "h3n2", "koel", "Koel antigenic sites", "h3n2_ha_mutations.csv"),
        ("B / RBS", "h1n1", "rbs", "RBS", "h1n1_ha_mutations.csv"),
        (
            "B / antigenic",
            "h1n1",
            "canton_globular_head",
            "Caton antigenic sites",
            "h1n1_ha_mutations.csv",
        ),
    )
    rows: list[ManuscriptStatistic] = []
    selection_dir = plotting_data / "inputs" / "selection"

    for panel, subtype, field, label, filename in specifications:
        source = selection_dir / filename
        records = [
            row
            for row in _read_csv(source)
            if int(row["mutation_count"]) >= 20
        ]
        selected = np.asarray(
            [
                100.0 * float(row["selection_coefficient"])
                for row in records
                if row[field].strip().lower() == "true"
            ],
            dtype=float,
        )
        other = np.asarray(
            [
                100.0 * float(row["selection_coefficient"])
                for row in records
                if row[field].strip().lower() == "false"
            ],
            dtype=float,
        )
        test = stats.ttest_ind(selected, other, equal_var=True, nan_policy="omit")
        rows.append(
            ManuscriptStatistic(
                figure_label="Figure 2",
                panel=panel,
                family="selection",
                subtype=subtype,
                segment="ha",
                comparison=f"{label} - other sites",
                test="independent Student t-test (equal variance)",
                n_1=int(selected.size),
                n_2=int(other.size),
                estimate_name="difference in mean selection coefficient",
                estimate=float(np.mean(selected) - np.mean(other)),
                statistic_name="t",
                statistic=float(test.statistic),
                p_value=float(test.pvalue),
                units="percentage points",
                data_release="selection_mutations_v1",
                source=_display_source(source, plotting_data),
                notes=(
                    "mutation_count >= 20; "
                    f"subset mean={np.mean(selected):.12g}%; "
                    f"other mean={np.mean(other):.12g}%"
                ),
            )
        )
    return rows


def _correlation_row(
    *,
    figure_label: str,
    panel: str,
    family: str,
    subtype: str | None,
    segment: str,
    comparison: str,
    method: str,
    first: Sequence[float],
    second: Sequence[float],
    data_release: str,
    source: str,
    notes: str,
) -> ManuscriptStatistic:
    np, stats = _dependencies()
    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    if x.size != y.size or x.size < 2:
        raise ValueError(f"Correlation inputs must have equal nontrivial size: {panel}")
    if method == "Pearson":
        result = stats.pearsonr(x, y)
    elif method == "Spearman":
        result = stats.spearmanr(x, y)
    else:  # pragma: no cover - internal programming guard
        raise ValueError(method)
    return ManuscriptStatistic(
        figure_label=figure_label,
        panel=panel,
        family=family,
        subtype=subtype,
        segment=segment,
        comparison=comparison,
        test=f"{method} correlation",
        n_1=int(x.size),
        n_2=None,
        estimate_name="correlation coefficient",
        estimate=float(result.statistic),
        statistic_name=None,
        statistic=None,
        p_value=float(result.pvalue),
        units="dimensionless",
        data_release=data_release,
        source=source,
        notes=notes,
    )


def _selection_correlation_statistics(plotting_data: Path) -> list[ManuscriptStatistic]:
    """Expose the sequence-profile and selection/SASA correlations."""

    aligned_dir = plotting_data / "inputs" / "selection"
    ha_source = aligned_dir / "aligned_ha_selection.csv"
    na_source = aligned_dir / "aligned_na_selection.csv"
    ha = _read_csv(ha_source)
    na = _read_csv(na_source)
    rows: list[ManuscriptStatistic] = []

    ha_windows: Mapping[str, list[dict[str, str]]] = {
        "full": ha,
        # These bounds reproduce the legacy zero-based slices 17:345 and
        # 345:566 used for the manuscript analysis.
        "HA1": [row for row in ha if 18 <= int(row["position"]) <= 345],
        "HA2": [row for row in ha if 346 <= int(row["position"]) <= 566],
    }
    for domain, records in ha_windows.items():
        for method in ("Pearson", "Spearman"):
            rows.append(
                _correlation_row(
                    figure_label="Figure 2",
                    panel=f"C / {domain}",
                    family="selection",
                    subtype="h3n2 vs h1n1",
                    segment="ha",
                    comparison="H3 vs H1 smoothed selection",
                    method=method,
                    first=[float(row["selection_h3n2"]) for row in records],
                    second=[float(row["selection_h1n1"]) for row in records],
                    data_release="aligned_selection_v1",
                    source=_display_source(ha_source, plotting_data),
                    notes="Cross-subtype reference-coordinate mapping; legacy domain bounds.",
                )
            )
    for method in ("Pearson", "Spearman"):
        rows.append(
            _correlation_row(
                figure_label="Figure 2",
                panel="D / full",
                family="selection",
                subtype="h3n2 vs h1n1",
                segment="na",
                comparison="N2 vs N1 smoothed selection",
                method=method,
                first=[float(row["selection_h3n2"]) for row in na],
                second=[float(row["selection_h1n1"]) for row in na],
                data_release="aligned_selection_v1",
                source=_display_source(na_source, plotting_data),
                notes="Cross-subtype reference-coordinate mapping.",
            )
        )

    sasa_source = plotting_data / "derived" / "structure" / "sasa_selection.csv"
    sasa = _read_csv(sasa_source)
    for label, subtype, segment in PANEL_ORDER:
        protein_records = [
            row
            for row in sasa
            if row["subtype"] == subtype and row["segment"] == segment
        ]
        windows: Mapping[str, list[dict[str, str]]]
        if segment == "ha":
            windows = {
                "full": protein_records,
                "HA1": [
                    row
                    for row in protein_records
                    if 18 <= int(row["position"]) <= 345
                ],
                "HA2": [
                    row
                    for row in protein_records
                    if 346 <= int(row["position"]) <= 566
                ],
            }
        else:
            windows = {"full": protein_records}
        for domain, records in windows.items():
            exposed = [
                row for row in records if float(row["sasa_mean_smoothed"]) > 0.0
            ]
            for method in ("Pearson", "Spearman"):
                rows.append(
                    _correlation_row(
                        figure_label="Figure S6",
                        panel=f"{label} / {domain}",
                        family="selection/SASA",
                        subtype=subtype,
                        segment=segment,
                        comparison="smoothed selection vs smoothed SASA",
                        method=method,
                        first=[
                            float(row["selection_coefficient_smoothed"])
                            for row in exposed
                        ],
                        second=[float(row["sasa_mean_smoothed"]) for row in exposed],
                        data_release="sasa_selection_v1",
                        source=_display_source(sasa_source, plotting_data),
                        notes=(
                            "positions with positive smoothed SASA; "
                            "legacy domain bounds for HA"
                        ),
                    )
                )
    return rows


def collect_selection_statistics(
    plotting_data_dir: Path | str,
) -> tuple[ManuscriptStatistic, ...]:
    """Return functional-region and sequence-profile statistics."""

    plotting_data = Path(plotting_data_dir)
    return tuple(
        [
            *_selection_region_statistics(plotting_data),
            *_selection_correlation_statistics(plotting_data),
        ]
    )


def collect_auc_statistics(plotting_data_dir: Path | str) -> tuple[ManuscriptStatistic, ...]:
    """Return the average AUROC improvements printed by legacy plots."""

    plotting_data = Path(plotting_data_dir)
    source = plotting_data / "derived" / "fixation_metrics" / "auc.csv"
    records = _read_csv(source)
    recipes = (
        ("Figure 3", "B", "full_model", "all_time", "full model"),
        ("Figure S7", "", "full_model", "all_time", "full model"),
        (
            "Figure S16",
            "",
            "slm",
            "paper_date_filtered",
            "single-locus model (SLM)",
        ),
    )
    rows: list[ManuscriptStatistic] = []
    for figure_label, panel_prefix, model_id, source_variant, label in recipes:
        for panel_label, subtype, segment in PANEL_ORDER:
            selected = [
                row
                for row in records
                if row["model_id"] == model_id
                and row["source_variant"] == source_variant
                and row["subtype"] == subtype
                and row["segment"] == segment
                and int(row["minimum_observation_day"]) == 1095
            ]
            if len(selected) != 9:
                raise ValueError(
                    f"Expected nine AUROC frequency bins for {figure_label} {panel_label}; "
                    f"found {len(selected)}"
                )
            selected.sort(key=lambda row: float(row["frequency"]))
            improvements = [
                float(row["auc_model"]) - float(row["auc_null"])
                for row in selected
            ]
            rows.append(
                ManuscriptStatistic(
                    figure_label=figure_label,
                    panel=f"{panel_prefix} / {panel_label}" if panel_prefix else panel_label,
                    family="fixation AUROC",
                    subtype=subtype,
                    segment=segment,
                    comparison=f"{label} AUROC - neutral AUROC",
                    test="descriptive mean across frequency bins",
                    n_1=len(improvements),
                    n_2=None,
                    estimate_name="mean AUROC improvement",
                    estimate=fmean(improvements),
                    statistic_name=None,
                    statistic=None,
                    p_value=None,
                    units="AUROC",
                    data_release="fixation_auc_v1",
                    source=_display_source(source, plotting_data),
                    notes="frequency-bin centers 0.1 through 0.9; minimum observation day 1095",
                )
            )
    return tuple(rows)


def _background_rows_for_metric(
    *,
    plotting_data: Path,
    figure_label: str,
    panel_prefix: str,
    metric_field: str,
    metric_label: str,
    minimum_duration_days: int,
    units: str,
) -> list[ManuscriptStatistic]:
    np, stats = _dependencies()
    source = plotting_data / "inputs" / "background" / "population_fitness.csv"
    prepared = prepare_background_plot_data(
        source,
        minimum_duration_days=minimum_duration_days,
    )
    rows: list[ManuscriptStatistic] = []
    for panel in prepared.panels:
        fixed = 100.0 * np.asarray(getattr(panel.fixed, metric_field), dtype=float)
        lost = 100.0 * np.asarray(getattr(panel.lost, metric_field), dtype=float)
        test = stats.ttest_ind(fixed, lost, equal_var=False, nan_policy="omit")
        panel_name = (
            f"{panel_prefix} / {panel.label}" if panel_prefix else panel.label
        )
        rows.append(
            ManuscriptStatistic(
                figure_label=figure_label,
                panel=panel_name,
                family="background fitness",
                subtype=panel.subtype,
                segment=panel.segment,
                comparison="fixed - lost",
                test="Welch independent t-test",
                n_1=int(fixed.size),
                n_2=int(lost.size),
                estimate_name=f"difference in mean {metric_label}",
                estimate=float(np.mean(fixed) - np.mean(lost)),
                statistic_name="t",
                statistic=float(test.statistic),
                p_value=float(test.pvalue),
                units=units,
                data_release="background_fitness_v2",
                source=_display_source(source, plotting_data),
                notes=(
                    f"minimum duration >= {minimum_duration_days} days; "
                    f"fixed mean={np.mean(fixed):.12g}, SD={np.std(fixed):.12g}; "
                    f"lost mean={np.mean(lost):.12g}, SD={np.std(lost):.12g}"
                ),
            )
        )
    return rows


def collect_background_statistics(
    plotting_data_dir: Path | str,
) -> tuple[ManuscriptStatistic, ...]:
    """Return main Figure 5 and Figures S13-S15 fixed/lost tests."""

    plotting_data = Path(plotting_data_dir)
    specifications = (
        (
            "Figure 5",
            "A",
            "selection_coefficient",
            "selection coefficient",
            90,
            "percentage points",
        ),
        (
            "Figure 5",
            "B",
            "initial_background_minus_population_fitness",
            "initial background fitness advantage",
            90,
            "percentage points",
        ),
        (
            "Figure 5",
            "C",
            "annualized_background_fitness_change",
            "annualized background fitness growth",
            90,
            "% per year",
        ),
        (
            "Figure S13",
            "",
            "selection_coefficient",
            "selection coefficient",
            0,
            "percentage points",
        ),
        (
            "Figure S14",
            "",
            "initial_background_minus_population_fitness",
            "initial background fitness advantage",
            90,
            "percentage points",
        ),
        (
            "Figure S15",
            "",
            "annualized_background_fitness_change",
            "annualized background fitness growth",
            90,
            "% per year",
        ),
    )
    rows: list[ManuscriptStatistic] = []
    for (
        figure_label,
        panel_prefix,
        metric_field,
        metric_label,
        minimum_duration,
        units,
    ) in specifications:
        rows.extend(
            _background_rows_for_metric(
                plotting_data=plotting_data,
                figure_label=figure_label,
                panel_prefix=panel_prefix,
                metric_field=metric_field,
                metric_label=metric_label,
                minimum_duration_days=minimum_duration,
                units=units,
            )
        )
    return tuple(rows)


def collect_manuscript_statistics(
    plotting_data_dir: Path | str,
) -> tuple[ManuscriptStatistic, ...]:
    """Collect every currently reproducible manuscript statistic."""

    return (
        *collect_selection_statistics(plotting_data_dir),
        *collect_auc_statistics(plotting_data_dir),
        *collect_emd_statistics(plotting_data_dir),
        *collect_background_statistics(plotting_data_dir),
    )


def _number(value: float | int | None) -> str:
    if value is None:
        return "n.a."
    if isinstance(value, int):
        return str(value)
    if not isfinite(value):
        return str(value)
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude < 1e-3 or magnitude >= 1e4:
        return f"{value:.4e}"
    return f"{value:.6g}"


def format_markdown_report(rows: Iterable[ManuscriptStatistic]) -> str:
    """Format results as a readable, version-controlled Markdown report."""

    grouped: dict[str, list[ManuscriptStatistic]] = {}
    for row in rows:
        grouped.setdefault(row.figure_label, []).append(row)

    lines = [
        "# Manuscript statistics",
        "",
        "This report is generated from the canonical plotting tables by ",
        "`python scripts/report_manuscript_statistics.py`. Statistical results are ",
        "computed independently of figure rendering so they can be reviewed and ",
        "used while editing the manuscript.",
        "",
        "The report preserves the analysis choices used by the historical figure ",
        "calls: a 1,095-day fixation/EMD warmup, `mutation_count >= 20` for ",
        "functional-region tests, equal-variance Student tests for Figure 2, ",
        "one-sample paired-difference tests for EMD, and Welch tests for ",
        "fixed-versus-lost background comparisons. Background estimates are shown ",
        "in the percentage units used on the figures.",
        "",
    ]
    for figure_label, figure_rows in grouped.items():
        lines.extend(
            [
                f"## {figure_label}",
                "",
                "| Panel | Analysis | Comparison | n1 | n2 | Estimate | Units | Statistic | p value | Release |",
                "|---|---|---|---:|---:|---:|---|---:|---:|---|",
            ]
        )
        for row in figure_rows:
            statistic = (
                "n.a."
                if row.statistic is None
                else f"{row.statistic_name}={_number(row.statistic)}"
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        row.panel,
                        row.test,
                        row.comparison,
                        _number(row.n_1),
                        _number(row.n_2),
                        _number(row.estimate),
                        row.units,
                        statistic,
                        _number(row.p_value),
                        row.data_release,
                    )
                )
                + " |"
            )
        lines.extend(["", "Details:", ""])
        for row in figure_rows:
            lines.append(
                f"- **{row.panel}:** {row.estimate_name}. Source: `{row.source}`. "
                f"{row.notes}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "EMD_MANUSCRIPT_RECIPES",
    "ManuscriptStatistic",
    "collect_auc_statistics",
    "collect_background_statistics",
    "collect_emd_statistics",
    "collect_manuscript_statistics",
    "collect_selection_statistics",
    "format_markdown_report",
]
