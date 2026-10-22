#!/usr/bin/env python3
"""Build the canonical non-EMD plotting datasets from the verified archive."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile


ARCHIVE_NAME = "pre_plot_reorg_2026-08-26"


@dataclass(frozen=True)
class MigrationRecord:
    original_path: str
    canonical_path: str
    relationship: str
    source_sha256: str
    canonical_sha256: str
    source_rows: int | None
    canonical_rows: int | None
    schema_id: str
    notes: str = ""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def clone_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["cp", "-c", str(source), str(destination)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        shutil.copy2(source, destination)


def record_copy(
    source_root: Path,
    output_root: Path,
    source_relative: str,
    destination_relative: str,
    schema_id: str,
    records: list[MigrationRecord],
    notes: str = "",
) -> None:
    source = source_root / source_relative
    destination = output_root / destination_relative
    clone_copy(source, destination)
    digest = file_sha256(destination)
    records.append(
        MigrationRecord(
            original_path=source_relative,
            canonical_path=f"data/plotting/{destination_relative}",
            relationship="byte_preserving_rename",
            source_sha256=file_sha256(source),
            canonical_sha256=digest,
            source_rows=None,
            canonical_rows=None,
            schema_id=schema_id,
            notes=notes,
        )
    )


def canonical_identity(filename: str) -> tuple[str, str]:
    match = re.search(r"(H1N1|H3N2)_(HA|NA)", filename)
    if match is None:
        raise ValueError(f"Cannot infer subtype/segment from {filename}")
    return match.group(1).lower(), match.group(2).lower()


def short_identity(filename: str) -> tuple[str, str]:
    match = re.search(r"_(H3|N2|H1|N1)_", filename)
    if match is None:
        raise ValueError(f"Cannot infer short subtype/segment identity from {filename}")
    return {
        "H3": ("h3n2", "ha"),
        "N2": ("h3n2", "na"),
        "H1": ("h1n1", "ha"),
        "N1": ("h1n1", "na"),
    }[match.group(1)]


def migrate_selection(
    source_root: Path, output_root: Path, records: list[MigrationRecord]
) -> None:
    sources = sorted(
        source_root.glob(
            "data/selection_coeff_*_since_01_2009_to_03_2020_filt_observed_add_counts.csv"
        )
    )
    if len(sources) != 4:
        raise ValueError(f"Expected four canonical selection sources, found {len(sources)}")

    rename = {
        "Position": "position",
        "Domain": "domain",
        "Reference": "reference_state",
        "Mutation": "mutation_state",
        "Selection_coefficient": "selection_coefficient",
        "Top_1pct": "top_1pct",
        "Top_20": "top_20",
        "RBS": "rbs",
        "Koel": "koel",
        "Lukza": "lukza",
        "Shih": "shih",
        "Wolf": "wolf",
        "Munoz": "munoz",
        "Bhatt": "bhatt",
        "MutationFrequency": "mutation_count",
    }
    for source in sources:
        subtype, segment = canonical_identity(source.name)
        header, rows = read_csv(source)
        output_fields = ["subtype", "segment"] + [rename.get(name, name.lower()) for name in header]
        converted: list[dict[str, object]] = []
        for row in rows:
            item: dict[str, object] = {"subtype": subtype, "segment": segment}
            for old_name in header:
                new_name = rename.get(old_name, old_name.lower())
                value: object = row[old_name]
                if new_name == "mutation_count" and re.fullmatch(r"-?\d+\.0", str(value)):
                    value = str(value).split(".", 1)[0]
                item[new_name] = value
            converted.append(item)
        destination = output_root / "inputs" / "selection" / f"{subtype}_{segment}_mutations.csv"
        write_csv(destination, output_fields, converted)
        records.append(
            MigrationRecord(
                original_path=source.relative_to(source_root).as_posix(),
                canonical_path=f"data/plotting/{destination.relative_to(output_root).as_posix()}",
                relationship="normalized_column_superset",
                source_sha256=file_sha256(source),
                canonical_sha256=file_sha256(destination),
                source_rows=len(rows),
                canonical_rows=len(converted),
                schema_id="selection_mutations_v1",
                notes="MutationFrequency renamed to mutation_count; identity added from filename",
            )
        )

    aligned_sources = {
        "data/gaussian_smoothed_selection_H3_H1_window_11.csv": (
            "inputs/selection/aligned_ha_selection.csv",
            {"s_h3": "selection_h3n2", "s_h1": "selection_h1n1"},
        ),
        "data/gaussian_smoothed_selection_N2_N1_window_11.csv": (
            "inputs/selection/aligned_na_selection.csv",
            {"s_N2": "selection_h3n2", "s_N1": "selection_h1n1"},
        ),
        "selection_coeff_align_pos_H3-H1.csv": (
            "inputs/selection/aligned_ha_summary.csv",
            {
                "aln_pos": "alignment_position",
                "sum_sel_h3": "selection_sum_h3n2",
                "sum_sel_h1": "selection_sum_h1n1",
                "max_sel_h3": "selection_max_h3n2",
                "max_sel_h1": "selection_max_h1n1",
            },
        ),
    }
    for source_relative, (destination_relative, column_map) in aligned_sources.items():
        source = source_root / source_relative
        header, rows = read_csv(source)
        fields = [column_map.get(name, name.lower()) for name in header]
        converted = [
            {column_map.get(name, name.lower()): row[name] for name in header} for row in rows
        ]
        destination = output_root / destination_relative
        write_csv(destination, fields, converted)
        records.append(
            MigrationRecord(
                original_path=source_relative,
                canonical_path=f"data/plotting/{destination_relative}",
                relationship="normalized_columns",
                source_sha256=file_sha256(source),
                canonical_sha256=file_sha256(destination),
                source_rows=len(rows),
                canonical_rows=len(converted),
                schema_id="aligned_selection_v1",
            )
        )

    for source_relative, destination_relative in {
        "data/fitness_change_h3_tw_11.dat": "inputs/selection/h3n2_ha_fitness_change.dat",
        "data/fitness_change_h1_tw_11.dat": "inputs/selection/h1n1_ha_fitness_change.dat",
        "data/dates_H3.dat": "inputs/selection/h3n2_observation_dates.dat",
        "data/dates_H1.dat": "inputs/selection/h1n1_observation_dates.dat",
    }.items():
        record_copy(
            source_root,
            output_root,
            source_relative,
            destination_relative,
            "selection_time_series_legacy_matrix_v1",
            records,
        )


def migrate_frequencies(
    source_root: Path, output_root: Path, records: list[MigrationRecord]
) -> None:
    for subtype in ("h3n2", "h1n1"):
        for segment in ("ha", "na"):
            source = (
                f"data/allele_frequency_{subtype.upper()}_{segment.upper()}_"
                "2009_to_2020_aa_ts_60_filt.txt"
            )
            destination = f"inputs/frequencies/{subtype}_{segment}_allele_frequencies.txt"
            record_copy(
                source_root,
                output_root,
                source,
                destination,
                "allele_frequency_matrix_v1",
                records,
                notes="Headerless legacy matrix retained byte-for-byte; coordinates documented separately",
            )


def migrate_structure(
    source_root: Path, output_root: Path, records: list[MigrationRecord]
) -> None:
    copies = {
        "data/structure/3LZG-assembly1.cif": "inputs/structure/h1n1_ha_3lzg_assembly1.cif",
        "data/structure/4WE8-assembly1.cif": "inputs/structure/h3n2_ha_4we8_assembly1.cif",
        "data/structure/h1_crop.png": "inputs/structure/h1n1_ha_surface.png",
        "data/structure/h3_crop.png": "inputs/structure/h3n2_ha_surface.png",
        "data/structure/colorbar_crop.png": "inputs/structure/selection_colorbar.png",
    }
    for source, destination in copies.items():
        record_copy(source_root, output_root, source, destination, "structure_asset_v1", records)

    combined: list[dict[str, object]] = []
    sources: list[Path] = []
    for code, subtype, segment in (
        ("H3", "h3n2", "ha"),
        ("N2", "h3n2", "na"),
        ("H1", "h1n1", "ha"),
        ("N1", "h1n1", "na"),
    ):
        source = source_root / f"data/structure/sasa_vs_selection_{code}.csv"
        sources.append(source)
        header, rows = read_csv(source)
        rename = {
            "Position": "position",
            "SASA_mean": "sasa_mean",
            "SASA_mean_smoothed": "sasa_mean_smoothed",
            "Selection_coeff": "selection_coefficient",
            "Selection_coeff_smoothed": "selection_coefficient_smoothed",
        }
        for row in rows:
            combined.append(
                {
                    "subtype": subtype,
                    "segment": segment,
                    **{rename[name]: row[name] for name in header},
                }
            )
    destination = output_root / "derived" / "structure" / "sasa_selection.csv"
    fields = [
        "subtype",
        "segment",
        "position",
        "sasa_mean",
        "sasa_mean_smoothed",
        "selection_coefficient",
        "selection_coefficient_smoothed",
    ]
    write_csv(destination, fields, combined)
    destination_hash = file_sha256(destination)
    for source in sources:
        _, rows = read_csv(source)
        records.append(
            MigrationRecord(
                original_path=source.relative_to(source_root).as_posix(),
                canonical_path="data/plotting/derived/structure/sasa_selection.csv",
                relationship="row_consolidation_and_column_normalization",
                source_sha256=file_sha256(source),
                canonical_sha256=destination_hash,
                source_rows=len(rows),
                canonical_rows=len(combined),
                schema_id="sasa_selection_v1",
            )
        )


def migrate_artwork(
    source_root: Path, output_root: Path, records: list[MigrationRecord]
) -> None:
    """Preserve editable source artwork for figures assembled outside Python."""

    record_copy(
        source_root,
        output_root,
        "figures/fig-model-single.pptx",
        "inputs/artwork/fig-model-overview.pptx",
        "editable_powerpoint_artwork_v1",
        records,
        notes=(
            "Main Figure 1 editable source; export the single slide to PDF "
            "without resizing"
        ),
    )


BACKGROUND_RENAME = {
    "Position": "position",
    "Nucleotide": "state",
    "StartTime": "start_time",
    "frequency": "frequency",
    "T0": "initial_time",
    "T1": "final_time",
    # H0/H1/dH in the full-family files are the same background-fitness
    # quantities named w0_bg/w1_bg/dw_bg in the enriched files.  Keeping
    # both source spellings mapped to one canonical field prevents the same
    # measurement from being split across sparse, misleading columns.
    "H0": "initial_background_fitness",
    "H1": "final_background_fitness",
    "dH": "background_fitness_change",
    "w0_bg": "initial_background_fitness",
    "w0_pop": "initial_population_fitness",
    "w1_bg": "final_background_fitness",
    "dw_bg": "background_fitness_change",
    "Fixation": "outcome",
    "Prediction": "prediction",
    # The legacy plotting code uses `fitness` as the inferred mutation
    # selection coefficient.  Population fitness is already represented by
    # w0_pop above.
    "fitness": "selection_coefficient",
}


def migrate_background(
    source_root: Path, output_root: Path, records: list[MigrationRecord]
) -> None:
    families = {
        "population_fitness": "data/background/df_fitness_diff_*_background_with_w_pop_fitness.csv",
        "full": "data/background/df_fitness_diff_*_background_full.csv",
    }
    fields = [
        "subtype",
        "segment",
        "source_variant",
        "position",
        "state",
        "start_time",
        "frequency",
        "initial_time",
        "final_time",
        "duration_days",
        "selection_coefficient",
        "initial_background_fitness",
        "initial_population_fitness",
        "final_background_fitness",
        "background_fitness_change",
        "outcome",
        "prediction",
    ]
    for variant, pattern in families.items():
        sources = [path for path in sorted(source_root.glob(pattern)) if not path.name.endswith("_old.csv")]
        if len(sources) != 4:
            raise ValueError(f"Expected four {variant} background files, found {len(sources)}")
        combined: list[dict[str, object]] = []
        source_counts: dict[Path, int] = {}
        for source in sources:
            subtype, segment = canonical_identity(source.name)
            header, rows = read_csv(source)
            source_counts[source] = len(rows)
            for row in rows:
                item: dict[str, object] = {
                    "subtype": subtype,
                    "segment": segment,
                    "source_variant": variant,
                }
                for field in fields[3:]:
                    item[field] = ""
                for old_name in header:
                    item[BACKGROUND_RENAME[old_name]] = row[old_name]
                item["duration_days"] = str(
                    int(float(row["T1"]) - float(row["T0"]))
                )
                combined.append(item)
        destination = output_root / "inputs" / "background" / f"{variant}.csv"
        write_csv(destination, fields, combined)
        destination_hash = file_sha256(destination)
        for source in sources:
            records.append(
                MigrationRecord(
                    original_path=source.relative_to(source_root).as_posix(),
                    canonical_path=f"data/plotting/{destination.relative_to(output_root).as_posix()}",
                    relationship="row_consolidation_and_column_normalization",
                    source_sha256=file_sha256(source),
                    canonical_sha256=destination_hash,
                    source_rows=source_counts[source],
                    canonical_rows=len(combined),
                    schema_id="background_fitness_v2",
                    notes=(
                        "Full and population-fitness variants remain separate; "
                        "shared background-fitness fields use common names and "
                        "raw fitness is identified as selection_coefficient"
                    ),
                )
            )


def normalize_date(value: str) -> str:
    value = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value
    for pattern in ("%m/%d/%y", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, pattern).date().isoformat()
        except ValueError:
            pass
    raise ValueError(f"Unsupported date value: {value!r}")


def migrate_clades(source_root: Path, output_root: Path, records: list[MigrationRecord]) -> None:
    source_dir = source_root / "data/data_figs/Fig_Clades_Evolution"
    source_series = {
        "df_observed_full.csv": "observed_full",
        "df_observed_hist.csv": "observed_history",
        "df_forecast.csv": "forecast",
    }
    combined: list[dict[str, object]] = []
    counts: dict[Path, int] = {}
    sources: list[Path] = []
    for filename, series in source_series.items():
        source = source_dir / filename
        sources.append(source)
        header, rows = read_csv(source)
        if header != ["date", "clade", "frequency"]:
            raise ValueError(f"Unexpected clade schema in {source}: {header}")
        counts[source] = len(rows)
        for row in rows:
            combined.append(
                {
                    "subtype": "h3n2",
                    "segment": "ha",
                    "series": series,
                    "date": normalize_date(row["date"]),
                    "clade": row["clade"],
                    "frequency": row["frequency"],
                }
            )
    destination = output_root / "inputs" / "clades" / "h3n2_ha_frequencies.csv"
    fields = ["subtype", "segment", "series", "date", "clade", "frequency"]
    write_csv(destination, fields, combined)
    destination_hash = file_sha256(destination)
    for source in sources:
        records.append(
            MigrationRecord(
                original_path=source.relative_to(source_root).as_posix(),
                canonical_path="data/plotting/inputs/clades/h3n2_ha_frequencies.csv",
                relationship="row_consolidation_and_date_normalization",
                source_sha256=file_sha256(source),
                canonical_sha256=destination_hash,
                source_rows=counts[source],
                canonical_rows=len(combined),
                schema_id="clade_frequency_v1",
            )
        )


FIXATION_RENAME = {
    "Position": "position",
    "Nucleotide": "state",
    "Reference": "reference_state",
    "Time": "observation_time",
    "Empirical_frequency": "empirical_frequency",
    "Time_active": "active_time",
    "Simulations_fixed": "simulated_fixation_probability",
    "Fixation": "outcome",
    "Date": "date",
}


def migrate_fixation(source_root: Path, output_root: Path, records: list[MigrationRecord]) -> None:
    prediction_sources: list[tuple[Path, str, str]] = []
    for source in sorted(
        (source_root / "data/fix-loss").glob("df_FullModel_*_w_time.csv")
    ):
        prediction_sources.append((source, "full_model", "all_time"))
    for source in sorted(
        (source_root / "data/fix-loss").glob("fixation_pred_model_SLM_*_w_date.csv")
    ):
        prediction_sources.append((source, "slm", "paper_date_filtered"))
    for source in sorted((source_root / "data/fix-loss").glob("df_SLM_*_w_time.csv")):
        prediction_sources.append((source, "slm", "extended_all_time"))
    if len(prediction_sources) != 12:
        raise ValueError(f"Expected twelve active prediction files, found {len(prediction_sources)}")

    prediction_fields = [
        "model_id",
        "source_variant",
        "subtype",
        "segment",
        "position",
        "state",
        "reference_state",
        "observation_time",
        "empirical_frequency",
        "active_time",
        "simulated_fixation_probability",
        "outcome",
        "date",
    ]
    predictions: list[dict[str, object]] = []
    counts: dict[Path, int] = {}
    for source, model_id, source_variant in prediction_sources:
        subtype, segment = (
            short_identity(source.name)
            if source.name.startswith("fixation_pred_model")
            else canonical_identity(source.name)
        )
        header, rows = read_csv(source)
        counts[source] = len(rows)
        for row in rows:
            predictions.append(
                {
                    "model_id": model_id,
                    "source_variant": source_variant,
                    "subtype": subtype,
                    "segment": segment,
                    **{FIXATION_RENAME[name]: row[name] for name in header},
                    **({} if "Date" in header else {"date": ""}),
                }
            )
    prediction_destination = output_root / "inputs" / "fixation" / "predictions.csv"
    write_csv(prediction_destination, prediction_fields, predictions)
    prediction_hash = file_sha256(prediction_destination)
    for source, _, source_variant in prediction_sources:
        records.append(
            MigrationRecord(
                original_path=source.relative_to(source_root).as_posix(),
                canonical_path="data/plotting/inputs/fixation/predictions.csv",
                relationship="row_consolidation_and_column_normalization",
                source_sha256=file_sha256(source),
                canonical_sha256=prediction_hash,
                source_rows=counts[source],
                canonical_rows=len(predictions),
                schema_id="fixation_predictions_v1",
                notes=(
                    f"source_variant={source_variant}; FMSS and SLFS excluded from active data "
                    "as legacy diagnostics"
                ),
            )
        )

    auc_fields = [
        "model_id",
        "source_variant",
        "subtype",
        "segment",
        "minimum_observation_day",
        "frequency",
        "auc_model",
        "auc_null",
        "z_score",
        "p_value",
    ]
    auc_rows: list[dict[str, object]] = []
    auc_sources: list[tuple[Path, str, str]] = []
    for source in sorted((source_root / "data/fix-loss").glob("df_FullModel_AUC_*_*.csv")):
        auc_sources.append((source, "full_model", "all_time"))
    for source in sorted((source_root / "data/fix-loss").glob("df_SLM_AUC_*_1095.csv")):
        auc_sources.append((source, "slm", "extended_all_time"))
    for source in sorted((source_root / "data/fix-loss").glob("df_SS_AUC_*_1095.csv")):
        auc_sources.append((source, "slm", "paper_date_filtered"))
    auc_counts: dict[Path, int] = {}
    for source, model_id, source_variant in auc_sources:
        subtype, segment = (
            short_identity(source.name)
            if source.name.startswith("df_SS_AUC")
            else canonical_identity(source.name)
        )
        cutoff_match = re.search(r"_(\d+)\.csv$", source.name)
        if cutoff_match is None:
            raise ValueError(f"No cutoff in {source.name}")
        _, rows = read_csv(source)
        auc_counts[source] = len(rows)
        for row in rows:
            auc_rows.append(
                {
                    "model_id": model_id,
                    "source_variant": source_variant,
                    "subtype": subtype,
                    "segment": segment,
                    "minimum_observation_day": cutoff_match.group(1),
                    "frequency": row["Frequency"],
                    "auc_model": row["AUC_simulation"],
                    "auc_null": row["AUC_null"],
                    "z_score": row.get("AUC_z", ""),
                    "p_value": row.get("AUC_p", ""),
                }
            )
    auc_destination = output_root / "derived" / "fixation_metrics" / "auc.csv"
    write_csv(auc_destination, auc_fields, auc_rows)
    auc_hash = file_sha256(auc_destination)
    for source, _, source_variant in auc_sources:
        records.append(
            MigrationRecord(
                original_path=source.relative_to(source_root).as_posix(),
                canonical_path="data/plotting/derived/fixation_metrics/auc.csv",
                relationship="row_consolidation_and_column_normalization",
                source_sha256=file_sha256(source),
                canonical_sha256=auc_hash,
                source_rows=auc_counts[source],
                canonical_rows=len(auc_rows),
                schema_id="fixation_auc_v1",
                notes=f"source_variant={source_variant}",
            )
        )

    pfix_fields = [
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
    ]
    pfix_rows: list[dict[str, object]] = []
    pfix_sources = sorted((source_root / "data/fix-loss").glob("df_FullModel_pfix_*_*.csv"))
    pfix_counts: dict[Path, int] = {}
    pfix_rename = {
        "Frequency": "frequency",
        "Top_score": "top_score",
        "Bottom_score": "bottom_score",
        "Top25_score": "top_25_score",
        "Bottom25_score": "bottom_25_score",
        "Null": "null",
    }
    for source in pfix_sources:
        subtype, segment = canonical_identity(source.name)
        cutoff_match = re.search(r"_(\d+)\.csv$", source.name)
        if cutoff_match is None:
            raise ValueError(f"No cutoff in {source.name}")
        header, rows = read_csv(source)
        pfix_counts[source] = len(rows)
        for row in rows:
            pfix_rows.append(
                {
                    "model_id": "full_model",
                    "subtype": subtype,
                    "segment": segment,
                    "minimum_observation_day": cutoff_match.group(1),
                    **{pfix_rename[name]: row[name] for name in header},
                }
            )
    pfix_destination = output_root / "derived" / "fixation_metrics" / "fixation_probability.csv"
    write_csv(pfix_destination, pfix_fields, pfix_rows)
    pfix_hash = file_sha256(pfix_destination)
    for source in pfix_sources:
        records.append(
            MigrationRecord(
                original_path=source.relative_to(source_root).as_posix(),
                canonical_path="data/plotting/derived/fixation_metrics/fixation_probability.csv",
                relationship="row_consolidation_and_column_normalization",
                source_sha256=file_sha256(source),
                canonical_sha256=pfix_hash,
                source_rows=pfix_counts[source],
                canonical_rows=len(pfix_rows),
                schema_id="fixation_probability_v1",
            )
        )


def migrate_presentation_metrics(
    source_root: Path, output_root: Path, records: list[MigrationRecord]
) -> None:
    auc_sources = sorted((source_root / "data/Fig_S1").glob("AUC_vs_f_T0_2015_*.csv"))
    auc_sources += sorted((source_root / "data/Fig_5").glob("AUC_vs_f_T0_2015_*.csv"))
    if len(auc_sources) != 4:
        raise ValueError(f"Expected four 2015 AUC presentation files, found {len(auc_sources)}")
    auc_rows: list[dict[str, object]] = []
    counts: dict[Path, int] = {}
    for source in auc_sources:
        subtype, segment = canonical_identity(source.name.replace("H1N1pdm", "H1N1"))
        _, rows = read_csv(source)
        counts[source] = len(rows)
        for row in rows:
            auc_rows.append(
                {
                    "year": "2015",
                    "subtype": subtype,
                    "segment": segment,
                    "frequency": row["Frequency"],
                    "auc_model": row["AUC_simulation"],
                    "auc_null": row["AUC_null"],
                }
            )
    auc_destination = output_root / "derived" / "fixation_metrics" / "auc_2015.csv"
    write_csv(
        auc_destination,
        ["year", "subtype", "segment", "frequency", "auc_model", "auc_null"],
        auc_rows,
    )
    auc_hash = file_sha256(auc_destination)
    for source in auc_sources:
        records.append(
            MigrationRecord(
                original_path=source.relative_to(source_root).as_posix(),
                canonical_path="data/plotting/derived/fixation_metrics/auc_2015.csv",
                relationship="row_consolidation_and_column_normalization",
                source_sha256=file_sha256(source),
                canonical_sha256=auc_hash,
                source_rows=counts[source],
                canonical_rows=len(auc_rows),
                schema_id="presentation_auc_2015_v1",
            )
        )

    pfix_sources = sorted(
        (source_root / "data/Fig_S2").glob("Fix_prob_vs_f_T0_2015_*_HA.csv")
    )
    if len(pfix_sources) != 2:
        raise ValueError(
            f"Expected two 2015 HA fixation-probability presentation files, found {len(pfix_sources)}"
        )
    pfix_rows: list[dict[str, object]] = []
    pfix_counts: dict[Path, int] = {}
    for source in pfix_sources:
        subtype, segment = canonical_identity(source.name.replace("H1N1pdm", "H1N1"))
        _, rows = read_csv(source)
        pfix_counts[source] = len(rows)
        for row in rows:
            pfix_rows.append(
                {
                    "year": "2015",
                    "subtype": subtype,
                    "segment": segment,
                    "frequency": row["Frequency"],
                    "null": row["Null"],
                    "top_score": row["Top_score"],
                    "bottom_score": row["Bottom_score"],
                }
            )
    pfix_destination = (
        output_root / "derived" / "fixation_metrics" / "fixation_probability_2015.csv"
    )
    write_csv(
        pfix_destination,
        [
            "year",
            "subtype",
            "segment",
            "frequency",
            "null",
            "top_score",
            "bottom_score",
        ],
        pfix_rows,
    )
    pfix_hash = file_sha256(pfix_destination)
    for source in pfix_sources:
        records.append(
            MigrationRecord(
                original_path=source.relative_to(source_root).as_posix(),
                canonical_path=(
                    "data/plotting/derived/fixation_metrics/fixation_probability_2015.csv"
                ),
                relationship="row_consolidation_and_column_normalization",
                source_sha256=file_sha256(source),
                canonical_sha256=pfix_hash,
                source_rows=pfix_counts[source],
                canonical_rows=len(pfix_rows),
                schema_id="presentation_fixation_probability_2015_v1",
            )
        )


def write_deterministic_zip(source_dir: Path, destination: Path) -> list[Path]:
    files = [
        path
        for path in sorted(source_dir.rglob("*"))
        if path.is_file() and path.name not in {".DS_Store"} and path.suffix != ".ipynb"
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(source_dir).as_posix(), (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return files


def migrate_examples(source_root: Path, output_root: Path, records: list[MigrationRecord]) -> None:
    for number in (1, 2):
        source_dir = source_root / f"data/Fix-loss_example_{number}"
        destination = output_root / "inputs" / "examples" / f"fix_loss_example_{number}.zip"
        files = write_deterministic_zip(source_dir, destination)
        destination_hash = file_sha256(destination)
        for source in files:
            records.append(
                MigrationRecord(
                    original_path=source.relative_to(source_root).as_posix(),
                    canonical_path=f"data/plotting/{destination.relative_to(output_root).as_posix()}",
                    relationship="deterministic_zip_consolidation",
                    source_sha256=file_sha256(source),
                    canonical_sha256=destination_hash,
                    source_rows=None,
                    canonical_rows=None,
                    schema_id="fix_loss_example_bundle_v1",
                )
            )
    for source_relative, destination_relative in {
        "data/example/freq_vs_time_toy_model_example_q_4_L_20_T_1000_mu_1e-3_example.txt": "inputs/examples/toy_model_frequencies.txt",
        "data/example/h0_q_4_L_20_mu_h_0.0_sigma_h_0.05_example.txt": "inputs/examples/toy_model_initial_fitness.txt",
        "data/example/hinf_q_4_L_20_mu_h_0.0_sigma_h_0.05_example.txt": "inputs/examples/toy_model_final_fitness.txt",
    }.items():
        record_copy(
            source_root,
            output_root,
            source_relative,
            destination_relative,
            "toy_model_example_v1",
            records,
        )


def migrate_surveillance(
    source_root: Path, output_root: Path, records: list[MigrationRecord]
) -> None:
    source_relative = "data/CumulativeNoOfSequencesByYear.csv"
    source = source_root / source_relative
    header, rows = read_csv(source)
    if header != ["date", "count"]:
        raise ValueError(f"Unexpected surveillance schema: {header}")
    converted = [
        {"year": row["date"], "sequence_count": row["count"]} for row in rows
    ]
    destination = output_root / "inputs" / "surveillance" / "sequences_by_year.csv"
    write_csv(destination, ["year", "sequence_count"], converted)
    records.append(
        MigrationRecord(
            original_path=source_relative,
            canonical_path="data/plotting/inputs/surveillance/sequences_by_year.csv",
            relationship="normalized_columns_and_bom_removal",
            source_sha256=file_sha256(source),
            canonical_sha256=file_sha256(destination),
            source_rows=len(rows),
            canonical_rows=len(converted),
            schema_id="surveillance_counts_v1",
        )
    )


def migrate_overview(source_root: Path, output_root: Path, records: list[MigrationRecord]) -> None:
    source_scenarios = {
        "figures/overview-frequencies-true.csv": "observed",
        "figures/overview-frequencies-sim.csv": "simulation",
        "figures/overview-frequencies-fixloss.csv": "fixation_loss",
    }
    combined: list[dict[str, object]] = []
    counts: dict[Path, int] = {}
    sources: list[Path] = []
    for source_relative, scenario in source_scenarios.items():
        source = source_root / source_relative
        sources.append(source)
        _, rows = read_csv(source)
        counts[source] = len(rows)
        for row in rows:
            combined.append(
                {
                    "scenario": scenario,
                    "simulation_id": row.get("sim", ""),
                    "variant_id": row["variant"],
                    "time": row["time"],
                    "frequency": row["frequency"],
                }
            )
    destination = output_root / "derived" / "overview" / "trajectories.csv"
    fields = ["scenario", "simulation_id", "variant_id", "time", "frequency"]
    write_csv(destination, fields, combined)
    destination_hash = file_sha256(destination)
    for source in sources:
        records.append(
            MigrationRecord(
                original_path=source.relative_to(source_root).as_posix(),
                canonical_path="data/plotting/derived/overview/trajectories.csv",
                relationship="row_consolidation_and_column_normalization",
                source_sha256=file_sha256(source),
                canonical_sha256=destination_hash,
                source_rows=counts[source],
                canonical_rows=len(combined),
                schema_id="overview_trajectories_v1",
            )
        )


def write_metadata(output_root: Path, records: list[MigrationRecord]) -> None:
    fields = [field.name for field in MigrationRecord.__dataclass_fields__.values()]
    rows = [asdict(record) for record in records]
    write_csv(output_root / "migration_map.csv", fields, rows)
    schema_summary: dict[str, dict[str, object]] = {}
    for record in records:
        item = schema_summary.setdefault(
            record.schema_id,
            {"source_files": 0, "canonical_paths": set()},
        )
        item["source_files"] = int(item["source_files"]) + 1
        paths = item["canonical_paths"]
        assert isinstance(paths, set)
        paths.add(record.canonical_path)
    serializable = {
        key: {
            "source_files": value["source_files"],
            "canonical_paths": sorted(value["canonical_paths"]),
        }
        for key, value in sorted(schema_summary.items())
    }
    (output_root / "schema_summary.json").write_text(
        json.dumps(serializable, indent=2) + "\n", encoding="utf-8"
    )


def build(source_root: Path, output_root: Path) -> None:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing canonical data: {output_root}")
    output_root.mkdir(parents=True)
    records: list[MigrationRecord] = []
    migrate_selection(source_root, output_root, records)
    migrate_frequencies(source_root, output_root, records)
    migrate_structure(source_root, output_root, records)
    migrate_artwork(source_root, output_root, records)
    migrate_background(source_root, output_root, records)
    migrate_clades(source_root, output_root, records)
    migrate_fixation(source_root, output_root, records)
    migrate_presentation_metrics(source_root, output_root, records)
    migrate_examples(source_root, output_root, records)
    migrate_surveillance(source_root, output_root, records)
    migrate_overview(source_root, output_root, records)
    write_metadata(output_root, records)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=repo / "archive" / ARCHIVE_NAME / "original",
        help="Root of the verified original snapshot",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo / "data" / "plotting",
        help="Canonical plotting data root",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source_root.is_dir():
        print(f"Missing source archive: {args.source_root}", file=sys.stderr)
        return 2
    build(args.source_root.resolve(), args.output_root.resolve())
    print(args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
