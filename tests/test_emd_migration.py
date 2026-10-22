"""Focused tests for deterministic canonical EMD migration."""

from __future__ import annotations

import csv
import hashlib
import sys
import tempfile
import unittest
from collections import Counter
from datetime import date
from decimal import Decimal
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
ARCHIVE_SOURCE_ROOT = (
    REPO_ROOT / "archive" / "pre_plot_reorg_2026-08-26" / "original"
)
MIGRATION_SOURCE_ROOT = (
    ARCHIVE_SOURCE_ROOT
    if (ARCHIVE_SOURCE_ROOT / "data" / "emd").is_dir()
    else REPO_ROOT
)
UNCORRECTED_CV_SOURCE_ROOT = (
    REPO_ROOT / "archive" / "emd_cv_2026-09-24" / "original" / "tmp"
)
PREVIOUS_CORRECTED_CV_9M_SOURCE_ROOT = (
    REPO_ROOT / "archive" / "emd_cv_2026-09-30" / "original" / "tmp"
)
PREVIOUS_CORRECTED_CV_6_12_SOURCE_ROOT = (
    REPO_ROOT / "archive" / "emd_cv_2026-10-10" / "original" / "tmp"
)
CORRECTED_CV_SOURCE_ROOT = (
    REPO_ROOT / "archive" / "emd_cv_2026-10-13" / "original" / "tmp"
)
CANONICAL_DIR = REPO_ROOT / "data" / "plotting" / "inputs" / "emd"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from influenza_figures.prepare.emd import (  # noqa: E402
    EVALUATION_FIELDS,
    EXPECTED_FAMILY_COUNTS,
    METHOD_FIELDS,
    MigrationError,
    SOURCE_FIELDS,
    VALUE_FIELDS,
    migrate_emd,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


class EmdMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.legacy_paths = sorted(
            (MIGRATION_SOURCE_ROOT / "data" / "emd").rglob("*.csv")
        )
        cls.legacy_hashes_before = {path: sha256(path) for path in cls.legacy_paths}
        cls.uncorrected_cv_paths = sorted(UNCORRECTED_CV_SOURCE_ROOT.glob("*.csv"))
        cls.uncorrected_cv_hashes_before = {
            path: sha256(path) for path in cls.uncorrected_cv_paths
        }
        cls.previous_corrected_cv_9m_paths = sorted(
            PREVIOUS_CORRECTED_CV_9M_SOURCE_ROOT.glob("*.csv")
        )
        cls.previous_corrected_cv_9m_hashes_before = {
            path: sha256(path) for path in cls.previous_corrected_cv_9m_paths
        }
        cls.previous_corrected_cv_6_12_paths = sorted(
            PREVIOUS_CORRECTED_CV_6_12_SOURCE_ROOT.glob("*.csv")
        )
        cls.previous_corrected_cv_6_12_hashes_before = {
            path: sha256(path) for path in cls.previous_corrected_cv_6_12_paths
        }
        cls.corrected_cv_paths = sorted(CORRECTED_CV_SOURCE_ROOT.glob("*.csv"))
        cls.corrected_cv_hashes_before = {
            path: sha256(path) for path in cls.corrected_cv_paths
        }
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="emd-migration-test-")
        cls.output_dir = Path(cls.temp_dir.name) / "canonical"
        cls.summary = migrate_emd(
            MIGRATION_SOURCE_ROOT,
            cls.output_dir,
            cv_9m_root=UNCORRECTED_CV_SOURCE_ROOT,
            corrected_cv_9m_root=CORRECTED_CV_SOURCE_ROOT,
            corrected_cv_6_12_root=CORRECTED_CV_SOURCE_ROOT,
            previous_corrected_cv_9m_root=(
                PREVIOUS_CORRECTED_CV_9M_SOURCE_ROOT
            ),
            previous_corrected_cv_6_12_root=(
                PREVIOUS_CORRECTED_CV_6_12_SOURCE_ROOT
            ),
        )
        cls.evaluation_header, cls.evaluations = read_csv(
            cls.output_dir / "evaluations.csv"
        )
        cls.value_header, cls.values = read_csv(cls.output_dir / "values.csv")
        cls.method_header, cls.methods = read_csv(cls.output_dir / "methods.csv")
        cls.source_header, cls.sources = read_csv(cls.output_dir / "sources.csv")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_evaluations_have_canonical_schema_unique_ids_and_expected_protocols(self) -> None:
        self.assertEqual(self.evaluation_header, EVALUATION_FIELDS)
        self.assertEqual(len(self.evaluations), 2591)
        evaluation_ids = [row["evaluation_id"] for row in self.evaluations]
        self.assertEqual(len(evaluation_ids), len(set(evaluation_ids)))
        self.assertEqual(
            Counter(row["evaluation_protocol"] for row in self.evaluations),
            {"full_grid": 1427, "results_grid": 1164},
        )
        self.assertEqual(
            Counter(
                (row["evaluation_protocol"], row["horizon_months"])
                for row in self.evaluations
            ),
            {
                ("full_grid", "6"): 476,
                ("full_grid", "9"): 476,
                ("full_grid", "12"): 475,
                ("results_grid", "6"): 388,
                ("results_grid", "9"): 388,
                ("results_grid", "12"): 388,
            },
        )
        self.assertTrue(all(row["protocol_version"] == "1" for row in self.evaluations))
        self.assertTrue(all(row["observation_date"].count("-") == 2 for row in self.evaluations))
        self.assertTrue(all(row["prediction_date"].count("-") == 2 for row in self.evaluations))

    def test_evaluations_replace_legacy_h1_time_offset_with_calendar_days(self) -> None:
        source = next(
            row
            for row in self.sources
            if row["original_path"]
            == "data/emd/EMD_12m_FullModel_H1N1_HA_full.csv"
        )
        first_source_values = [
            row
            for row in self.values
            if row["source_id"] == source["source_id"] and row["source_row"] == "2"
        ]
        self.assertEqual(len(first_source_values), 5)
        evaluation_id = first_source_values[0]["evaluation_id"]
        evaluation = next(
            row for row in self.evaluations if row["evaluation_id"] == evaluation_id
        )
        self.assertEqual(evaluation["observation_date"], "2009-05-31")
        self.assertEqual(evaluation["observation_day_2009"], "150")

    def test_values_have_unique_keys_and_expected_method_counts(self) -> None:
        self.assertEqual(self.value_header, VALUE_FIELDS)
        self.assertEqual(len(self.values), 16578)
        keys = [
            (row["evaluation_id"], row["method_id"], row["result_version"])
            for row in self.values
        ]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(
            Counter(row["method_id"] for row in self.values),
            {
                "lower_bound": 2591,
                "null": 2591,
                "full_model_raw": 1427,
                "full_model_unlinked": 1427,
                "full_model_rare_filtered": 2591,
                "full_model_cross_validated_rare_filtered": 1164,
                "single_locus_model_rare_filtered": 1508,
                "multinomial_logistic_regression_rare_filtered": 2591,
                "full_model_single_site_simulation": 344,
                "single_locus_full_sites_simulation": 344,
            },
        )
        self.assertTrue(all(row["result_version"] == "current" for row in self.values))
        self.assertTrue(
            all(row["transform_version"] == "emd_migration_v6" for row in self.values)
        )

    def test_corrected_six_month_columns_use_confirmed_model_semantics(self) -> None:
        source = next(
            row
            for row in self.sources
            if row["original_path"]
            == "tmp/n1_6m_hyperparam_selection_summary_corrected.csv"
        )
        first_row_values = {
            row["source_column"]: row
            for row in self.values
            if row["source_id"] == source["source_id"] and row["source_row"] == "2"
        }
        self.assertEqual(
            {column: row["method_id"] for column, row in first_row_values.items()},
            {
                "Lower Bound": "lower_bound",
                "Null": "null",
                "SLM": "single_locus_model_rare_filtered",
                "MLR": "multinomial_logistic_regression_rare_filtered",
                "GenModel (baseline)": "full_model_rare_filtered",
                "best GenModel": "full_model_cross_validated_rare_filtered",
            },
        )
        self.assertIn("definitive", first_row_values["Null"]["value_note"])
        self.assertEqual(
            first_row_values["best GenModel"]["selected_hyperparameters"],
            "gamma=200, f=1e-2",
        )
        evaluation = next(
            row
            for row in self.evaluations
            if row["evaluation_id"]
            == first_row_values["GenModel (baseline)"]["evaluation_id"]
        )
        self.assertEqual(
            (evaluation["subtype"], evaluation["segment"], evaluation["horizon_months"]),
            ("h1n1", "na", "6"),
        )

    def test_legacy_diagnostics_have_confirmed_expanded_names(self) -> None:
        source = next(
            row
            for row in self.sources
            if row["original_path"]
            == "data/emd/EMD_12m_AllModels_H3N2_HA_filt.csv"
        )
        first_row = {
            row["source_column"]: row["method_id"]
            for row in self.values
            if row["source_id"] == source["source_id"] and row["source_row"] == "2"
        }
        self.assertEqual(
            first_row,
            {
                "FMSS": "full_model_single_site_simulation",
                "SLM": "single_locus_model_rare_filtered",
                "SLFS": "single_locus_full_sites_simulation",
            },
        )

    def test_cross_validated_release_maps_best_and_baseline_separately(self) -> None:
        source = next(
            row
            for row in self.sources
            if row["original_path"]
            == "tmp/h3_9m_hyperparam_selection_summary_corrected.csv"
        )
        first_row_values = {
            row["source_column"]: row
            for row in self.values
            if row["source_id"] == source["source_id"] and row["source_row"] == "2"
        }
        self.assertEqual(
            {column: row["method_id"] for column, row in first_row_values.items()},
            {
                "Lower Bound": "lower_bound",
                "Null": "null",
                "SLM": "single_locus_model_rare_filtered",
                "MLR": "multinomial_logistic_regression_rare_filtered",
                "GenModel (baseline)": "full_model_rare_filtered",
                "best GenModel": "full_model_cross_validated_rare_filtered",
            },
        )
        best = first_row_values["best GenModel"]
        self.assertEqual(
            best["selected_hyperparameters"],
            "gamma=20, f=1e-4 (baseline)",
        )
        self.assertIn("cross-validation", best["value_note"])
        self.assertEqual(
            first_row_values["GenModel (baseline)"]["selected_hyperparameters"],
            "",
        )
        self.assertIn(
            "Corrected null EMD",
            first_row_values["Null"]["value_note"],
        )

    def test_every_cross_validated_cell_has_exact_value_and_source_provenance(self) -> None:
        sources_by_path = {row["original_path"]: row for row in self.sources}
        numeric_columns = (
            "Lower Bound",
            "Null",
            "SLM",
            "MLR",
            "GenModel (baseline)",
            "best GenModel",
        )
        for path in self.corrected_cv_paths:
            with self.subTest(path=path.name):
                source = sources_by_path[f"tmp/{path.name}"]
                canonical = {
                    (row["source_row"], row["source_column"]): row
                    for row in self.values
                    if row["source_id"] == source["source_id"]
                }
                self.assertEqual(len(canonical), 97 * len(numeric_columns))
                with path.open(encoding="utf-8", newline="") as handle:
                    for source_row, raw in enumerate(csv.DictReader(handle), start=2):
                        for column in numeric_columns:
                            value = canonical[(str(source_row), column)]
                            self.assertEqual(Decimal(value["emd"]), Decimal(raw[column]))
                        best = canonical[(str(source_row), "best GenModel")]
                        self.assertEqual(
                            best["selected_hyperparameters"],
                            raw["CV parameter"],
                        )
                        if raw["CV parameter"].endswith("(baseline)"):
                            self.assertEqual(
                                Decimal(raw["best GenModel"]),
                                Decimal(raw["GenModel (baseline)"]),
                            )

    def test_corrected_cross_validated_grids_align_across_all_horizons(self) -> None:
        panels: dict[tuple[int, str], tuple[tuple[str, ...], ...]] = {}
        for path in self.corrected_cv_paths:
            code, horizon_label = path.name.split("_", 2)[:2]
            horizon = int(horizon_label.removesuffix("m"))
            _, rows = read_csv(path)
            self.assertEqual(len(rows), 97)
            grid: list[tuple[str, ...]] = []
            previous_observation: date | None = None
            for row in rows:
                observed = date.fromisoformat(row["obs_date"])
                predicted = date.fromisoformat(row["pred_date"])
                declared_days = int(row["horizon_days"])
                self.assertEqual(declared_days, {6: 180, 9: 270, 12: 360}[horizon])
                self.assertLessEqual(abs((predicted - observed).days - declared_days), 2)
                if previous_observation is not None:
                    self.assertGreater(observed, previous_observation)
                previous_observation = observed
                grid.append(
                    (
                        row["subtype"],
                        row["protein"],
                        row["region"],
                        row["start_date"],
                        row["obs_date"],
                    )
                )
            panels[(horizon, code)] = tuple(grid)

        self.assertEqual(len(panels), 12)
        for code in ("h1", "h3", "n1", "n2"):
            self.assertEqual(panels[(6, code)], panels[(9, code)])
            self.assertEqual(panels[(12, code)], panels[(9, code)])

    def test_authoritative_cv_release_has_no_lower_bound_inversions(self) -> None:
        inversions: list[tuple[object, ...]] = []
        model_columns = (
            "Null",
            "SLM",
            "MLR",
            "GenModel (baseline)",
            "best GenModel",
        )
        for path in self.corrected_cv_paths:
            code, horizon_label = path.name.split("_", 2)[:2]
            horizon = int(horizon_label.removesuffix("m"))
            _, rows = read_csv(path)
            for source_row, row in enumerate(rows, start=2):
                lower_bound = Decimal(row["Lower Bound"])
                for column in model_columns:
                    if lower_bound > Decimal(row[column]):
                        inversions.append(
                            (
                                horizon,
                                code,
                                source_row,
                                row["obs_date"],
                                row["pred_date"],
                                column,
                                row["Lower Bound"],
                                row[column],
                            )
                        )
        self.assertEqual(inversions, [])

        source = next(
            row
            for row in self.sources
            if row["original_path"]
            == "tmp/h3_12m_hyperparam_selection_summary_corrected.csv"
        )
        canonical = {
            row["source_column"]: row
            for row in self.values
            if row["source_id"] == source["source_id"]
            and row["source_row"] == "32"
        }
        self.assertEqual(canonical["Lower Bound"]["emd"], "2.8464822104905676")
        self.assertEqual(canonical["best GenModel"]["emd"], "4.1554496544696775")
        self.assertEqual(
            canonical["best GenModel"]["selected_hyperparameters"],
            "gamma=250, f=1e-3",
        )
        self.assertIn(
            "horizon pairing",
            canonical["best GenModel"]["value_note"],
        )

    def test_horizon_pairing_correction_changes_only_best_model_selection(
        self,
    ) -> None:
        previous_paths = (
            self.previous_corrected_cv_9m_paths
            + self.previous_corrected_cv_6_12_paths
        )
        current_by_name = {path.name: path for path in self.corrected_cv_paths}
        self.assertEqual(
            set(current_by_name),
            {path.name for path in previous_paths},
        )
        changes_by_panel: Counter[tuple[int, str]] = Counter()
        invariant_columns = (
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
        )

        for previous_path in previous_paths:
            code, horizon_label = previous_path.name.split("_", 2)[:2]
            horizon = int(horizon_label.removesuffix("m"))
            current_path = current_by_name[previous_path.name]
            with self.subTest(source=previous_path.name):
                previous_header, previous_rows = read_csv(previous_path)
                current_header, current_rows = read_csv(current_path)
                self.assertEqual(previous_header, current_header)
                self.assertEqual(len(previous_rows), 97)
                self.assertEqual(len(current_rows), 97)
                for previous, current in zip(
                    previous_rows,
                    current_rows,
                    strict=True,
                ):
                    for column in invariant_columns:
                        self.assertEqual(previous[column], current[column])
                    best_changed = Decimal(previous["best GenModel"]) != Decimal(
                        current["best GenModel"]
                    )
                    parameter_changed = (
                        previous["CV parameter"] != current["CV parameter"]
                    )
                    self.assertEqual(best_changed, parameter_changed)
                    if best_changed:
                        changes_by_panel[(horizon, code)] += 1

        self.assertEqual(
            changes_by_panel,
            Counter(
                {
                    (6, "n1"): 13,
                    (12, "h3"): 97,
                    (12, "n1"): 10,
                }
            ),
        )
        self.assertEqual(sum(changes_by_panel.values()), 120)

    def test_correction_changes_only_null_for_the_same_388_evaluations(self) -> None:
        key_columns = (
            "subtype",
            "protein",
            "region",
            "start_date",
            "obs_date",
            "pred_date",
            "horizon_days",
        )
        null_differences = 0

        for uncorrected_path in self.uncorrected_cv_paths:
            corrected_name = uncorrected_path.stem + "_corrected.csv"
            corrected_path = PREVIOUS_CORRECTED_CV_9M_SOURCE_ROOT / corrected_name
            with self.subTest(source=uncorrected_path.name):
                uncorrected_header, uncorrected_rows = read_csv(uncorrected_path)
                corrected_header, corrected_rows = read_csv(corrected_path)
                self.assertEqual(uncorrected_header, corrected_header)

                def index_rows(rows: list[dict[str, str]]) -> dict[tuple[str, ...], dict[str, str]]:
                    return {
                        tuple(row[column] for column in key_columns): row
                        for row in rows
                    }

                uncorrected = index_rows(uncorrected_rows)
                corrected = index_rows(corrected_rows)
                self.assertEqual(len(uncorrected), 97)
                self.assertEqual(uncorrected.keys(), corrected.keys())

                unchanged_columns = tuple(
                    column for column in uncorrected_header if column != "Null"
                )
                for key in uncorrected:
                    for column in unchanged_columns:
                        self.assertEqual(
                            uncorrected[key][column],
                            corrected[key][column],
                            f"Unexpected change in {column} for {key}",
                        )
                    self.assertNotEqual(
                        Decimal(uncorrected[key]["Null"]),
                        Decimal(corrected[key]["Null"]),
                    )
                    null_differences += 1

        self.assertEqual(null_differences, 388)

    def test_methods_encode_filtering_and_diagnostic_definitions(self) -> None:
        self.assertEqual(self.method_header, METHOD_FIELDS)
        self.assertEqual(len(self.methods), 10)
        methods = {row["method_id"]: row for row in self.methods}
        for method_id in (
            "full_model_rare_filtered",
            "full_model_cross_validated_rare_filtered",
            "single_locus_model_rare_filtered",
            "multinomial_logistic_regression_rare_filtered",
        ):
            self.assertEqual(methods[method_id]["rare_mutation_policy"], "removed")
            self.assertEqual(methods[method_id]["definition_status"], "confirmed")
        mlr = methods["multinomial_logistic_regression_rare_filtered"]
        self.assertEqual(mlr["model_family"], "multinomial_logistic_regression")
        self.assertEqual(
            mlr["display_label"], "Multinomial logistic regression (MLR)"
        )
        self.assertIn(
            "full model evaluated with single-site simulation",
            methods["full_model_single_site_simulation"]["notes"],
        )
        self.assertIn(
            "single-locus model evaluated with full-sites simulation",
            methods["single_locus_full_sites_simulation"]["notes"],
        )

    def test_sources_cover_every_legacy_csv_with_hashes_and_family_counts(self) -> None:
        self.assertEqual(self.source_header, SOURCE_FIELDS)
        self.assertEqual(len(self.sources), 76)
        self.assertEqual(
            Counter(row["source_family"] for row in self.sources),
            EXPECTED_FAMILY_COUNTS,
        )
        sources_by_path = {row["original_path"]: row for row in self.sources}
        self.assertEqual(
            set(sources_by_path),
            {
                path.relative_to(MIGRATION_SOURCE_ROOT).as_posix()
                for path in self.legacy_paths
            }
            | {f"tmp/{path.name}" for path in self.uncorrected_cv_paths}
            | {f"tmp/{path.name}" for path in self.corrected_cv_paths},
        )
        for path in self.legacy_paths:
            relative = path.relative_to(MIGRATION_SOURCE_ROOT).as_posix()
            source = sources_by_path[relative]
            self.assertEqual(source["sha256"], sha256(path))
            self.assertEqual(source["bytes"], str(path.stat().st_size))
            self.assertEqual(source["canonical_release"], "emd_v6")
            self.assertTrue(source["archived_path"].endswith(relative))
        for path in self.uncorrected_cv_paths:
            relative = f"tmp/{path.name}"
            source = sources_by_path[relative]
            self.assertEqual(source["sha256"], sha256(path))
            self.assertEqual(source["bytes"], str(path.stat().st_size))
            self.assertEqual(source["rows"], "97")
            self.assertEqual(source["canonical_release"], "emd_v6")
            self.assertEqual(
                source["source_family"],
                "results_9m_cross_validated_uncorrected",
            )
            self.assertTrue(source["archived_path"].endswith(relative))
        for path in self.corrected_cv_paths:
            relative = f"tmp/{path.name}"
            source = sources_by_path[relative]
            self.assertEqual(source["sha256"], sha256(path))
            self.assertEqual(source["bytes"], str(path.stat().st_size))
            self.assertEqual(source["rows"], "97")
            self.assertEqual(source["canonical_release"], "emd_v6")
            horizon = int(path.name.split("_", 2)[1].removesuffix("m"))
            self.assertEqual(
                source["source_family"],
                f"results_{horizon}m_cross_validated_corrected",
            )
            self.assertEqual(
                source["schema_id"],
                f"emd.results_cv_{horizon}m_corrected.v1",
            )
            self.assertEqual(
                source["upstream_version"],
                "cross_validated_lower_bound_horizon_pairing_corrected",
            )
            self.assertIn("2026-10-13", source["archived_path"])
            self.assertTrue(source["archived_path"].endswith(relative))

    def test_validation_only_and_historical_sources_do_not_supply_values(self) -> None:
        excluded_families = {
            "full_model_union_9m",
            "results_12m",
            "results_6m",
            "results_9m_pre_cv",
            "results_9m_cross_validated_uncorrected",
            "legacy_full_model",
            "legacy_since_2012",
        }
        excluded_ids = {
            row["source_id"]
            for row in self.sources
            if row["source_family"] in excluded_families
        }
        self.assertFalse(excluded_ids & {row["source_id"] for row in self.values})

    def test_only_corrected_cross_validated_sources_supply_current_results_grid_values(
        self,
    ) -> None:
        source_ids_by_family: dict[str, set[str]] = {}
        for source in self.sources:
            source_ids_by_family.setdefault(source["source_family"], set()).add(
                source["source_id"]
            )
        value_source_ids = {row["source_id"] for row in self.values}

        self.assertFalse(
            source_ids_by_family["results_9m_cross_validated_uncorrected"]
            & value_source_ids
        )
        for horizon in (6, 9, 12):
            family = f"results_{horizon}m_cross_validated_corrected"
            corrected_ids = source_ids_by_family[family]
            corrected_values = [
                row for row in self.values if row["source_id"] in corrected_ids
            ]
            self.assertEqual(len(corrected_values), 388 * 6)

    def test_conversion_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="emd-migration-repeat-") as directory:
            second_output = Path(directory) / "canonical"
            second_summary = migrate_emd(
                MIGRATION_SOURCE_ROOT,
                second_output,
                cv_9m_root=UNCORRECTED_CV_SOURCE_ROOT,
                corrected_cv_9m_root=CORRECTED_CV_SOURCE_ROOT,
                corrected_cv_6_12_root=CORRECTED_CV_SOURCE_ROOT,
                previous_corrected_cv_9m_root=(
                    PREVIOUS_CORRECTED_CV_9M_SOURCE_ROOT
                ),
                previous_corrected_cv_6_12_root=(
                    PREVIOUS_CORRECTED_CV_6_12_SOURCE_ROOT
                ),
            )
            self.assertEqual(self.summary.as_dict(), second_summary.as_dict())
            for filename in ("evaluations.csv", "values.csv", "methods.csv", "sources.csv"):
                self.assertEqual(
                    (self.output_dir / filename).read_bytes(),
                    (second_output / filename).read_bytes(),
                )

    def test_materialized_release_matches_archive_conversion_byte_for_byte(self) -> None:
        for filename in ("evaluations.csv", "values.csv", "methods.csv", "sources.csv"):
            materialized = CANONICAL_DIR / filename
            self.assertTrue(materialized.is_file(), f"Missing canonical EMD table: {materialized}")
            self.assertEqual(
                (self.output_dir / filename).read_bytes(),
                materialized.read_bytes(),
            )

    def test_existing_outputs_are_not_overwritten_without_opt_in(self) -> None:
        with self.assertRaisesRegex(
            MigrationError,
            "Refusing to overwrite canonical outputs without overwrite=True",
        ):
            migrate_emd(
                MIGRATION_SOURCE_ROOT,
                self.output_dir,
                cv_9m_root=UNCORRECTED_CV_SOURCE_ROOT,
                corrected_cv_9m_root=CORRECTED_CV_SOURCE_ROOT,
                corrected_cv_6_12_root=CORRECTED_CV_SOURCE_ROOT,
                previous_corrected_cv_9m_root=(
                    PREVIOUS_CORRECTED_CV_9M_SOURCE_ROOT
                ),
                previous_corrected_cv_6_12_root=(
                    PREVIOUS_CORRECTED_CV_6_12_SOURCE_ROOT
                ),
            )

    def test_legacy_inputs_are_unchanged(self) -> None:
        self.assertEqual(
            self.legacy_hashes_before,
            {path: sha256(path) for path in self.legacy_paths},
        )
        self.assertEqual(
            self.uncorrected_cv_hashes_before,
            {path: sha256(path) for path in self.uncorrected_cv_paths},
        )
        self.assertEqual(
            self.previous_corrected_cv_9m_hashes_before,
            {path: sha256(path) for path in self.previous_corrected_cv_9m_paths},
        )
        self.assertEqual(
            self.previous_corrected_cv_6_12_hashes_before,
            {
                path: sha256(path)
                for path in self.previous_corrected_cv_6_12_paths
            },
        )
        self.assertEqual(
            self.corrected_cv_hashes_before,
            {path: sha256(path) for path in self.corrected_cv_paths},
        )


if __name__ == "__main__":
    unittest.main()
