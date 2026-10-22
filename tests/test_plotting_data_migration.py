"""Integration checks for the non-EMD canonical plotting-data migration."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "archive" / "pre_plot_reorg_2026-08-26" / "original"
CANONICAL = REPO / "data" / "plotting"
SCRIPT = REPO / "scripts" / "migrate_plotting_data.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class PlottingDataMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("migrate_plotting_data", SCRIPT)
        assert spec is not None and spec.loader is not None
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        spec.loader.exec_module(cls.module)

    def test_every_recorded_source_hash_matches_the_verified_archive(self) -> None:
        rows = csv_rows(CANONICAL / "migration_map.csv")
        self.assertEqual(len(rows), 4131)
        for row in rows:
            source = ARCHIVE / row["original_path"]
            self.assertTrue(source.is_file(), row["original_path"])
            self.assertEqual(sha256(source), row["source_sha256"], row["original_path"])

    def test_selection_schema_uses_canonical_names_and_integer_counts(self) -> None:
        for path in sorted((CANONICAL / "inputs" / "selection").glob("*_mutations.csv")):
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertIn("mutation_count", reader.fieldnames or [])
                self.assertNotIn("MutationFrequency", reader.fieldnames or [])
                for row in reader:
                    self.assertNotRegex(row["mutation_count"], r"\.0$")

    def test_background_variants_remain_separate_and_complete(self) -> None:
        full = csv_rows(CANONICAL / "inputs" / "background" / "full.csv")
        population = csv_rows(
            CANONICAL / "inputs" / "background" / "population_fitness.csv"
        )
        self.assertEqual(len(full), 15234)
        self.assertEqual(len(population), 15277)
        self.assertEqual({row["source_variant"] for row in full}, {"full"})
        self.assertEqual(
            {row["source_variant"] for row in population}, {"population_fitness"}
        )

    def test_background_schema_uses_scientific_quantity_names(self) -> None:
        full = csv_rows(CANONICAL / "inputs" / "background" / "full.csv")
        population = csv_rows(
            CANONICAL / "inputs" / "background" / "population_fitness.csv"
        )
        expected_fields = {
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
        }
        self.assertEqual(set(full[0]), expected_fields)
        self.assertEqual(set(population[0]), expected_fields)
        self.assertNotIn("population_fitness", full[0])
        self.assertNotIn("initial_fitness", full[0])
        self.assertTrue(all(row["initial_background_fitness"] for row in full))
        self.assertTrue(all(row["selection_coefficient"] for row in population))
        self.assertTrue(all(not row["selection_coefficient"] for row in full))
        for row in full + population:
            self.assertEqual(
                int(row["duration_days"]),
                int(row["final_time"]) - int(row["initial_time"]),
            )

    def test_shared_background_observations_use_identical_canonical_fields(self) -> None:
        full = csv_rows(CANONICAL / "inputs" / "background" / "full.csv")
        population = csv_rows(
            CANONICAL / "inputs" / "background" / "population_fitness.csv"
        )
        identity = (
            "subtype",
            "segment",
            "position",
            "state",
            "start_time",
            "initial_time",
            "final_time",
            "outcome",
            "prediction",
        )
        background_fields = (
            "initial_background_fitness",
            "final_background_fitness",
            "background_fitness_change",
        )
        population_by_identity = {
            tuple(row[field] for field in identity): row for row in population
        }
        shared = 0
        for row in full:
            match = population_by_identity.get(tuple(row[field] for field in identity))
            if match is None:
                continue
            shared += 1
            self.assertEqual(
                tuple(row[field] for field in background_fields),
                tuple(match[field] for field in background_fields),
            )
        self.assertEqual(shared, 15012)

    def test_active_fixation_data_excludes_retired_diagnostics(self) -> None:
        predictions = csv_rows(CANONICAL / "inputs" / "fixation" / "predictions.csv")
        auc = csv_rows(CANONICAL / "derived" / "fixation_metrics" / "auc.csv")
        self.assertEqual({row["model_id"] for row in predictions}, {"full_model", "slm"})
        self.assertEqual({row["model_id"] for row in auc}, {"full_model", "slm"})
        self.assertEqual(
            {row["source_variant"] for row in predictions if row["model_id"] == "slm"},
            {"paper_date_filtered", "extended_all_time"},
        )
        self.assertEqual(
            {row["source_variant"] for row in auc if row["model_id"] == "slm"},
            {"paper_date_filtered", "extended_all_time"},
        )

    def test_example_bundles_are_complete_and_readable(self) -> None:
        for number in (1, 2):
            path = CANONICAL / "inputs" / "examples" / f"fix_loss_example_{number}.zip"
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(len(archive.namelist()), 2007)
                self.assertIsNone(archive.testzip())

    def test_clade_dates_are_iso_formatted(self) -> None:
        rows = csv_rows(CANONICAL / "inputs" / "clades" / "h3n2_ha_frequencies.csv")
        self.assertEqual(len(rows), 4788)
        self.assertTrue(all(len(row["date"]) == 10 and row["date"][4] == "-" for row in rows))

    def test_conversion_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plotting"
            self.module.build(ARCHIVE, output)
            expected = {
                path.relative_to(CANONICAL): sha256(path)
                for path in CANONICAL.rglob("*")
                if path.is_file()
                and "emd" not in path.parts
                and path.name != "README.md"
            }
            actual = {
                path.relative_to(output): sha256(path)
                for path in output.rglob("*")
                if path.is_file()
            }
            self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
