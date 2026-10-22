"""Tests for exact canonical EMD pairing."""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from influenza_figures.emd import (
    SUBTYPE_INITIAL_INFERENCE_DATES,
    EmdComparisonError,
    compare_emd,
)


EMD = REPO / "data" / "plotting" / "inputs" / "emd"
FULL = "full_model_rare_filtered"
CV_FULL = "full_model_cross_validated_rare_filtered"
SLM = "single_locus_model_rare_filtered"
MLR = "multinomial_logistic_regression_rare_filtered"
DEPRECATED_MLR = "multilinear_regression_rare_filtered"


class EmdComparisonTests(unittest.TestCase):
    def test_repository_initial_inference_dates_are_exact(self) -> None:
        self.assertEqual(
            dict(SUBTYPE_INITIAL_INFERENCE_DATES),
            {"h3n2": "2009-01-02", "h1n1": "2009-05-02"},
        )
        self.assertEqual(
            date.fromisoformat(SUBTYPE_INITIAL_INFERENCE_DATES["h3n2"])
            + timedelta(days=1095),
            date(2012, 1, 2),
        )
        self.assertEqual(
            date.fromisoformat(SUBTYPE_INITIAL_INFERENCE_DATES["h1n1"])
            + timedelta(days=1095),
            date(2012, 5, 1),
        )

    def test_full_vs_mlr_pairs_every_full_grid_evaluation(self) -> None:
        result = compare_emd(
            EMD,
            target_method=FULL,
            comparator_method=MLR,
            evaluation_protocol="full_grid",
        )
        self.assertEqual(result.protocols, ("full_grid",))
        self.assertEqual(result.target_count, 1427)
        self.assertEqual(result.comparator_count, 1427)
        self.assertEqual(result.paired_count, 1427)
        self.assertFalse(result.unmatched_target_ids)
        self.assertFalse(result.unmatched_comparator_ids)

    def test_deprecated_mlr_identifier_resolves_to_correct_canonical_method(self) -> None:
        with self.assertWarnsRegex(
            DeprecationWarning, "multinomial logistic regression"
        ):
            result = compare_emd(
                EMD,
                target_method=FULL,
                comparator_method=DEPRECATED_MLR,
                evaluation_protocol="results_grid",
            )
        self.assertEqual(result.comparator_method, MLR)
        self.assertEqual(result.paired_count, 1164)

    def test_full_vs_slm_reports_unmatched_full_grid_evaluations(self) -> None:
        result = compare_emd(
            EMD,
            target_method=FULL,
            comparator_method=SLM,
            evaluation_protocol="full_grid",
        )
        self.assertEqual(result.target_count, 1427)
        self.assertEqual(result.comparator_count, 344)
        self.assertEqual(result.paired_count, 344)
        self.assertEqual(len(result.unmatched_target_ids), 1083)

    def test_results_grid_compares_null_slm_and_mlr_separately(self) -> None:
        for comparator in ("null", SLM, MLR):
            with self.subTest(comparator=comparator):
                result = compare_emd(
                    EMD,
                    target_method=FULL,
                    comparator_method=comparator,
                    evaluation_protocol="results_grid",
                )
                self.assertEqual(result.comparator_method, comparator)
                self.assertEqual(result.paired_count, 1164)
                self.assertEqual(
                    len({row["evaluation_id"] for row in result.rows}),
                    result.paired_count,
                )

    def test_legacy_diagnostic_pair_is_supported_without_promoting_it(self) -> None:
        result = compare_emd(
            EMD,
            target_method="full_model_single_site_simulation",
            comparator_method="single_locus_full_sites_simulation",
            evaluation_protocol="full_grid",
        )
        self.assertEqual(result.paired_count, 344)

    def test_cross_validated_models_pair_with_baseline_at_every_horizon(self) -> None:
        for horizon_months in (6, 9, 12):
            with self.subTest(horizon_months=horizon_months):
                uncut = compare_emd(
                    EMD,
                    target_method=CV_FULL,
                    comparator_method=FULL,
                    evaluation_protocol="results_grid",
                    horizon_months=horizon_months,
                )
                self.assertEqual(uncut.paired_count, 388)
                self.assertFalse(uncut.unmatched_target_ids)
                self.assertFalse(uncut.unmatched_comparator_ids)
                first = uncut.rows[0]
                self.assertAlmostEqual(
                    float(first["delta_emd"]),
                    float(first["comparator_emd"])
                    - float(first["target_emd"]),
                )

                after_warmup = compare_emd(
                    EMD,
                    target_method=CV_FULL,
                    comparator_method=FULL,
                    evaluation_protocol="results_grid",
                    horizon_months=horizon_months,
                    minimum_observation_day=1095,
                )
                self.assertEqual(after_warmup.paired_count, 330)

    def test_comparison_requires_an_explicit_evaluation_protocol(self) -> None:
        with self.assertRaisesRegex(TypeError, "evaluation_protocol"):
            compare_emd(EMD, target_method=FULL, comparator_method=MLR)

    def test_comparison_rejects_an_unknown_evaluation_protocol(self) -> None:
        with self.assertRaisesRegex(
            EmdComparisonError, "Unknown EMD evaluation protocol"
        ):
            compare_emd(
                EMD,
                target_method=FULL,
                comparator_method=MLR,
                evaluation_protocol="nearest_dates",
            )

    def test_delta_sign_is_comparator_minus_target(self) -> None:
        result = compare_emd(
            EMD,
            target_method=FULL,
            comparator_method="null",
            evaluation_protocol="results_grid",
            horizon_months=6,
            subtype="h3n2",
            segment="ha",
        )
        self.assertGreater(result.paired_count, 0)
        first = result.rows[0]
        self.assertAlmostEqual(
            float(first["delta_emd"]),
            float(first["comparator_emd"]) - float(first["target_emd"]),
        )

    def test_results_cutoff_uses_recorded_model_start_date_inclusively(self) -> None:
        h3 = compare_emd(
            EMD,
            target_method=FULL,
            comparator_method=MLR,
            evaluation_protocol="results_grid",
            horizon_months=6,
            subtype="h3n2",
            segment="ha",
            minimum_observation_day=1095,
        )
        self.assertEqual(h3.paired_count, 84)
        self.assertEqual(len(h3.cutoffs), 1)
        self.assertEqual(h3.cutoffs[0].initial_inference_date, "2009-01-02")
        self.assertEqual(h3.cutoffs[0].relative_day, 1095)
        self.assertEqual(h3.cutoffs[0].cutoff_date, "2012-01-02")
        self.assertTrue(h3.cutoffs[0].inclusive)
        self.assertEqual(h3.cutoffs[0].source, "evaluation model_start_date")
        self.assertGreaterEqual(h3.rows[0]["observation_date"], "2012-01-02")

        h1 = compare_emd(
            EMD,
            target_method=FULL,
            comparator_method=MLR,
            evaluation_protocol="results_grid",
            horizon_months=6,
            subtype="h1n1",
            segment="ha",
            minimum_observation_day=1095,
        )
        self.assertEqual(h1.paired_count, 81)
        self.assertEqual(h1.cutoffs[0].cutoff_date, "2012-05-01")
        self.assertEqual(h1.rows[0]["observation_date"], "2012-05-01")
        self.assertIn("initial inference 2009-05-02 + 1095 days", h1.describe_cutoff())
        self.assertIn("on or after 2012-05-01", h1.describe_cutoff())

    def test_full_grid_cutoff_uses_repository_subtype_default(self) -> None:
        result = compare_emd(
            EMD,
            target_method=FULL,
            comparator_method=MLR,
            evaluation_protocol="full_grid",
            horizon_months=6,
            subtype="h1n1",
            segment="ha",
            minimum_observation_day=1095,
        )
        self.assertEqual(result.paired_count, 83)
        self.assertEqual(result.cutoffs[0].initial_inference_date, "2009-05-02")
        self.assertEqual(result.cutoffs[0].cutoff_date, "2012-05-01")
        self.assertEqual(result.cutoffs[0].source, "repository subtype default")

    def test_cutoff_reports_each_subtype_without_manual_offsets(self) -> None:
        result = compare_emd(
            EMD,
            target_method=FULL,
            comparator_method=SLM,
            evaluation_protocol="results_grid",
            minimum_observation_day=1095,
        )
        self.assertEqual(result.paired_count, 990)
        self.assertEqual(
            {(item.subtype, item.cutoff_date) for item in result.cutoffs},
            {("h3n2", "2012-01-02"), ("h1n1", "2012-05-01")},
        )
        self.assertNotIn("offset", result.describe_cutoff().lower())

    def test_caller_can_override_ambiguous_legacy_full_grid_origin(self) -> None:
        result = compare_emd(
            EMD,
            target_method=FULL,
            comparator_method=MLR,
            evaluation_protocol="full_grid",
            horizon_months=6,
            subtype="h3n2",
            segment="ha",
            minimum_observation_day=1095,
            initial_inference_dates={"H3N2": "2009-01-01"},
        )
        self.assertEqual(result.cutoffs[0].initial_inference_date, "2009-01-01")
        self.assertEqual(result.cutoffs[0].cutoff_date, "2012-01-01")
        self.assertEqual(result.cutoffs[0].source, "caller override")

    def test_comparison_tolerates_canonical_inputs_already_trimmed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trimmed = Path(temporary)
            shutil.copyfile(EMD / "methods.csv", trimmed / "methods.csv")

            with (EMD / "evaluations.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                reader = csv.DictReader(handle)
                evaluation_fields = reader.fieldnames
                evaluations = [
                    row
                    for row in reader
                    if row["evaluation_protocol"] == "results_grid"
                    and row["horizon_months"] == "6"
                    and row["subtype"] == "h1n1"
                    and row["segment"] == "ha"
                    and row["observation_date"] >= "2012-05-01"
                ]
            self.assertIsNotNone(evaluation_fields)
            for row in evaluations:
                # The comparison must be driven by ISO dates, not a universal
                # day counter inherited from a legacy time origin.
                row["observation_day_2009"] = "0"
            with (trimmed / "evaluations.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=evaluation_fields)
                writer.writeheader()
                writer.writerows(evaluations)

            evaluation_ids = {row["evaluation_id"] for row in evaluations}
            with (EMD / "values.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                reader = csv.DictReader(handle)
                value_fields = reader.fieldnames
                values = [
                    row
                    for row in reader
                    if row["evaluation_id"] in evaluation_ids
                    and row["method_id"] in {FULL, MLR}
                ]
            self.assertIsNotNone(value_fields)
            with (trimmed / "values.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=value_fields)
                writer.writeheader()
                writer.writerows(values)

            result = compare_emd(
                trimmed,
                target_method=FULL,
                comparator_method=MLR,
                evaluation_protocol="results_grid",
                horizon_months=6,
                subtype="h1n1",
                segment="ha",
                minimum_observation_day=1095,
            )
            self.assertEqual(result.paired_count, len(evaluations))
            self.assertEqual(result.rows[0]["observation_date"], "2012-05-01")

    def test_negative_relative_cutoff_is_rejected(self) -> None:
        with self.assertRaisesRegex(EmdComparisonError, "must be nonnegative"):
            compare_emd(
                EMD,
                target_method=FULL,
                comparator_method=MLR,
                evaluation_protocol="results_grid",
                minimum_observation_day=-1,
            )

    def test_no_cutoff_has_an_explicit_description(self) -> None:
        result = compare_emd(
            EMD,
            target_method=FULL,
            comparator_method=MLR,
            evaluation_protocol="results_grid",
        )
        self.assertEqual(result.cutoffs, ())
        self.assertEqual(result.describe_cutoff(), "No observation-date cutoff applied.")


if __name__ == "__main__":
    unittest.main()
