"""Tests for canonical, mutation-level background-fitness extraction."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from influenza_figures.background import (  # noqa: E402
    EVENT_KEY_FIELDS,
    MUTATION_KEY_FIELDS,
    BackgroundEventError,
    extract_background_events,
    extract_background_mutations,
)


CANONICAL_BACKGROUND = REPO / "data" / "plotting" / "inputs" / "background"

FIELDS = [
    "subtype",
    "segment",
    "source_variant",
    "position",
    "state",
    "start_time",
    "frequency",
    "initial_time",
    "final_time",
    "initial_fitness",
    "final_fitness",
    "fitness_change",
    "initial_background_fitness",
    "initial_population_fitness",
    "final_background_fitness",
    "background_fitness_change",
    "outcome",
    "prediction",
    "population_fitness",
    "selection_coefficient",
]


def observation(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {field: "" for field in FIELDS}
    row.update(
        {
            "subtype": "h3n2",
            "segment": "ha",
            "source_variant": "population_fitness",
            "position": "161",
            "state": "17",
            "start_time": 100,
            "frequency": 0.08,
            "initial_time": 90,
            "final_time": 180,
            "initial_background_fitness": 1.01,
            "initial_population_fitness": 1.00,
            "final_background_fitness": 1.02,
            "background_fitness_change": 0.01,
            "outcome": "fixed",
            "prediction": 0,
            # This is the legacy canonical name for selection coefficient.
            "population_fitness": 0.006,
        }
    )
    row.update(updates)
    return row


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class BackgroundEventTests(unittest.TestCase):
    def test_mutation_key_and_earliest_record_are_row_order_independent(self) -> None:
        rows = [
            observation(
                start_time=120,
                frequency=0.20,
                prediction=1,
                initial_background_fitness=1.03,
                final_background_fitness=1.04,
                background_fitness_change=0.01,
                population_fitness=0.009,
            ),
            observation(start_time=100, frequency=0.08, prediction=0),
            observation(
                position="8",
                state="4",
                start_time=50,
                initial_time=40,
                final_time=129,
                outcome="lost",
                population_fitness=-0.002,
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.csv"
            second = Path(directory) / "second.csv"
            write_rows(first, rows)
            write_rows(second, list(reversed(rows)))

            result = extract_background_events(first)
            reversed_result = extract_background_events(second)
            canonical_result = extract_background_mutations(first)

        expected_key = ("subtype", "segment", "position", "state")
        self.assertEqual(MUTATION_KEY_FIELDS, expected_key)
        self.assertEqual(EVENT_KEY_FIELDS, MUTATION_KEY_FIELDS)
        self.assertEqual(
            [dict(event) for event in result.events],
            [dict(event) for event in reversed_result.events],
        )
        self.assertEqual(result, canonical_result)
        self.assertEqual(result.input_row_count, 3)
        self.assertEqual(result.event_count_before_filter, 2)
        self.assertEqual(result.collapsed_observation_count, 1)
        repeated = next(event for event in result.events if event["position"] == "161")
        self.assertEqual(repeated["observation_count"], 2)
        self.assertEqual(repeated["start_times"], (100, 120))
        self.assertEqual(repeated["first_start_time"], 100)
        # Scientific values come from the earliest observation, even when the
        # CSV rows are reversed.
        self.assertEqual(repeated["initial_background_fitness"], 1.01)
        self.assertEqual(repeated["selection_coefficient"], 0.006)

    def test_duration_filter_precedes_earliest_eligible_record_selection(self) -> None:
        rows = [
            # This later observation is the earliest eligible record and must
            # represent the mutation after applying the 90-day cutoff.
            observation(
                start_time=120,
                initial_time=20,
                final_time=120,
                frequency=0.99,
                outcome="fixed",
                population_fitness=0.02,
            ),
            # The mutation's first appearance lasts only 89 days.  It remains
            # authoritative when no cutoff is requested, but it is ineligible
            # for the duration-filtered plot.
            observation(
                start_time=100,
                initial_time=10,
                final_time=99,
                frequency=0.01,
                outcome="lost",
                population_fitness=-0.01,
            ),
            observation(
                position="20",
                state="5",
                start_time=50,
                initial_time=10,
                final_time=100,
                outcome="fixed",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duration.csv"
            reversed_path = Path(directory) / "duration-reversed.csv"
            write_rows(path, rows)
            write_rows(reversed_path, list(reversed(rows)))
            unfiltered = extract_background_events(path)
            result = extract_background_events(path, minimum_duration_days=90)
            reversed_result = extract_background_events(
                reversed_path,
                minimum_duration_days=90,
            )

        self.assertEqual(
            [dict(event) for event in result.events],
            [dict(event) for event in reversed_result.events],
        )
        self.assertEqual(result.event_count_before_filter, 2)
        self.assertEqual(result.event_count, 2)
        self.assertEqual(result.excluded_by_duration_count, 0)
        self.assertEqual(
            dict(result.outcome_counts_before_filter), {"fixed": 1, "lost": 1}
        )
        self.assertEqual(dict(result.outcome_counts), {"fixed": 2})

        selected = next(
            event for event in result.events if event["position"] == "161"
        )
        self.assertEqual(selected["representative_start_time"], 120)
        self.assertEqual(selected["first_start_time"], 100)
        self.assertEqual(selected["start_times"], (100, 120))
        self.assertEqual(selected["eligible_start_times"], (120,))
        self.assertEqual(selected["observation_count"], 2)
        self.assertEqual(selected["eligible_observation_count"], 1)
        self.assertEqual(selected["duration_days"], 100)
        self.assertEqual(selected["outcome"], "fixed")
        self.assertEqual(selected["selection_coefficient"], 0.02)

        earliest = next(
            event for event in unfiltered.events if event["position"] == "161"
        )
        self.assertEqual(earliest["first_start_time"], 100)
        self.assertEqual(earliest["duration_days"], 89)
        self.assertEqual(earliest["outcome"], "lost")
        self.assertEqual(earliest["selection_coefficient"], -0.01)

    def test_initial_time_breaks_tied_start_times(self) -> None:
        rows = [
            observation(
                start_time=100,
                initial_time=90,
                final_time=180,
                outcome="fixed",
                population_fitness=0.006,
            ),
            observation(
                start_time=100,
                initial_time=80,
                final_time=170,
                outcome="lost",
                population_fitness=-0.003,
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.csv"
            second = Path(directory) / "second.csv"
            write_rows(first, rows)
            write_rows(second, list(reversed(rows)))
            result = extract_background_events(first)
            reversed_result = extract_background_events(second)

        self.assertEqual(
            [dict(event) for event in result.events],
            [dict(event) for event in reversed_result.events],
        )
        self.assertEqual(result.event_count, 1)
        self.assertEqual(result.events[0]["initial_time"], 80)
        self.assertEqual(result.events[0]["outcome"], "lost")
        self.assertEqual(result.events[0]["selection_coefficient"], -0.003)

    def test_conflicting_populated_aliases_are_rejected(self) -> None:
        rows = [
            observation(
                fitness_change=0.02,
                background_fitness_change=0.01,
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aliases.csv"
            write_rows(path, rows)
            with self.assertRaisesRegex(
                BackgroundEventError, "Conflicting aliases.*background_fitness_change"
            ):
                extract_background_events(path)

    def test_legacy_full_columns_normalize_to_background_fitness(self) -> None:
        rows = [
            observation(
                source_variant="full",
                initial_fitness=1.01,
                final_fitness=1.02,
                fitness_change=0.01,
                initial_background_fitness="",
                initial_population_fitness="",
                final_background_fitness="",
                background_fitness_change="",
                population_fitness="",
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "full.csv"
            write_rows(path, rows)
            result = extract_background_events(path)

        event = result.events[0]
        self.assertEqual(event["initial_background_fitness"], 1.01)
        self.assertEqual(event["final_background_fitness"], 1.02)
        self.assertEqual(event["background_fitness_change"], 0.01)
        self.assertIsNone(event["initial_population_fitness"])
        self.assertIsNone(event["selection_coefficient"])

    def test_canonical_population_table_has_expected_mutation_audit(self) -> None:
        result = extract_background_events(
            CANONICAL_BACKGROUND / "population_fitness.csv",
            minimum_duration_days=90,
        )

        self.assertEqual(result.input_row_count, 15_277)
        self.assertEqual(result.event_count_before_filter, 418)
        self.assertEqual(result.collapsed_observation_count, 14_859)
        self.assertEqual(result.event_count, 315)
        self.assertEqual(result.excluded_by_duration_count, 103)
        self.assertEqual(
            dict(result.outcome_counts_before_filter), {"fixed": 45, "lost": 373}
        )
        self.assertEqual(dict(result.outcome_counts), {"fixed": 54, "lost": 261})
        self.assertTrue(all(event["duration_days"] >= 90 for event in result.events))
        self.assertTrue(
            all(event["selection_coefficient"] is not None for event in result.events)
        )

    def test_canonical_full_table_has_expected_mutation_audit(self) -> None:
        result = extract_background_events(
            CANONICAL_BACKGROUND / "full.csv",
            minimum_duration_days=90,
        )

        self.assertEqual(result.input_row_count, 15_234)
        self.assertEqual(result.event_count_before_filter, 414)
        self.assertEqual(result.collapsed_observation_count, 14_820)
        self.assertEqual(result.event_count, 314)
        self.assertEqual(result.excluded_by_duration_count, 100)
        self.assertEqual(
            dict(result.outcome_counts_before_filter), {"fixed": 43, "lost": 371}
        )
        self.assertEqual(dict(result.outcome_counts), {"fixed": 54, "lost": 260})
        self.assertTrue(all(event["duration_days"] >= 90 for event in result.events))
        self.assertTrue(
            all(
                event["initial_background_fitness"] is not None
                and event["final_background_fitness"] is not None
                and event["background_fitness_change"] is not None
                for event in result.events
            )
        )

    def test_invalid_minimum_duration_is_rejected(self) -> None:
        with self.assertRaisesRegex(BackgroundEventError, "non-negative number"):
            extract_background_events(
                CANONICAL_BACKGROUND / "population_fitness.csv",
                minimum_duration_days=-1,
            )


if __name__ == "__main__":
    unittest.main()
