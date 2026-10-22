from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from validate_figure_manifest import (  # noqa: E402
    ARCHIVE_PREFIX,
    ARCHIVE_PREFIXES,
    CANONICAL_DATA_PREFIX,
    PURPOSE_OUTPUT_PREFIX,
    REQUIRED_ACTIVITY_STATES,
    expand_path_spec,
    load_manifest,
    validate_manifest,
)
from influenza_figures.registry import CANVASES, FIGURE_REGISTRY  # noqa: E402
from influenza_figures.styles import STYLES  # noqa: E402


class FigureManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_path = REPO_ROOT / "figure_manifest.yaml"
        cls.manifest = load_manifest(cls.manifest_path)

    def test_repository_manifest_is_valid(self) -> None:
        self.assertEqual(validate_manifest(self.manifest), [])

    def test_declared_present_inputs_exist(self) -> None:
        self.assertEqual(
            validate_manifest(self.manifest, repo_root=REPO_ROOT, check_paths=True),
            [],
        )

    def test_all_conservative_activity_states_are_represented(self) -> None:
        observed = {figure["activity_status"] for figure in self.manifest["figures"]}
        self.assertEqual(observed, REQUIRED_ACTIVITY_STATES)

    def test_driver_and_archived_cell_provenance_are_distinct(self) -> None:
        scope = self.manifest["scope"]
        self.assertEqual(scope["driver_notebook"], "figures.ipynb")
        self.assertEqual(
            scope["notebook_cell_provenance"],
            "archive/pre_plot_reorg_2026-08-26/original/figures.ipynb",
        )
        self.assertEqual(scope["cell_numbering"], "zero_based_in_archived_original")
        for figure in self.manifest["figures"]:
            self.assertEqual(
                figure["notebook"]["path"], scope["notebook_cell_provenance"]
            )

    def test_method_provenance_policy_is_explicit(self) -> None:
        semantics = self.manifest["method_semantics"]
        for method in ("genmodel", "slm", "mlr"):
            self.assertEqual(semantics[method]["selection_filter"], "rare_filtered")
        for method in ("fmss", "slfs"):
            self.assertEqual(semantics[method]["lifecycle"], "legacy_diagnostic")
        self.assertEqual(
            semantics["best-genmodel"]["hyperparameter_policy"],
            "selected_by_cross_validation",
        )
        self.assertEqual(semantics["best-genmodel"]["lifecycle"], "active")

    def test_canonical_emd_v6_release_metadata_is_explicit(self) -> None:
        canonical = self.manifest["input_sets"]["emd.canonical"]
        self.assertEqual(canonical["schema"], "emd_normalized_v6")
        self.assertEqual(
            canonical["verification"],
            {
                "release": "emd_v6",
                "evaluations": 2591,
                "values": 16578,
                "methods": 10,
                "sources": 76,
            },
        )
        self.assertEqual(
            {
                artifact["path"]: (artifact["sha256"], artifact["bytes"])
                for artifact in canonical["artifacts"]
            },
            {
                "data/plotting/inputs/emd/evaluations.csv": (
                    "339c0111b221ae558abb4085af4d166d45ae1a4c86e2c58d79a404717487d14b",
                    252124,
                ),
                "data/plotting/inputs/emd/values.csv": (
                    "69d2c3e99f4daeea014fa0a9ee2a6123ebb3b7df50fc2d23c25d084e0a09c1b3",
                    2470291,
                ),
                "data/plotting/inputs/emd/methods.csv": (
                    "27bdcb2e64fd3bfb7663a42aec33be76e05eca7126adb3098c39efc2ba5dc7de",
                    1835,
                ),
                "data/plotting/inputs/emd/sources.csv": (
                    "8ef452f8b31e3ef119934dfd125dcebd3727a62796ad291bb7c85b0135d27243",
                    29901,
                ),
            },
        )
        self.assertIn("October 13", canonical["note"])
        self.assertIn("zero", canonical["note"].lower())
        self.assertIn("120 source rows", canonical["note"])

    def test_background_inputs_use_schema_v2(self) -> None:
        inputs = self.manifest["input_sets"]
        for input_id in ("background.population-fitness", "background.delta-h"):
            with self.subTest(input_id=input_id):
                self.assertEqual(inputs[input_id]["schema"], "background_fitness_v2")

    def test_paper_emd_recipes_are_explicit_and_registry_owned(self) -> None:
        expected_comparators = {
            "paper.emd-vs-null": ("null", "null"),
            "paper.emd-vs-slm": (
                "slm",
                "single_locus_model_rare_filtered",
            ),
            "paper.emd-vs-mlr": (
                "mlr",
                "multinomial_logistic_regression_rare_filtered",
            ),
        }
        expected_protocols = {
            "6": "results_grid",
            "9": "results_grid",
            "12": "results_grid",
        }
        expected_initial_dates = {
            "h3n2": "2009-01-02",
            "h1n1": "2009-05-02",
        }
        expected_cutoff_dates = {
            "h3n2": "2012-01-02",
            "h1n1": "2012-05-01",
        }
        figures = {
            figure["figure_id"]: figure for figure in self.manifest["figures"]
        }

        for figure_id, (short_name, comparator) in expected_comparators.items():
            with self.subTest(figure_id=figure_id):
                figure = figures[figure_id]
                self.assertEqual(figure["activity_status"], "active")
                self.assertIn(
                    figure["run_state"],
                    {"runnable_unverified", "verified_clean"},
                )
                self.assertEqual(
                    figure["render_revision"],
                    "significance_annotation_v2_date_axis_padding_50d",
                )
                self.assertEqual(
                    figure["implementation"]["entrypoint"],
                    "influenza_figures.emd_plot.render_emd_comparison_figure",
                )

                transformations = {
                    item["operation"]: item for item in figure["transformations"]
                }
                cutoff = transformations[
                    "inference_relative_observation_date_cutoff"
                ]
                self.assertEqual(cutoff["relative_day"], 1095)
                self.assertIs(cutoff["inclusive"], True)
                self.assertEqual(cutoff["initial_dates"], expected_initial_dates)
                self.assertEqual(cutoff["cutoff_dates"], expected_cutoff_dates)
                self.assertIs(cutoff["print_report"], True)
                self.assertEqual(
                    transformations["evaluation_protocol_by_horizon"]["values"],
                    expected_protocols,
                )
                self.assertIn(
                    "exact_evaluation_id_pairing",
                    transformations,
                )
                self.assertEqual(
                    transformations["date_axis_padding"],
                    {
                        "operation": "date_axis_padding",
                        "left_days": 50,
                        "right_days": 50,
                        "tick_dates_unchanged": True,
                    },
                )
                self.assertEqual(
                    transformations["significance_annotation"],
                    {
                        "operation": "significance_annotation",
                        "test": "paired_t_test",
                        "alpha": 0.05,
                        "delta_definition": "comparator_emd_minus_target_emd",
                        "display_policy": {
                            "p_gt_0_05": "omit",
                            "p_gt_0_01_and_lte_0_05": "exact_two_decimal",
                            "p_lte_0_01": "order_of_magnitude",
                        },
                        "target_worse_marker": {
                            "condition": "p_value <= 0.05 and mean_delta_emd < 0",
                            "marker": "*",
                            "figure_key": "* full model worse",
                        },
                    },
                )
                self.assertEqual(
                    transformations["comparison"],
                    {
                        "operation": "comparison",
                        "model_by_horizon": {
                            "6": "full_model_cross_validated_rare_filtered",
                            "9": "full_model_cross_validated_rare_filtered",
                            "12": "full_model_cross_validated_rare_filtered",
                        },
                        "reference": comparator,
                    },
                )

                expected_registry_keys = {
                    f"paper.emd_vs_{short_name}.{horizon}m"
                    for horizon in (6, 9, 12)
                }
                registry_keys = set(figure["implementation"]["registry_keys"])
                self.assertEqual(registry_keys, expected_registry_keys)

                manifest_outputs = {
                    path.removeprefix("figures/")
                    for output in figure["outputs"]
                    for path in expand_path_spec(output)
                }
                registry_outputs = {
                    FIGURE_REGISTRY.get(key).output_name for key in registry_keys
                }
                self.assertEqual(registry_outputs, manifest_outputs)
                self.assertEqual(
                    {
                        (
                            FIGURE_REGISTRY.get(key).recipe_parameters[
                                "horizon_months"
                            ],
                            FIGURE_REGISTRY.get(key).recipe_parameters[
                                "comparator_method"
                            ],
                        )
                        for key in registry_keys
                    },
                    {(horizon, comparator) for horizon in (6, 9, 12)},
                )
                for key in registry_keys:
                    self.assertEqual(
                        FIGURE_REGISTRY.get(key).renderer,
                        "render_emd_comparison_figure",
                    )
                    self.assertEqual(
                        FIGURE_REGISTRY.get(key).layout,
                        "four_by_two_trajectory_and_distribution",
                    )

                if figure["run_state"] == "verified_clean":
                    v6_instances = {
                        item["horizon"]: item
                        for output in figure["outputs"]
                        for item in output.get("verified_instances", [])
                        if item["release"] == "emd_v6"
                        and item["render_revision"] == figure["render_revision"]
                    }
                    self.assertEqual(set(v6_instances), {"6m", "9m", "12m"})
                    self.assertTrue(
                        all(
                            item["visual_qa"] == {
                                "status": "passed",
                                "clean_render": True,
                                "no_clipping": True,
                                "paper_style_intact": True,
                            }
                            for item in v6_instances.values()
                        )
                    )

        slm = figures["paper.emd-vs-slm"]
        self.assertTrue(
            all("legacy_artifact" not in output for output in slm["outputs"])
        )

    def test_final_manuscript_labels_map_to_canonical_outputs(self) -> None:
        figures = {
            figure["figure_id"]: figure for figure in self.manifest["figures"]
        }
        self.assertEqual(
            {
                figure_id: figures[figure_id].get("manuscript_labels")
                for figure_id in (
                    "paper.model-overview",
                    "paper.selection-structure-space",
                    "paper.fixation-probability-vs-auc",
                    "paper.background-clade",
                    "paper.frequency-selection",
                    "paper.selection-sasa",
                    "paper.fixation-auc",
                    "paper.single-site-auc",
                )
            },
            {
                "paper.model-overview": ["Figure 1"],
                "paper.selection-structure-space": ["Figure 2"],
                "paper.fixation-probability-vs-auc": ["Figure 3"],
                "paper.background-clade": ["Figure 5"],
                "paper.frequency-selection": ["Supplementary Figure S5"],
                "paper.selection-sasa": ["Supplementary Figure S6"],
                "paper.fixation-auc": ["Supplementary Figure S7"],
                "paper.single-site-auc": ["Supplementary Figure S16"],
            },
        )
        self.assertEqual(
            figures["paper.emd-vs-null"]["manuscript_labels_by_horizon"],
            {
                "6": "Supplementary Figure S8",
                "9": "Figure 4",
                "12": "Supplementary Figure S9",
            },
        )
        self.assertEqual(
            figures["paper.emd-vs-mlr"]["manuscript_labels_by_horizon"],
            {
                "6": "Supplementary Figure S10",
                "9": "Supplementary Figure S11",
                "12": "Supplementary Figure S12",
            },
        )
        self.assertEqual(
            figures["paper.emd-vs-slm"]["manuscript_labels_by_horizon"],
            {
                "6": "Supplementary Figure S17",
                "9": "Supplementary Figure S18",
                "12": "Supplementary Figure S19",
            },
        )
        self.assertEqual(
            figures["paper.background-components"]["manuscript_labels_by_output"],
            {
                "figures/supplement/fig-ds.pdf": "Supplementary Figure S13",
                "figures/supplement/fig-dw-bg.pdf": "Supplementary Figure S14",
                "figures/supplement/fig-dw-rate.pdf": "Supplementary Figure S15",
            },
        )
        labels = []
        for figure in figures.values():
            labels.extend(figure.get("manuscript_labels", []))
            labels.extend(figure.get("manuscript_labels_by_horizon", {}).values())
            labels.extend(figure.get("manuscript_labels_by_output", {}).values())
        self.assertEqual(len(labels), 20)
        self.assertEqual(
            set(labels),
            {
                *(f"Figure {number}" for number in range(1, 6)),
                *(f"Supplementary Figure S{number}" for number in range(5, 20)),
            },
        )

    def test_best_vs_baseline_emd_diagnostic_is_explicit_and_registry_owned(self) -> None:
        figure = next(
            item
            for item in self.manifest["figures"]
            if item["figure_id"] == "diagnostic.emd-best-vs-baseline"
        )
        registry_keys = {
            f"diagnostic.emd_best_vs_baseline.{horizon}m"
            for horizon in (6, 9, 12)
        }

        self.assertEqual(figure["purpose"], "diagnostic")
        self.assertEqual(figure["activity_status"], "active")
        self.assertIn(
            figure["run_state"],
            {"runnable_unverified", "verified_clean"},
        )
        self.assertEqual(
            figure["render_revision"],
            "significance_annotation_v2_date_axis_padding_50d",
        )
        self.assertEqual(
            set(figure["implementation"]["registry_keys"]),
            registry_keys,
        )
        self.assertEqual(
            {FIGURE_REGISTRY.get(key).output_name for key in registry_keys},
            {
                path.removeprefix("figures/")
                for path in expand_path_spec(figure["outputs"][0])
            },
        )

        transformations = {
            item["operation"]: item for item in figure["transformations"]
        }
        self.assertEqual(
            transformations["date_axis_padding"],
            {
                "operation": "date_axis_padding",
                "left_days": 50,
                "right_days": 50,
                "tick_dates_unchanged": True,
            },
        )
        self.assertEqual(
            transformations["evaluation_protocol_by_horizon"]["values"],
            {"6": "results_grid", "9": "results_grid", "12": "results_grid"},
        )
        self.assertEqual(
            transformations["significance_annotation"]["display_policy"],
            {
                "p_gt_0_05": "omit",
                "p_gt_0_01_and_lte_0_05": "exact_two_decimal",
                "p_lte_0_01": "order_of_magnitude",
            },
        )
        self.assertEqual(
            transformations["significance_annotation"]["target_worse_marker"],
            {
                "condition": "p_value <= 0.05 and mean_delta_emd < 0",
                "marker": "*",
                "figure_key": "* target model worse",
            },
        )
        self.assertEqual(
            transformations["comparison"],
            {
                "operation": "comparison",
                "target": "full_model_cross_validated_rare_filtered",
                "reference": "full_model_rare_filtered",
                "difference_sign": "reference_minus_target",
                "interpretation": (
                    "Positive values mean the cross-validation-selected best GenModel "
                    "has lower EMD than the same-run GenModel baseline."
                ),
            },
        )
        for horizon in (6, 9, 12):
            spec = FIGURE_REGISTRY.get(
                f"diagnostic.emd_best_vs_baseline.{horizon}m"
            )
            self.assertEqual(
                dict(spec.recipe_parameters),
                {
                    "horizon_months": horizon,
                    "target_method": "full_model_cross_validated_rare_filtered",
                    "comparator_method": "full_model_rare_filtered",
                    "difference_sign": "comparator_minus_target",
                },
            )
        if figure["run_state"] == "verified_clean":
            v6_instances = {
                item["horizon"]: item
                for item in figure["outputs"][0]["verified_instances"]
                if item["release"] == "emd_v6"
                and item["render_revision"] == figure["render_revision"]
            }
            self.assertEqual(set(v6_instances), {"6m", "9m", "12m"})
            self.assertTrue(
                all(
                    item["visual_qa"]["no_clipping"]
                    for item in v6_instances.values()
                )
            )

    def test_live_emd_recipes_do_not_use_wrong_or_generic_registry_keys(self) -> None:
        forbidden_registry_keys = {
            "paper.emd",
            "paper.emd_vs_ss",
            "paper.emd_vs_null",
            "paper.emd_vs_slm",
            "paper.emd_vs_mlr",
        }
        registered_keys = {spec.key for spec in FIGURE_REGISTRY}
        self.assertTrue(forbidden_registry_keys.isdisjoint(registered_keys))

        live_figures = [
            figure
            for figure in self.manifest["figures"]
            if figure["activity_status"] in {"active", "candidate"}
        ]
        self.assertNotIn(
            "blocker.paper.emd-vs-single-site",
            {figure["figure_id"] for figure in live_figures},
        )
        for figure in live_figures:
            with self.subTest(figure_id=figure["figure_id"]):
                self.assertNotIn("emd-vs-ss", figure["figure_id"])
                self.assertTrue(
                    forbidden_registry_keys.isdisjoint(
                        figure["implementation"].get("registry_keys", [])
                    )
                )
                canonical_outputs = [
                    path
                    for output in figure["outputs"]
                    for path in expand_path_spec(output)
                ]
                self.assertTrue(
                    all("emd-vs-ss" not in path for path in canonical_outputs)
                )

    def test_legacy_multi_output_emd_slides_are_not_marked_corrected(self) -> None:
        figure = next(
            item
            for item in self.manifest["figures"]
            if item["figure_id"] == "slides.emd-vs-mlr"
        )
        self.assertEqual(figure["activity_status"], "candidate")
        self.assertIn("figures.plot_emd_slides", figure["implementation"]["entrypoint"])
        self.assertTrue(
            any(
                "strict Obs_Time > t_min" in blocker
                and "canonical slide wrapper" in blocker
                for blocker in figure["blockers"]
            )
        )

    def test_live_clade_recipes_filter_before_normalizing(self) -> None:
        clade_figures = [
            figure
            for figure in self.manifest["figures"]
            if figure["activity_status"] in {"active", "candidate"}
            and any(
                reference.get("input_set") == "clades.tidy"
                for reference in figure["inputs"]
            )
        ]
        self.assertEqual(
            {figure["figure_id"] for figure in clade_figures},
            {"paper.clade", "paper.background-clade", "paper.emd-clade"},
        )

        for figure in clade_figures:
            with self.subTest(figure_id=figure["figure_id"]):
                operations = [
                    item["operation"] for item in figure["transformations"]
                ]
                self.assertLess(
                    operations.index("select_clade_reference"),
                    operations.index("select_clades_by_total_frequency"),
                )
                self.assertLess(
                    operations.index("select_clades_by_total_frequency"),
                    operations.index("filter_all_series_to_selected_clades"),
                )
                self.assertLess(
                    operations.index("filter_all_series_to_selected_clades"),
                    operations.index("normalize_frequency"),
                )

                transformations = {
                    item["operation"]: item for item in figure["transformations"]
                }
                self.assertEqual(
                    transformations["select_clade_reference"],
                    {
                        "operation": "select_clade_reference",
                        "series": "observed_full",
                        "open_interval": ["2011-08-01", "2015-06-01"],
                    },
                )
                selection = transformations["select_clades_by_total_frequency"]
                self.assertEqual(selection["comparison"], "strictly_greater_than")
                self.assertEqual(selection["value"], 1.0)
                self.assertEqual(
                    transformations["normalize_frequency"]["group_by"],
                    ["subtype", "segment", "series", "date"],
                )
                self.assertNotIn("legacy_noop_clade_filter", operations)

    def test_background_figures_apply_inclusive_cutoffs_before_selection(self) -> None:
        expected_mutation_key = ["subtype", "segment", "position", "state"]
        figures = {
            figure["figure_id"]: figure for figure in self.manifest["figures"]
        }
        expected_cutoffs = {
            "paper.background-clade": (
                "minimum_duration_days",
                90,
                "before_representative_selection",
            ),
            "paper.background-components": (
                "minimum_duration_days_by_output",
                {"fig-dw-rate": 90, "fig-dw-bg": 90, "fig-ds": None},
                "before_representative_selection_when_present",
            ),
            "diagnostic.delta-h-distributions": (
                "minimum_duration_days",
                90,
                "before_representative_selection",
            ),
        }

        for figure_id, (
            cutoff_operation,
            expected_value,
            expected_filter_order,
        ) in expected_cutoffs.items():
            with self.subTest(figure_id=figure_id):
                figure = figures[figure_id]
                operations = [
                    item["operation"] for item in figure["transformations"]
                ]
                self.assertLess(
                    operations.index(cutoff_operation),
                    operations.index("extract_background_mutations"),
                )
                transformations = {
                    item["operation"]: item for item in figure["transformations"]
                }
                extraction = transformations["extract_background_mutations"]
                self.assertEqual(extraction["key"], expected_mutation_key)
                self.assertEqual(
                    extraction["representative"],
                    "minimum_eligible_start_time",
                )
                self.assertEqual(
                    extraction["duration_filter_order"],
                    expected_filter_order,
                )
                self.assertIs(
                    extraction["source_times_and_outcomes_authoritative"], True
                )

                cutoff = transformations[cutoff_operation]
                self.assertIs(cutoff["inclusive"], True)
                if cutoff_operation == "minimum_duration_days":
                    self.assertEqual(cutoff["value"], expected_value)
                else:
                    self.assertEqual(cutoff["values"], expected_value)

    def test_style_and_canvas_names_align_with_code(self) -> None:
        self.assertEqual(set(self.manifest["style_profiles"]), set(STYLES))
        manifest_canvases = set(self.manifest["canvas_profiles"])
        self.assertTrue(set(CANVASES).issubset(manifest_canvases))
        self.assertEqual(manifest_canvases - set(CANVASES), {"legacy_matplotlib_default"})

    def test_canonical_and_archived_input_zones_are_explicit(self) -> None:
        for input_id, input_set in self.manifest["input_sets"].items():
            paths = [
                path
                for artifact in input_set["artifacts"]
                for path in expand_path_spec(artifact)
            ]
            if input_set["availability"] == "present":
                self.assertTrue(
                    all(path.startswith(CANONICAL_DATA_PREFIX) for path in paths),
                    input_id,
                )
            elif input_set["availability"] == "archived":
                self.assertTrue(
                    all(path.startswith(ARCHIVE_PREFIXES) for path in paths),
                    input_id,
                )

    def test_corrected_emd_snapshots_are_allowed_archive_roots(self) -> None:
        expected = {
            "emd.results-9m-cross-validated-corrected": (
                "archive/emd_cv_2026-09-30/original/",
                4,
            ),
            "emd.results-6m-12m-cross-validated-corrected": (
                "archive/emd_cv_2026-10-10/original/",
                8,
            ),
            "emd.results-6m-9m-12m-cross-validated-corrected": (
                "archive/emd_cv_2026-10-13/original/",
                12,
            ),
        }
        for input_id, (prefix, expected_count) in expected.items():
            with self.subTest(input_id=input_id):
                self.assertIn(prefix, ARCHIVE_PREFIXES)
                input_set = self.manifest["input_sets"][input_id]
                paths = [
                    path
                    for artifact in input_set["artifacts"]
                    for path in expand_path_spec(artifact)
                ]
                self.assertEqual(len(paths), expected_count)
                self.assertTrue(all(path.startswith(prefix) for path in paths))

        source_schemas = self.manifest["input_sets"][
            "emd.results-6m-9m-12m-cross-validated-corrected"
        ]["source_schemas"]
        self.assertEqual(
            source_schemas,
            {
                "6m": "emd.results_cv_6m_corrected.v1",
                "9m": "emd.results_cv_9m_corrected.v1",
                "12m": "emd.results_cv_12m_corrected.v1",
            },
        )

    def test_live_figures_consume_only_canonical_records(self) -> None:
        inputs = self.manifest["input_sets"]
        products = {item["product_id"]: item for item in self.manifest["data_products"]}
        for figure in self.manifest["figures"]:
            if figure["activity_status"] not in {"active", "candidate"}:
                continue
            for reference in figure["inputs"]:
                if "input_set" in reference:
                    availability = inputs[reference["input_set"]]["availability"]
                else:
                    availability = products[reference["data_product"]]["availability"]
                self.assertEqual(availability, "present", figure["figure_id"])

    def test_live_outputs_are_purpose_qualified(self) -> None:
        for figure in self.manifest["figures"]:
            if figure["activity_status"] not in {"active", "candidate"}:
                continue
            for output in figure["outputs"]:
                output_purpose = output.get("purpose", figure["purpose"])
                prefix = PURPOSE_OUTPUT_PREFIX[output_purpose]
                self.assertTrue(
                    all(path.startswith(prefix) for path in expand_path_spec(output)),
                    figure["figure_id"],
                )

    def test_single_site_auc_uses_paper_date_filtered_slm_family(self) -> None:
        figure = next(
            item
            for item in self.manifest["figures"]
            if item["figure_id"] == "paper.single-site-auc"
        )
        reference = figure["inputs"][0]
        self.assertEqual(reference["data_product"], "derived.fixation-auc")
        self.assertEqual(
            reference["selector"],
            {
                "model_id": "slm",
                "source_variant": "paper_date_filtered",
                "minimum_observation_day": 1095,
            },
        )
        self.assertEqual(
            reference["lineage"]["prediction_family"],
            "fixation.single-site-predictions",
        )
        self.assertIn("df_SLM", reference["lineage"]["excluded_variant"])

        product = next(
            item
            for item in self.manifest["data_products"]
            if item["product_id"] == "derived.fixation-auc"
        )
        self.assertIn("slm/paper_date_filtered", product["source_variants"])
        self.assertIn("slm/extended_all_time", product["source_variants"])

    def test_template_expansion_is_deterministic(self) -> None:
        paths = expand_path_spec(
            {
                "path_template": "figures/{purpose}/figure-{panel}.pdf",
                "dimensions": {"purpose": ["paper", "slides"], "panel": ["a", "b"]},
            }
        )
        self.assertEqual(
            paths,
            [
                "figures/paper/figure-a.pdf",
                "figures/paper/figure-b.pdf",
                "figures/slides/figure-a.pdf",
                "figures/slides/figure-b.pdf",
            ],
        )

    def test_undeclared_output_collision_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        figure = next(
            item for item in manifest["figures"] if item["figure_id"] == "paper.selection-sasa"
        )
        figure["outputs"][0]["path"] = "figures/supplement/fig-frequency-selection.pdf"
        figure["outputs"][0]["migration_target"] = figure["outputs"][0]["path"]
        errors = validate_manifest(manifest)
        self.assertTrue(
            any(
                "undeclared collision for figures/supplement/fig-frequency-selection.pdf"
                in error
                for error in errors
            ),
            errors,
        )

    def test_output_collisions_are_case_insensitive(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        figure = next(
            item for item in manifest["figures"] if item["figure_id"] == "paper.selection-sasa"
        )
        figure["outputs"][0]["path"] = (
            "FIGURES/SUPPLEMENT/FIG-FREQUENCY-SELECTION.PDF"
        )
        figure["outputs"][0]["migration_target"] = figure["outputs"][0]["path"]
        errors = validate_manifest(manifest)
        self.assertTrue(any("undeclared collision" in error for error in errors), errors)

    def test_paths_cannot_escape_the_repository(self) -> None:
        with self.assertRaisesRegex(ValueError, "stay inside the repository"):
            expand_path_spec({"path": "../outside.pdf"})

    def test_blocker_requires_a_reason(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        figure = next(
            item for item in manifest["figures"] if item["figure_id"] == "blocker.grant.auc"
        )
        figure["blockers"] = []
        errors = validate_manifest(manifest)
        self.assertTrue(any("blocker figures require" in error for error in errors), errors)

    def test_output_owner_must_match_figure_id(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        figure = next(
            item for item in manifest["figures"] if item["figure_id"] == "paper.frequency-selection"
        )
        figure["outputs"][0]["owner"] = "paper.selection-sasa"
        errors = validate_manifest(manifest)
        self.assertTrue(any("owner must equal paper.frequency-selection" in error for error in errors), errors)

    def test_live_figure_cannot_consume_archived_input(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        figure = next(
            item
            for item in manifest["figures"]
            if item["figure_id"] == "paper.frequency-selection"
        )
        figure["inputs"] = [
            {"input_set": "selection.nested-legacy", "use": "rendered"}
        ]
        errors = validate_manifest(manifest)
        self.assertTrue(any("cannot consume noncanonical input set" in error for error in errors), errors)

    def test_present_product_path_is_checked(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        product = next(
            item
            for item in manifest["data_products"]
            if item["product_id"] == "derived.fixation-auc"
        )
        product["outputs"][0]["path"] = (
            "data/plotting/derived/fixation_metrics/does_not_exist.csv"
        )
        errors = validate_manifest(manifest, repo_root=REPO_ROOT, check_paths=True)
        self.assertTrue(any("declared-present product does not exist" in error for error in errors), errors)

    def test_rare_filtered_method_policy_cannot_be_silently_removed(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["method_semantics"]["slm"]["selection_filter"] = "unknown"
        errors = validate_manifest(manifest)
        self.assertTrue(
            any("method_semantics.slm" in error and "rare_filtered" in error for error in errors),
            errors,
        )

    def test_single_site_source_variant_cannot_be_silently_swapped(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        figure = next(
            item
            for item in manifest["figures"]
            if item["figure_id"] == "paper.single-site-auc"
        )
        figure["inputs"][0]["selector"]["source_variant"] = "extended_all_time"
        errors = validate_manifest(manifest)
        self.assertTrue(
            any("paper.single-site-auc.inputs.selector" in error for error in errors),
            errors,
        )


if __name__ == "__main__":
    unittest.main()
