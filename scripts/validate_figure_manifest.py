#!/usr/bin/env python3
"""Validate the plotting figure manifest without third-party dependencies.

``figure_manifest.yaml`` deliberately uses the JSON subset of YAML 1.2.  That
keeps the manifest readable by YAML tools while allowing this validator to run
with only the Python standard library.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
REQUIRED_ACTIVITY_STATES = {"active", "candidate", "legacy", "blocker"}
REQUIRED_RARE_FILTERED_METHODS = {"genmodel", "slm", "mlr"}
REQUIRED_LEGACY_DIAGNOSTICS = {"fmss", "slfs"}
REQUIRED_AVAILABILITY_STATES = {"present", "archived", "missing"}
REQUIRED_STYLE_PROFILES = {
    "paper_v1",
    "slides_v1",
    "poster_v1",
    "grant_v1",
    "legacy_current_v1",
}
CANONICAL_DATA_PREFIX = "data/plotting/"
ARCHIVE_PREFIX = "archive/pre_plot_reorg_2026-08-26/original/"
ARCHIVE_PREFIXES = (
    ARCHIVE_PREFIX,
    "archive/emd_cv_2026-09-24/original/",
    "archive/emd_cv_2026-09-30/original/",
    "archive/emd_cv_2026-10-10/original/",
    "archive/emd_cv_2026-10-13/original/",
)
PURPOSE_OUTPUT_PREFIX = {
    "paper_main": "figures/paper/",
    "paper_supplement": "figures/supplement/",
    "slides": "figures/slides/",
    "grant": "figures/grant/",
    "poster": "figures/poster/",
    "diagnostic": "figures/diagnostic/",
    "preprocessing": "figures/diagnostic/",
}
PURPOSE_STYLE = {
    "paper_main": "paper_v1",
    "paper_supplement": "paper_v1",
    "slides": "slides_v1",
    "grant": "grant_v1",
    "poster": "poster_v1",
}


def load_manifest(path: Path | str) -> dict[str, Any]:
    """Load the JSON-compatible YAML manifest."""

    manifest_path = Path(path)
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{manifest_path}: expected JSON-compatible YAML; {exc.msg} "
            f"at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"{manifest_path}: top-level value must be an object")
    return value


def expand_path_spec(spec: Mapping[str, Any]) -> list[str]:
    """Expand one path, a path list, or a templated path family."""

    selectors = [key for key in ("path", "paths", "path_template") if key in spec]
    if len(selectors) != 1:
        raise ValueError("path spec must define exactly one of path, paths, path_template")

    selector = selectors[0]
    if selector == "path":
        path = spec["path"]
        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non-empty string")
        expanded = [path]
        _validate_relative_paths(expanded)
        return expanded

    if selector == "paths":
        paths = spec["paths"]
        if not isinstance(paths, list) or not paths:
            raise ValueError("paths must be a non-empty list")
        if not all(isinstance(path, str) and path for path in paths):
            raise ValueError("every paths item must be a non-empty string")
        expanded = list(paths)
        _validate_relative_paths(expanded)
        return expanded

    template = spec["path_template"]
    dimensions = spec.get("dimensions")
    if not isinstance(template, str) or not template:
        raise ValueError("path_template must be a non-empty string")
    if not isinstance(dimensions, dict) or not dimensions:
        raise ValueError("templated paths require a non-empty dimensions object")

    names = list(dimensions)
    values: list[list[Any]] = []
    for name in names:
        dimension_values = dimensions[name]
        if not isinstance(dimension_values, list) or not dimension_values:
            raise ValueError(f"dimension {name!r} must be a non-empty list")
        values.append(dimension_values)

    expanded: list[str] = []
    for combination in itertools.product(*values):
        context = dict(zip(names, combination))
        try:
            expanded.append(template.format(**context))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"cannot expand {template!r}: {exc}") from exc
    _validate_relative_paths(expanded)
    return expanded


def _validate_relative_paths(paths: Iterable[str]) -> None:
    for path in paths:
        pure_path = PurePosixPath(path)
        if "\\" in path:
            raise ValueError(f"paths must use forward slashes: {path!r}")
        if pure_path.is_absolute() or ".." in pure_path.parts:
            raise ValueError(f"path must stay inside the repository: {path!r}")


def _is_id(value: Any) -> bool:
    return isinstance(value, str) and bool(ID_RE.fullmatch(value))


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
    check_paths: bool = False,
) -> list[str]:
    """Return human-readable validation errors; an empty list means valid."""

    errors: list[str] = []

    def error(location: str, message: str) -> None:
        errors.append(f"{location}: {message}")

    if manifest.get("manifest_version") != 1:
        error("manifest_version", "must equal 1")

    scope = manifest.get("scope")
    if not isinstance(scope, dict):
        error("scope", "must be an object")
        scope = {}
    driver_notebook = scope.get("driver_notebook")
    provenance_notebook = scope.get("notebook_cell_provenance")
    if driver_notebook != "figures.ipynb":
        error("scope.driver_notebook", "must identify the thin figures.ipynb driver")
    expected_provenance = f"{ARCHIVE_PREFIX}figures.ipynb"
    if provenance_notebook != expected_provenance:
        error(
            "scope.notebook_cell_provenance",
            f"must identify {expected_provenance}",
        )
    if scope.get("cell_numbering") != "zero_based_in_archived_original":
        error(
            "scope.cell_numbering",
            "must be zero_based_in_archived_original",
        )
    for location, path in (
        ("scope.driver_notebook", driver_notebook),
        ("scope.notebook_cell_provenance", provenance_notebook),
    ):
        if isinstance(path, str):
            try:
                _validate_relative_paths((path,))
            except ValueError as exc:
                error(location, str(exc))
            if check_paths and repo_root is not None and not (repo_root / path).is_file():
                error(location, f"declared notebook does not exist: {path}")

    vocabularies = manifest.get("vocabularies")
    if not isinstance(vocabularies, dict):
        error("vocabularies", "must be an object")
        vocabularies = {}

    activity_values = set(_as_list(vocabularies.get("activity_status")))
    if activity_values != REQUIRED_ACTIVITY_STATES:
        error(
            "vocabularies.activity_status",
            f"must contain exactly {sorted(REQUIRED_ACTIVITY_STATES)}",
        )
    run_values = set(_as_list(vocabularies.get("run_state")))
    migration_values = set(_as_list(vocabularies.get("migration_state")))
    purpose_values = set(_as_list(vocabularies.get("purpose")))
    availability_values = set(_as_list(vocabularies.get("availability_status")))
    if not run_values:
        error("vocabularies.run_state", "must be a non-empty list")
    if not migration_values:
        error("vocabularies.migration_state", "must be a non-empty list")
    if not purpose_values:
        error("vocabularies.purpose", "must be a non-empty list")
    if availability_values != REQUIRED_AVAILABILITY_STATES:
        error(
            "vocabularies.availability_status",
            f"must contain exactly {sorted(REQUIRED_AVAILABILITY_STATES)}",
        )

    style_profiles = manifest.get("style_profiles")
    if not isinstance(style_profiles, dict) or not style_profiles:
        error("style_profiles", "must be a non-empty object")
        style_profiles = {}
    else:
        missing_styles = REQUIRED_STYLE_PROFILES - set(style_profiles)
        if missing_styles:
            error(
                "style_profiles",
                f"missing code-defined profiles {sorted(missing_styles)}",
            )
    canvas_profiles = manifest.get("canvas_profiles")
    if not isinstance(canvas_profiles, dict) or not canvas_profiles:
        error("canvas_profiles", "must be a non-empty object")
        canvas_profiles = {}
    else:
        for profile_id, profile in canvas_profiles.items():
            location = f"canvas_profiles.{profile_id}"
            if not _is_id(profile_id):
                error(location, "profile name must be a semantic lowercase ID")
            if not isinstance(profile, dict):
                error(location, "must be an object")
                continue
            for numeric_key in ("width_in", "height_in", "dpi"):
                value = profile.get(numeric_key)
                if not isinstance(value, (int, float)) or value <= 0:
                    error(f"{location}.{numeric_key}", "must be a positive number")
            if profile.get("format") not in {"pdf", "png", "notebook_display"}:
                error(f"{location}.format", "must be pdf, png, or notebook_display")

    input_sets = manifest.get("input_sets")
    if not isinstance(input_sets, dict):
        error("input_sets", "must be an object")
        input_sets = {}
    for input_id, input_set in input_sets.items():
        location = f"input_sets.{input_id}"
        if not _is_id(input_id):
            error(location, "input set name must be a semantic lowercase ID")
        if not isinstance(input_set, dict):
            error(location, "must be an object")
            continue
        for required_key in ("owner", "schema", "availability", "artifacts"):
            if required_key not in input_set:
                error(location, f"missing {required_key}")
        availability = input_set.get("availability")
        if availability not in availability_values:
            error(f"{location}.availability", f"unknown availability {availability!r}")
        expected_zone = {
            "present": "canonical",
            "archived": "archive",
            "missing": "legacy_missing",
        }.get(availability)
        if expected_zone is not None and input_set.get("storage_zone") != expected_zone:
            error(
                f"{location}.storage_zone",
                f"availability={availability} requires storage_zone={expected_zone}",
            )
        artifacts = _as_list(input_set.get("artifacts"))
        if not artifacts:
            error(f"{location}.artifacts", "must be a non-empty list")
        for index, artifact in enumerate(artifacts):
            artifact_location = f"{location}.artifacts[{index}]"
            if not isinstance(artifact, dict):
                error(artifact_location, "must be an object")
                continue
            try:
                paths = expand_path_spec(artifact)
            except ValueError as exc:
                error(artifact_location, str(exc))
                continue
            if availability == "present":
                for path in paths:
                    if not path.startswith(CANONICAL_DATA_PREFIX):
                        error(
                            artifact_location,
                            f"declared-present input must use canonical data path: {path}",
                        )
            elif availability == "archived":
                for path in paths:
                    if not path.startswith(ARCHIVE_PREFIXES):
                        error(
                            artifact_location,
                            f"archived input must use an immutable snapshot path: {path}",
                        )
            if check_paths and repo_root is not None and availability == "present":
                for path in paths:
                    if not (repo_root / path).exists():
                        error(artifact_location, f"declared-present path does not exist: {path}")

    method_semantics = manifest.get("method_semantics")
    if not isinstance(method_semantics, dict):
        error("method_semantics", "must be an object")
        method_semantics = {}
    for method in REQUIRED_RARE_FILTERED_METHODS:
        entry = method_semantics.get(method)
        if not isinstance(entry, dict) or entry.get("selection_filter") != "rare_filtered":
            error(
                f"method_semantics.{method}",
                "Results method must declare selection_filter=rare_filtered",
            )
    for method in REQUIRED_LEGACY_DIAGNOSTICS:
        entry = method_semantics.get(method)
        if not isinstance(entry, dict) or entry.get("lifecycle") != "legacy_diagnostic":
            error(
                f"method_semantics.{method}",
                "must declare lifecycle=legacy_diagnostic",
            )

    data_products = manifest.get("data_products")
    if not isinstance(data_products, list):
        error("data_products", "must be a list")
        data_products = []
    data_product_ids: set[str] = set()
    data_product_availability: dict[str, str] = {}
    artifact_owners: dict[str, set[str]] = defaultdict(set)
    artifact_display_paths: dict[str, str] = {}
    for index, product in enumerate(data_products):
        location = f"data_products[{index}]"
        if not isinstance(product, dict):
            error(location, "must be an object")
            continue
        product_id = product.get("product_id")
        if not _is_id(product_id):
            error(f"{location}.product_id", "must be a semantic lowercase ID")
            continue
        if product_id in data_product_ids:
            error(f"{location}.product_id", f"duplicate product ID {product_id}")
        data_product_ids.add(product_id)
        availability = product.get("availability")
        if availability not in availability_values:
            error(f"{location}.availability", f"unknown availability {availability!r}")
        else:
            data_product_availability[product_id] = availability
        product_inputs = product.get("inputs")
        if not isinstance(product_inputs, list):
            error(f"{location}.inputs", "must be a list of input-set IDs")
        else:
            for input_id in product_inputs:
                if input_id not in input_sets:
                    error(f"{location}.inputs", f"unknown input set {input_id!r}")
        outputs = _as_list(product.get("outputs"))
        if not outputs:
            error(f"{location}.outputs", "must be a non-empty list")
        for output_index, output in enumerate(outputs):
            output_location = f"{location}.outputs[{output_index}]"
            if not isinstance(output, dict):
                error(output_location, "must be an object")
                continue
            if output.get("owner") != product_id:
                error(output_location, f"owner must equal {product_id}")
            try:
                for path in expand_path_spec(output):
                    folded_path = path.casefold()
                    artifact_owners[folded_path].add(product_id)
                    artifact_display_paths.setdefault(folded_path, path)
                    if availability == "present" and not path.startswith(
                        f"{CANONICAL_DATA_PREFIX}derived/"
                    ):
                        error(
                            output_location,
                            f"present derived product must use data/plotting/derived/: {path}",
                        )
                    if availability == "archived" and not path.startswith(ARCHIVE_PREFIX):
                        error(
                            output_location,
                            f"archived product must use the immutable snapshot path: {path}",
                        )
                    if (
                        check_paths
                        and repo_root is not None
                        and availability == "present"
                        and not (repo_root / path).exists()
                    ):
                        error(output_location, f"declared-present product does not exist: {path}")
            except ValueError as exc:
                error(output_location, str(exc))

    figures = manifest.get("figures")
    if not isinstance(figures, list) or not figures:
        error("figures", "must be a non-empty list")
        figures = []
    figure_ids: set[str] = set()
    observed_activity_states: set[str] = set()
    for index, figure in enumerate(figures):
        location = f"figures[{index}]"
        if not isinstance(figure, dict):
            error(location, "must be an object")
            continue
        figure_id = figure.get("figure_id")
        if not _is_id(figure_id):
            error(f"{location}.figure_id", "must be a semantic lowercase ID")
            continue
        if figure_id in figure_ids:
            error(f"{location}.figure_id", f"duplicate figure ID {figure_id}")
        figure_ids.add(figure_id)

        for required_key in (
            "title",
            "purpose",
            "activity_status",
            "run_state",
            "migration_state",
            "notebook",
            "implementation",
            "inputs",
            "transformations",
            "style_profile",
            "canvas",
            "outputs",
            "blockers",
        ):
            if required_key not in figure:
                error(location, f"missing {required_key}")

        purpose = figure.get("purpose")
        if purpose not in purpose_values:
            error(f"{location}.purpose", f"unknown purpose {purpose!r}")
        activity = figure.get("activity_status")
        if activity not in activity_values:
            error(f"{location}.activity_status", f"unknown state {activity!r}")
        else:
            observed_activity_states.add(activity)
        if figure.get("run_state") not in run_values:
            error(f"{location}.run_state", f"unknown state {figure.get('run_state')!r}")
        if figure.get("migration_state") not in migration_values:
            error(
                f"{location}.migration_state",
                f"unknown state {figure.get('migration_state')!r}",
            )
        blockers = figure.get("blockers")
        if not isinstance(blockers, list):
            error(f"{location}.blockers", "must be a list")
        elif activity == "blocker" and not blockers:
            error(f"{location}.blockers", "blocker figures require at least one blocker")

        notebook = figure.get("notebook")
        if not isinstance(notebook, dict):
            error(f"{location}.notebook", "must be an object")
        else:
            expected_notebook = manifest.get("scope", {}).get(
                "notebook_cell_provenance"
            )
            if notebook.get("path") != expected_notebook:
                error(
                    f"{location}.notebook.path",
                    "must identify scope.notebook_cell_provenance",
                )
            cells = notebook.get("cells")
            if not isinstance(cells, list) or not cells or not all(
                isinstance(cell, int) and cell >= 0 for cell in cells
            ):
                error(f"{location}.notebook.cells", "must be non-empty non-negative integers")

        implementation = figure.get("implementation")
        if not isinstance(implementation, dict) or not implementation.get("entrypoint"):
            error(f"{location}.implementation", "must name an entrypoint")

        style_profile = figure.get("style_profile")
        if style_profile not in style_profiles:
            error(f"{location}.style_profile", f"unknown profile {style_profile!r}")
        expected_style = PURPOSE_STYLE.get(purpose)
        if activity in {"active", "candidate"} and expected_style is not None:
            if style_profile != expected_style:
                error(
                    f"{location}.style_profile",
                    f"{activity} {purpose} figure must use {expected_style}",
                )
        canvas = figure.get("canvas")
        if not isinstance(canvas, dict):
            error(f"{location}.canvas", "must be an object")
        else:
            profiles = []
            if "profile" in canvas:
                profiles.append(canvas["profile"])
            profiles.extend(_as_list(canvas.get("profiles")))
            if not profiles:
                error(f"{location}.canvas", "must reference profile or profiles")
            for profile in profiles:
                if profile not in canvas_profiles:
                    error(f"{location}.canvas", f"unknown canvas profile {profile!r}")

        inputs = figure.get("inputs")
        if not isinstance(inputs, list):
            error(f"{location}.inputs", "must be a list")
        else:
            for input_index, input_ref in enumerate(inputs):
                input_location = f"{location}.inputs[{input_index}]"
                if not isinstance(input_ref, dict):
                    error(input_location, "must be an object")
                    continue
                selectors = [key for key in ("input_set", "data_product") if key in input_ref]
                if len(selectors) != 1:
                    error(input_location, "must reference exactly one input_set or data_product")
                elif selectors[0] == "input_set" and input_ref["input_set"] not in input_sets:
                    error(input_location, f"unknown input set {input_ref['input_set']!r}")
                elif (
                    selectors[0] == "input_set"
                    and activity in {"active", "candidate"}
                    and input_sets[input_ref["input_set"]].get("availability") != "present"
                ):
                    error(
                        input_location,
                        f"{activity} figure cannot consume noncanonical input set "
                        f"{input_ref['input_set']!r}",
                    )
                elif selectors[0] == "data_product" and input_ref["data_product"] not in data_product_ids:
                    error(input_location, f"unknown data product {input_ref['data_product']!r}")
                elif (
                    selectors[0] == "data_product"
                    and activity in {"active", "candidate"}
                    and data_product_availability.get(input_ref["data_product"]) != "present"
                ):
                    error(
                        input_location,
                        f"{activity} figure cannot consume noncanonical data product "
                        f"{input_ref['data_product']!r}",
                    )
                if input_ref.get("use") not in {"rendered", "analysis_only", "preprocessing"}:
                    error(input_location, "use must be rendered, analysis_only, or preprocessing")

        if not isinstance(figure.get("transformations"), list):
            error(f"{location}.transformations", "must be a list")

        outputs = figure.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            error(f"{location}.outputs", "must be a non-empty list")
        else:
            for output_index, output in enumerate(outputs):
                output_location = f"{location}.outputs[{output_index}]"
                if not isinstance(output, dict):
                    error(output_location, "must be an object")
                    continue
                if output.get("owner") != figure_id:
                    error(output_location, f"owner must equal {figure_id}")
                if output.get("artifact_kind") not in {"rendered_figure", "notebook_display"}:
                    error(output_location, "invalid artifact_kind")
                try:
                    expanded_output_paths = expand_path_spec(output)
                    for path in expanded_output_paths:
                        folded_path = path.casefold()
                        artifact_owners[folded_path].add(figure_id)
                        artifact_display_paths.setdefault(folded_path, path)
                    if activity in {"active", "candidate"}:
                        output_purpose = output.get("purpose", purpose)
                        if output_purpose not in purpose_values:
                            error(output_location, f"invalid purpose {output_purpose!r}")
                        expected_prefix = PURPOSE_OUTPUT_PREFIX.get(output_purpose)
                        if expected_prefix is None:
                            error(
                                output_location,
                                f"no canonical output folder for {output_purpose!r}",
                            )
                        else:
                            for path in expanded_output_paths:
                                if not path.startswith(expected_prefix):
                                    error(
                                        output_location,
                                        f"canonical output must use {expected_prefix}: {path}",
                                    )
                        if "path" in output:
                            if output.get("migration_target") != output.get("path"):
                                error(
                                    output_location,
                                    "canonical path output must repeat its path as migration_target",
                                )
                        elif "paths" in output:
                            if output.get("migration_targets") != output.get("paths"):
                                error(
                                    output_location,
                                    "canonical paths output must repeat its paths as migration_targets",
                                )
                        elif output.get("migration_target_template") != output.get("path_template"):
                            error(
                                output_location,
                                "canonical template output must repeat path_template as "
                                "migration_target_template",
                            )
                except ValueError as exc:
                    error(output_location, str(exc))

    missing_activity_states = REQUIRED_ACTIVITY_STATES - observed_activity_states
    if missing_activity_states:
        error("figures", f"missing activity states {sorted(missing_activity_states)}")

    collisions = manifest.get("collisions")
    if not isinstance(collisions, list):
        error("collisions", "must be a list")
        collisions = []
    collision_ids: set[str] = set()
    declared_collision_paths: dict[str, set[str]] = defaultdict(set)
    known_owner_ids = figure_ids | data_product_ids
    for index, collision in enumerate(collisions):
        location = f"collisions[{index}]"
        if not isinstance(collision, dict):
            error(location, "must be an object")
            continue
        collision_id = collision.get("collision_id")
        if not _is_id(collision_id):
            error(f"{location}.collision_id", "must be a semantic lowercase ID")
        elif collision_id in collision_ids:
            error(f"{location}.collision_id", f"duplicate collision ID {collision_id}")
        else:
            collision_ids.add(collision_id)
        kind = collision.get("kind")
        if kind not in {"path_collision", "producer_duplication", "naming_alias"}:
            error(f"{location}.kind", "unknown collision kind")
            continue
        if not collision.get("resolution_status"):
            error(f"{location}.resolution_status", "must be recorded")
        owner_ids = set(_as_list(collision.get("owner_ids")))
        unknown_owners = owner_ids - known_owner_ids
        if unknown_owners:
            error(f"{location}.owner_ids", f"unknown owners {sorted(unknown_owners)}")
        artifacts = _as_list(collision.get("artifacts"))
        if kind in {"path_collision", "naming_alias"} and not artifacts:
            error(f"{location}.artifacts", "must identify affected artifacts")
        for artifact_index, artifact in enumerate(artifacts):
            artifact_location = f"{location}.artifacts[{artifact_index}]"
            if not isinstance(artifact, dict):
                error(artifact_location, "must be an object")
                continue
            try:
                for path in expand_path_spec(artifact):
                    declared_collision_paths[path.casefold()].update(owner_ids)
            except ValueError as exc:
                error(artifact_location, str(exc))
        if kind == "path_collision" and len(owner_ids) < 2:
            error(f"{location}.owner_ids", "path collisions require at least two owners")
        if kind == "producer_duplication":
            producer_cells = collision.get("producer_cells")
            if not isinstance(producer_cells, list) or len(producer_cells) < 2:
                error(f"{location}.producer_cells", "must contain at least two cells")

    for folded_path, owners in sorted(artifact_owners.items()):
        if len(owners) <= 1:
            continue
        declared_owners = declared_collision_paths.get(folded_path, set())
        if not owners.issubset(declared_owners):
            display_path = artifact_display_paths.get(folded_path, folded_path)
            error(
                "artifact ownership",
                f"undeclared collision for {display_path}: owners={sorted(owners)}",
            )

    # This distinction is scientifically material: the paper single-site AUC
    # used date-filtered SLM predictions and df_SS_AUC, while df_SLM_* is a
    # separate extended-all-time family retained for other diagnostics.
    single_site = next(
        (
            figure
            for figure in figures
            if isinstance(figure, dict)
            and figure.get("figure_id") == "paper.single-site-auc"
        ),
        None,
    )
    if single_site is None:
        error("figures", "missing paper.single-site-auc provenance record")
    else:
        refs = _as_list(single_site.get("inputs"))
        matching_refs = [
            ref
            for ref in refs
            if isinstance(ref, dict) and ref.get("data_product") == "derived.fixation-auc"
        ]
        if len(matching_refs) != 1:
            error(
                "paper.single-site-auc.inputs",
                "must consume exactly one derived.fixation-auc product",
            )
        else:
            ref = matching_refs[0]
            selector = ref.get("selector")
            expected_selector = {
                "model_id": "slm",
                "source_variant": "paper_date_filtered",
                "minimum_observation_day": 1095,
            }
            if selector != expected_selector:
                error(
                    "paper.single-site-auc.inputs.selector",
                    f"must equal {expected_selector!r}",
                )
            lineage = ref.get("lineage")
            if not isinstance(lineage, dict) or lineage.get("prediction_family") != (
                "fixation.single-site-predictions"
            ):
                error(
                    "paper.single-site-auc.inputs.lineage",
                    "must identify fixation.single-site-predictions",
                )
            if not isinstance(lineage, dict) or "df_SLM" not in lineage.get(
                "excluded_variant", ""
            ):
                error(
                    "paper.single-site-auc.inputs.lineage",
                    "must explicitly exclude the differing df_SLM_* family",
                )

    product_by_id = {
        product.get("product_id"): product
        for product in data_products
        if isinstance(product, dict)
    }
    fixation_auc = product_by_id.get("derived.fixation-auc")
    source_variants = fixation_auc.get("source_variants") if isinstance(fixation_auc, dict) else None
    if not isinstance(source_variants, dict) or not {
        "slm/paper_date_filtered",
        "slm/extended_all_time",
    }.issubset(source_variants):
        error(
            "data_products.derived.fixation-auc.source_variants",
            "must retain both paper_date_filtered and extended_all_time SLM families",
        )

    legacy_single_site = product_by_id.get("derived.single-site-auc-1095")
    if not isinstance(legacy_single_site, dict) or legacy_single_site.get("inputs") != [
        "fixation.single-site-predictions"
    ]:
        error(
            "data_products.derived.single-site-auc-1095.inputs",
            "must preserve fixation.single-site-predictions as the legacy source",
        )

    return errors


def summarize(manifest: Mapping[str, Any]) -> str:
    figures = _as_list(manifest.get("figures"))
    products = _as_list(manifest.get("data_products"))
    collisions = _as_list(manifest.get("collisions"))
    activities = Counter(
        figure.get("activity_status", "missing")
        for figure in figures
        if isinstance(figure, dict)
    )
    activity_summary = ", ".join(f"{key}={activities[key]}" for key in sorted(activities))
    return (
        f"figures={len(figures)} ({activity_summary}); "
        f"data_products={len(products)}; collisions={len(collisions)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        default="figure_manifest.yaml",
        type=Path,
        help="manifest path (default: figure_manifest.yaml)",
    )
    parser.add_argument(
        "--check-paths",
        action="store_true",
        help="also check artifacts declared availability=present",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    repo_root = args.manifest.resolve().parent
    errors = validate_manifest(manifest, repo_root=repo_root, check_paths=args.check_paths)
    if errors:
        print(f"Manifest validation failed with {len(errors)} error(s):", file=sys.stderr)
        for issue in errors:
            print(f"- {issue}", file=sys.stderr)
        return 1

    print(f"Manifest valid: {summarize(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
