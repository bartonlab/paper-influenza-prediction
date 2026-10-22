#!/usr/bin/env python3
"""Create the initial editable, executable figure-recipe notebook.

The archived exploratory notebook remains available for familiar reruns. Its
ordered cell sources are pinned, while outputs, execution counts, and notebook
metadata may change. The generated notebook is only a starting point for
hands-on figure work and may be edited normally. Replacement therefore refuses
to overwrite a modified working notebook unless ``--force`` is explicitly
supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


# Historical byte hash recorded when the archive was first created. The user
# intentionally reruns cells in this archived notebook, so execution counts,
# outputs, and notebook metadata are expected to change after that snapshot.
ARCHIVED_ORIGINAL_SHA256 = (
    "14dfd3646e24fd1ab7ec188337d65cbe4fceb17288bafc9633b3935507da5c44"
)
ARCHIVED_SOURCE_SHA256 = (
    "09d970289744066bd95478a0ca3ef988e9a584ba1d1ba514edfae9e6422cc6da"
)
ARCHIVED_NOTEBOOK = Path(
    "archive/pre_plot_reorg_2026-08-26/original/figures.ipynb"
)
CURRENT_NOTEBOOK = Path("figures.ipynb")


# This inventory follows the attached manuscript draft, not the historical
# notebook order. It is intentionally plain data so tests can lock both the
# order and honest run-state classification without parsing prose heuristically.
MANUSCRIPT_FIGURES: tuple[dict[str, Any], ...] = (
    {
        "label": "Main Figure 1",
        "cell_id": "main001",
        "title": "Sequence-structured forecasting framework",
        "status": "EDITABLE ARTWORK",
        "output": "figures/paper/fig-model-overview.pdf",
        "entrypoint": (
            "manual export from "
            "`data/plotting/inputs/artwork/fig-model-overview.pptx`"
        ),
        "details": (
            "Figure ID: `paper.model-overview`. This conceptual schematic is "
            "maintained as editable PowerPoint artwork rather than a "
            "data-driven plot. The source cell below checks and links both "
            "the editable source and exported PDF; export the slide without "
            "rescaling its canvas."
        ),
        "statistics": "none; this is a conceptual schematic",
        "recipe": "model_overview_artwork",
    },
    {
        "label": "Main Figure 2",
        "cell_id": "main002",
        "title": "Selection, structure, and convergent landscapes",
        "status": "RUNNABLE",
        "output": "figures/paper/fig-selection-structure-space.pdf",
        "entrypoint": "`figures.plot_selection_structure_space`",
        "details": (
            "Canonical inputs: `data/plotting/inputs/selection/`, "
            "`data/plotting/derived/structure/sasa_selection.csv`, and "
            "`data/plotting/inputs/structure/`. The wrapper reads those "
            "maintained tables and image assets directly. Selection-axis "
            "display limits preserve padding above the outer ticks: HA "
            "0-0.42 (top tick 0.4) and NA 0-0.26 (top tick 0.2)."
        ),
        "statistics": (
            "`result.audit`, including the RBS/antigenic-site tests and "
            "selection-curve ranges"
        ),
        "recipe": "selection_structure_space",
    },
    {
        "label": "Main Figure 3",
        "cell_id": "main003",
        "title": "Fixation probability and AUROC improvement",
        "status": "RUNNABLE",
        "output": "figures/paper/fig-pfix-dauc.pdf",
        "entrypoint": "`figures.plot_pfix_dauc`",
        "details": (
            "Canonical inputs: "
            "`data/plotting/derived/fixation_metrics/fixation_probability.csv` "
            "and `data/plotting/derived/fixation_metrics/auc.csv`; the wrapper "
            "selects the full-model, all-time, 1,095-day records explicitly."
        ),
        "statistics": (
            "`result.audit`, with the exact frequency bins and AUROC summaries"
        ),
        "recipe": "pfix_dauc",
    },
    {
        "label": "Main Figure 4",
        "cell_id": "main004",
        "title": "Full model versus neutral forecasts, 9 months",
        "status": "RUNNABLE",
        "output": "figures/paper/fig-emd-vs-null-9m.pdf",
        "entrypoint": "`figures.plot_emd_comparison`",
        "details": (
            "Scientific recipe: `best GenModel` versus neutral, exact "
            "`evaluation_id` pairing, `results_grid`, and the inclusive "
            "1,095-day warmup."
        ),
        "statistics": (
            "`result.audit`; the run cell displays paired n, mean delta EMD, "
            "t statistic, and p-value for every protein"
        ),
        "emd": {
            "comparator": "null",
            "horizon_months": 9,
            "output": "figures/paper/fig-emd-vs-null-9m.pdf",
        },
    },
    {
        "label": "Main Figure 5",
        "cell_id": "main005",
        "title": "Mutation effects, backgrounds, and clade forecasts",
        "status": "RUNNABLE",
        "output": "figures/paper/fig-dw-clade.pdf",
        "entrypoint": "`figures.plot_background_clade`",
        "details": (
            "Canonical inputs: "
            "`data/plotting/inputs/background/population_fitness.csv` and "
            "`data/plotting/inputs/clades/h3n2_ha_frequencies.csv`. The "
            "renderer first applies the inclusive 90-day duration cutoff, "
            "then selects the earliest eligible source record for each "
            "mutation identity. Panel B uses a -1.6% to +1.6% vertical "
            "range, and the background-panel y-axis lines terminate at their "
            "outer major ticks."
        ),
        "statistics": (
            "`result.audit`, with fixed-versus-lost n, means, test statistics, "
            "and p-values for all three background quantities"
        ),
        "recipe": "background_clade",
    },
    {
        "label": "Supplementary Figure S1",
        "cell_id": "supp001",
        "title": "Toy-model selection inference",
        "status": "MISSING SOURCE",
        "output": (
            "not registered; intended "
            "figures/supplement/fig-toy-selection-inference.pdf"
        ),
        "entrypoint": "none in the active repository",
        "details": (
            "The archived PDF "
            "`selection_coeff_inference_toy_model_shared_cbar.pdf` is a "
            "reference only. An active canonical table containing the 1,000 "
            "replicate true and inferred coefficients is missing."
        ),
        "statistics": (
            "a future recipe should expose replicate count and true/inferred "
            "error summaries"
        ),
    },
    {
        "label": "Supplementary Figure S2",
        "cell_id": "supp002",
        "title": "Toy-model fixation probability and AUROC",
        "status": "OUT OF SCOPE",
        "output": "figures/supplement/fig-toy-fixation-auc.pdf",
        "entrypoint": (
            "a canonical adapter based on "
            "`notebooks/Fig_AUC_and_Fix_Prob_layout_toy_model_Py.ipynb`"
        ),
        "details": (
            "Related inputs and plotting code exist, but the historical panel "
            "order does not match the draft. S2 was not included in the "
            "supplied final-figure set, so it is deliberately outside this "
            "migration pass."
        ),
        "statistics": (
            "fixation-probability and AUROC values by frequency bin for both "
            "learned-model and neutral predictions"
        ),
    },
    {
        "label": "Supplementary Figure S3",
        "cell_id": "supp003",
        "title": "Toy-model EMD versus forecast horizon",
        "status": "OUT OF SCOPE",
        "output": "figures/supplement/fig-toy-emd-vs-horizon.pdf",
        "entrypoint": (
            "a canonical adapter based on "
            "`notebooks/Fig_EMD_toy_model_Py.ipynb`"
        ),
        "details": (
            "The plotting notebook and archived result family have not been "
            "migrated into one active, documented table and wrapper. S3 was "
            "not included in the supplied final-figure set."
        ),
        "statistics": (
            "sample count and mean/uncertainty for learned, neutral, and "
            "lower-bound EMD at every horizon"
        ),
    },
    {
        "label": "Supplementary Figure S4",
        "cell_id": "supp004",
        "title": "Sequence counts through time",
        "status": "OUT OF SCOPE",
        "output": "figures/supplement/fig-sequence-counts.pdf",
        "entrypoint": (
            "a canonical wrapper based on "
            "`notebooks/Fig_numb_seq_per_month_Py.ipynb`"
        ),
        "details": (
            "Canonical input: "
            "`data/plotting/inputs/surveillance/sequences_by_year.csv`. The "
            "active table is ready, but S4 was not included in the supplied "
            "final-figure set and remains outside this migration pass."
        ),
        "statistics": (
            "the plotted date range and per-protein sequence totals should be "
            "returned with the figure"
        ),
    },
    {
        "label": "Supplementary Figure S5",
        "cell_id": "supp005",
        "title": "Allele frequencies and selection distributions",
        "status": "RUNNABLE",
        "output": "figures/supplement/fig-frequency-selection.pdf",
        "entrypoint": "`figures.plot_frequency_selection`",
        "details": (
            "Canonical inputs: `data/plotting/inputs/frequencies/` and "
            "`data/plotting/inputs/selection/*_mutations.csv`. The wrapper "
            "keeps the four protein-specific time shifts visible."
        ),
        "statistics": (
            "`result.audit`, including plotted trajectory and mutation counts"
        ),
        "recipe": "frequency_selection",
    },
    {
        "label": "Supplementary Figure S6",
        "cell_id": "supp006",
        "title": "Selection versus solvent accessibility",
        "status": "RUNNABLE",
        "output": "figures/supplement/fig-selection-sasa.pdf",
        "entrypoint": "`figures.plot_selection_sasa`",
        "details": (
            "Canonical input: "
            "`data/plotting/derived/structure/sasa_selection.csv`, read as one "
            "maintained long table. Selection-axis limits are HA 0-0.42 "
            "(top tick 0.4) and NA 0-0.26 (top tick 0.2); SASA limits are "
            "0-100 (top tick 80)."
        ),
        "statistics": (
            "`result.audit`, including Pearson and Spearman correlations, "
            "p-values, and paired n"
        ),
        "recipe": "selection_sasa",
    },
    {
        "label": "Supplementary Figure S7",
        "cell_id": "supp007",
        "title": "Full-model fixation AUROC",
        "status": "RUNNABLE",
        "output": "figures/supplement/fig-fixation-auc.pdf",
        "entrypoint": "`figures.plot_fixation_auc`",
        "details": (
            "Canonical input: `data/plotting/derived/fixation_metrics/auc.csv`, "
            "filtered to the full model, all-time source, and 1,095-day "
            "minimum observation time."
        ),
        "statistics": (
            "`result.audit`, with exact frequency bins and AUROC summaries"
        ),
        "recipe": "fixation_auc",
    },
    {
        "label": "Supplementary Figure S8",
        "cell_id": "supp008",
        "title": "Full model versus neutral forecasts, 6 months",
        "status": "RUNNABLE",
        "output": "figures/supplement/fig-emd-vs-null-6m.pdf",
        "entrypoint": "`figures.plot_emd_comparison`",
        "details": "One explicit six-month neutral-comparison recipe.",
        "statistics": "`result.audit`, displayed as a paired EMD t-test table",
        "emd": {
            "comparator": "null",
            "horizon_months": 6,
            "output": "figures/supplement/fig-emd-vs-null-6m.pdf",
        },
    },
    {
        "label": "Supplementary Figure S9",
        "cell_id": "supp009",
        "title": "Full model versus neutral forecasts, 12 months",
        "status": "RUNNABLE",
        "output": "figures/supplement/fig-emd-vs-null-12m.pdf",
        "entrypoint": "`figures.plot_emd_comparison`",
        "details": "One explicit twelve-month neutral-comparison recipe.",
        "statistics": "`result.audit`, displayed as a paired EMD t-test table",
        "emd": {
            "comparator": "null",
            "horizon_months": 12,
            "output": "figures/supplement/fig-emd-vs-null-12m.pdf",
        },
    },
    {
        "label": "Supplementary Figure S10",
        "cell_id": "supp010",
        "title": "Full model versus MLR forecasts, 6 months",
        "status": "RUNNABLE",
        "output": "figures/supplement/fig-emd-vs-mlr-6m.pdf",
        "entrypoint": "`figures.plot_emd_comparison`",
        "details": (
            "One explicit six-month comparison. MLR means multinomial "
            "logistic regression."
        ),
        "statistics": "`result.audit`, displayed as a paired EMD t-test table",
        "emd": {
            "comparator": "MLR",
            "horizon_months": 6,
            "output": "figures/supplement/fig-emd-vs-mlr-6m.pdf",
        },
    },
    {
        "label": "Supplementary Figure S11",
        "cell_id": "supp011",
        "title": "Full model versus MLR forecasts, 9 months",
        "status": "RUNNABLE",
        "output": "figures/supplement/fig-emd-vs-mlr-9m.pdf",
        "entrypoint": "`figures.plot_emd_comparison`",
        "details": "One explicit nine-month MLR-comparison recipe.",
        "statistics": "`result.audit`, displayed as a paired EMD t-test table",
        "emd": {
            "comparator": "MLR",
            "horizon_months": 9,
            "output": "figures/supplement/fig-emd-vs-mlr-9m.pdf",
        },
    },
    {
        "label": "Supplementary Figure S12",
        "cell_id": "supp012",
        "title": "Full model versus MLR forecasts, 12 months",
        "status": "RUNNABLE",
        "output": "figures/supplement/fig-emd-vs-mlr-12m.pdf",
        "entrypoint": "`figures.plot_emd_comparison`",
        "details": "One explicit twelve-month MLR-comparison recipe.",
        "statistics": "`result.audit`, displayed as a paired EMD t-test table",
        "emd": {
            "comparator": "MLR",
            "horizon_months": 12,
            "output": "figures/supplement/fig-emd-vs-mlr-12m.pdf",
        },
    },
    {
        "label": "Supplementary Figure S13",
        "cell_id": "supp013",
        "title": "Mutation fitness for fixed and lost mutations",
        "status": "RUNNABLE",
        "output": "figures/supplement/fig-ds.pdf",
        "entrypoint": "`figures.plot_background_selection`",
        "details": (
            "Canonical input: "
            "`data/plotting/inputs/background/population_fitness.csv`. The "
            "renderer selects one earliest source record per mutation identity; "
            "this figure intentionally applies no duration cutoff."
        ),
        "statistics": (
            "`result.audit`, with fixed/lost n, means, Welch t statistic, and "
            "p-value for each protein"
        ),
        "recipe": "background_selection",
    },
    {
        "label": "Supplementary Figure S14",
        "cell_id": "supp014",
        "title": "Initial sequence-background fitness",
        "status": "RUNNABLE",
        "output": "figures/supplement/fig-dw-bg.pdf",
        "entrypoint": "`figures.plot_background_initial_advantage`",
        "details": (
            "Canonical input: "
            "`data/plotting/inputs/background/population_fitness.csv`. The "
            "renderer first applies the inclusive 90-day duration cutoff and "
            "then selects the earliest eligible source record per mutation "
            "identity; source "
            "times and fixed/lost classifications remain authoritative."
        ),
        "statistics": (
            "`result.audit`, with fixed/lost n, means, Welch t statistic, and "
            "p-value for each protein"
        ),
        "recipe": "background_initial_advantage",
    },
    {
        "label": "Supplementary Figure S15",
        "cell_id": "supp015",
        "title": "Subsequent background-fitness growth",
        "status": "RUNNABLE",
        "output": "figures/supplement/fig-dw-rate.pdf",
        "entrypoint": "`figures.plot_background_growth`",
        "details": (
            "Canonical input: "
            "`data/plotting/inputs/background/population_fitness.csv`. The "
            "renderer first applies the inclusive 90-day duration cutoff and "
            "then selects the earliest eligible source record per mutation "
            "identity."
        ),
        "statistics": (
            "`result.audit`, with fixed/lost n, mean growth, Welch t statistic, "
            "and p-value for each protein"
        ),
        "recipe": "background_growth",
    },
    {
        "label": "Supplementary Figure S16",
        "cell_id": "supp016",
        "title": "Single-site fixation AUROC",
        "status": "RUNNABLE",
        "output": "figures/supplement/fig-auc-ss.pdf",
        "entrypoint": "`figures.plot_single_site_auc`",
        "details": (
            "Canonical input: `data/plotting/derived/fixation_metrics/auc.csv`, "
            "filtered to rare-filtered SLM, the paper date-filtered source, and "
            "a 1,095-day minimum observation time."
        ),
        "statistics": (
            "`result.audit`, with exact SLM and neutral AUROC summaries"
        ),
        "recipe": "single_site_auc",
    },
    {
        "label": "Supplementary Figure S17",
        "cell_id": "supp017",
        "title": "Full model versus single-site forecasts, 6 months",
        "status": "RUNNABLE",
        "output": "figures/supplement/fig-emd-vs-slm-6m.pdf",
        "entrypoint": "`figures.plot_emd_comparison`",
        "details": (
            "SLM is the rare-filtered single-site (single-locus) model. This "
            "is one explicit six-month comparison recipe."
        ),
        "statistics": "`result.audit`, displayed as a paired EMD t-test table",
        "emd": {
            "comparator": "SLM",
            "horizon_months": 6,
            "output": "figures/supplement/fig-emd-vs-slm-6m.pdf",
        },
    },
    {
        "label": "Supplementary Figure S18",
        "cell_id": "supp018",
        "title": "Full model versus single-site forecasts, 9 months",
        "status": "RUNNABLE",
        "output": "figures/supplement/fig-emd-vs-slm-9m.pdf",
        "entrypoint": "`figures.plot_emd_comparison`",
        "details": (
            "SLM is the rare-filtered single-site (single-locus) model. This "
            "is one explicit nine-month comparison recipe."
        ),
        "statistics": "`result.audit`, displayed as a paired EMD t-test table",
        "emd": {
            "comparator": "SLM",
            "horizon_months": 9,
            "output": "figures/supplement/fig-emd-vs-slm-9m.pdf",
        },
    },
    {
        "label": "Supplementary Figure S19",
        "cell_id": "supp019",
        "title": "Full model versus single-site forecasts, 12 months",
        "status": "RUNNABLE",
        "output": "figures/supplement/fig-emd-vs-slm-12m.pdf",
        "entrypoint": "`figures.plot_emd_comparison`",
        "details": (
            "SLM is the rare-filtered single-site (single-locus) model. This "
            "is one explicit twelve-month comparison recipe."
        ),
        "statistics": "`result.audit`, displayed as a paired EMD t-test table",
        "emd": {
            "comparator": "SLM",
            "horizon_months": 12,
            "output": "figures/supplement/fig-emd-vs-slm-12m.pdf",
        },
    },
)

MANUSCRIPT_FIGURE_ORDER = tuple(figure["label"] for figure in MANUSCRIPT_FIGURES)
MANUSCRIPT_FIGURE_STATUS = {
    figure["label"]: figure["status"] for figure in MANUSCRIPT_FIGURES
}


class NotebookReplacementRefused(RuntimeError):
    """Raised when replacing the notebook could overwrite user work."""


def _markdown(cell_id: str, source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.strip() + "\n",
    }


def _code(cell_id: str, source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def _figure_inventory_markdown(figure: dict[str, Any]) -> str:
    """Render one manuscript entry with a consistent visible contract."""

    return f"""
## {figure['label']} - {figure['title']}

**Status: {figure['status']}**

- Canonical output: `{figure['output']}`
- Entry point: {figure['entrypoint']}
- Data/adapter note: {figure['details']}
- Statistical readout: {figure['statistics']}.
"""


def _emd_recipe_source(figure: dict[str, Any]) -> str:
    """Return one deliberately explicit, one-horizon manuscript EMD cell."""

    recipe = figure["emd"]
    comparator = recipe["comparator"]
    horizon_months = recipe["horizon_months"]
    output_parts = recipe["output"].split("/")
    output_expression = " / ".join(f'"{part}"' for part in output_parts)
    return f'''# {figure["label"]}: {figure["title"]}
INPUT_DIRECTORY = ROOT / "data" / "plotting" / "inputs" / "emd"
INPUTS = [INPUT_DIRECTORY / name for name in ("evaluations.csv", "values.csv", "methods.csv", "sources.csv")]
COMPARATOR = "{comparator}"
HORIZON_MONTHS = {horizon_months}
# FIXED_RECIPE_METADATA: enforced inside f.plot_emd_comparison; shown here
# so the reviewed scientific choices remain visible, not as editable arguments.
EVALUATION_PROTOCOL = "results_grid"
FULL_MODEL_SOURCE = f.FULL_MODEL_SOURCE_BY_HORIZON[HORIZON_MONTHS]
WARMUP_DAYS = 1095
VERSION = "paper"
OUTPUT_FILE = ROOT / {output_expression}

assert all(path.is_file() for path in INPUTS)
if SAVE:
    result = f.plot_emd_comparison(
        comparator=COMPARATOR,
        horizon_months=HORIZON_MONTHS,
        input_directory=INPUT_DIRECTORY,
        version=VERSION,
        output_file=OUTPUT_FILE,
        overwrite=OVERWRITE,
    )
    display(result.figure)
    STATISTICS = display_emd_statistics(result)
    link_to(OUTPUT_FILE)'''


def _artwork_recipe_source(figure: dict[str, Any]) -> str:
    """Return the explicit source/export cell for conceptual Main Figure 1."""

    return f'''# {figure["label"]}: {figure["title"]}
SOURCE_ARTWORK = ROOT / "data" / "plotting" / "inputs" / "artwork" / "fig-model-overview.pptx"
OUTPUT_FILE = ROOT / "figures" / "paper" / "fig-model-overview.pdf"
EXPORT_RULE = "Export the single-slide PowerPoint at its native canvas size; do not rescale."

assert SOURCE_ARTWORK.is_file()
print(EXPORT_RULE)
print("Editable source:", SOURCE_ARTWORK)
display(FileLink(str(SOURCE_ARTWORK)))
if OUTPUT_FILE.is_file():
    print("Current exported PDF:", OUTPUT_FILE)
    link_to(OUTPUT_FILE)
else:
    print("Exported PDF is not present yet:", OUTPUT_FILE)'''


def _manuscript_recipe_source(figure: dict[str, Any]) -> str:
    """Return one transparent code cell for a non-EMD manuscript figure."""

    recipe = figure["recipe"]
    title = f'# {figure["label"]}: {figure["title"]}'
    recipes = {
        "selection_structure_space": f'''{title}
INPUT_DIRECTORY = ROOT / "data" / "plotting"
INPUTS = [
    INPUT_DIRECTORY / "inputs" / "selection",
    INPUT_DIRECTORY / "derived" / "structure" / "sasa_selection.csv",
    INPUT_DIRECTORY / "inputs" / "structure",
]
MINIMUM_MUTATION_COUNT = 20
# FIXED_DISPLAY_METADATA: outer ticks retain the manuscript labels while the
# limits provide room for every curve and the HA-domain annotations.
SELECTION_Y_LIMITS = {{"HA": (0, 0.42), "NA": (0, 0.26)}}
SELECTION_TOP_TICKS = {{"HA": 0.4, "NA": 0.2}}
VERSION = "paper"
OUTPUT_FILE = ROOT / "figures" / "paper" / "fig-selection-structure-space.pdf"

assert all(path.exists() for path in INPUTS)
if SAVE:
    result = f.plot_selection_structure_space(
        input_directory=INPUT_DIRECTORY,
        minimum_mutation_count=MINIMUM_MUTATION_COUNT,
        version=VERSION,
        output_file=OUTPUT_FILE,
        overwrite=OVERWRITE,
    )
    display(result.figure)
    AUDIT = display_figure_audit(result)
    link_to(OUTPUT_FILE)''',
        "pfix_dauc": f'''{title}
FIXATION_PROBABILITY_FILE = ROOT / "data" / "plotting" / "derived" / "fixation_metrics" / "fixation_probability.csv"
AUC_FILE = ROOT / "data" / "plotting" / "derived" / "fixation_metrics" / "auc.csv"
# FIXED_RECIPE_METADATA: model and source are enforced by f.plot_pfix_dauc.
MODEL_ID = "full_model"
AUC_SOURCE_VARIANT = "all_time"
MINIMUM_OBSERVATION_DAY = 1095
VERSION = "paper"
OUTPUT_FILE = ROOT / "figures" / "paper" / "fig-pfix-dauc.pdf"

assert FIXATION_PROBABILITY_FILE.is_file() and AUC_FILE.is_file()
if SAVE:
    result = f.plot_pfix_dauc(
        fixation_probability_file=FIXATION_PROBABILITY_FILE,
        auc_file=AUC_FILE,
        minimum_observation_day=MINIMUM_OBSERVATION_DAY,
        version=VERSION,
        output_file=OUTPUT_FILE,
        overwrite=OVERWRITE,
    )
    display(result.figure)
    AUDIT = display_figure_audit(result)
    link_to(OUTPUT_FILE)''',
        "background_clade": f'''{title}
BACKGROUND_FILE = ROOT / "data" / "plotting" / "inputs" / "background" / "population_fitness.csv"
CLADE_FILE = ROOT / "data" / "plotting" / "inputs" / "clades" / "h3n2_ha_frequencies.csv"
# FIXED_RECIPE_METADATA: outcomes and initial/final times were assigned
# upstream; the plot does not reconstruct them from sparse observations. It
# first applies the inclusive duration cutoff, then selects the earliest
# eligible start_time record for each mutation identity. Clade filters and
# dates remain editable below.
EMERGENCE_THRESHOLD = 0.05
FIXATION_THRESHOLD = 0.95
MUTATION_IDENTITY_FIELDS = ("subtype", "segment", "position", "state")
REPRESENTATIVE_RECORD = "earliest eligible start_time"
MINIMUM_DURATION_DAYS = 90
# FIXED_DISPLAY_METADATA: these document renderer settings in background_plot.py.
BACKGROUND_ADVANTAGE_Y_LIMITS_PERCENT = (-1.6, 1.6)
BACKGROUND_Y_SPINE_BOUNDS = "outer major ticks"
MINIMUM_TOTAL_CLADE_FREQUENCY = 1.0
START_DATE = "2011-08-01"
END_DATE = "2015-06-01"
FORECAST_BASELINE_DATE = "2014-06-01"
VERSION = "paper"
OUTPUT_FILE = ROOT / "figures" / "paper" / "fig-dw-clade.pdf"

assert BACKGROUND_FILE.is_file() and CLADE_FILE.is_file()
if SAVE:
    result = f.plot_background_clade(
        background_file=BACKGROUND_FILE,
        clade_file=CLADE_FILE,
        minimum_total_frequency=MINIMUM_TOTAL_CLADE_FREQUENCY,
        start_date=START_DATE,
        end_date=END_DATE,
        forecast_baseline_date=FORECAST_BASELINE_DATE,
        version=VERSION,
        output_file=OUTPUT_FILE,
        overwrite=OVERWRITE,
    )
    display(result.figure)
    AUDIT = display_figure_audit(result)
    link_to(OUTPUT_FILE)''',
        "frequency_selection": f'''{title}
INPUT_DIRECTORY = ROOT / "data" / "plotting"
FREQUENCY_DIRECTORY = INPUT_DIRECTORY / "inputs" / "frequencies"
SELECTION_DIRECTORY = INPUT_DIRECTORY / "inputs" / "selection"
MINIMUM_MUTATION_COUNT = 20
# FIXED_RECIPE_METADATA: these display alignments are enforced by the reviewed
# frequency renderer; the mutation-count filter below is an editable argument.
H1_TIME_SHIFT_DAYS = 120
H3_TIME_SHIFT_DAYS = 0
VERSION = "paper"
OUTPUT_FILE = ROOT / "figures" / "supplement" / "fig-frequency-selection.pdf"

assert FREQUENCY_DIRECTORY.is_dir() and SELECTION_DIRECTORY.is_dir()
if SAVE:
    result = f.plot_frequency_selection(
        input_directory=INPUT_DIRECTORY,
        minimum_mutation_count=MINIMUM_MUTATION_COUNT,
        version=VERSION,
        output_file=OUTPUT_FILE,
        overwrite=OVERWRITE,
    )
    display(result.figure)
    AUDIT = display_figure_audit(result)
    link_to(OUTPUT_FILE)''',
        "selection_sasa": f'''{title}
INPUT_DIRECTORY = ROOT / "data" / "plotting"
INPUT_FILE = INPUT_DIRECTORY / "derived" / "structure" / "sasa_selection.csv"
# FIXED_DISPLAY_METADATA: limits extend beyond the outer ticks so curves are
# not clipped while the manuscript tick labels remain unchanged.
SELECTION_Y_LIMITS = {{"HA": (0, 0.42), "NA": (0, 0.26)}}
SELECTION_TOP_TICKS = {{"HA": 0.4, "NA": 0.2}}
SASA_Y_LIMITS = (0, 100)
SASA_TOP_TICK = 80
VERSION = "paper"
OUTPUT_FILE = ROOT / "figures" / "supplement" / "fig-selection-sasa.pdf"

assert INPUT_FILE.is_file()
if SAVE:
    result = f.plot_selection_sasa(
        input_directory=INPUT_DIRECTORY,
        version=VERSION,
        output_file=OUTPUT_FILE,
        overwrite=OVERWRITE,
    )
    display(result.figure)
    AUDIT = display_figure_audit(result)
    link_to(OUTPUT_FILE)''',
        "fixation_auc": f'''{title}
AUC_FILE = ROOT / "data" / "plotting" / "derived" / "fixation_metrics" / "auc.csv"
# FIXED_RECIPE_METADATA: model and source are enforced by f.plot_fixation_auc.
MODEL_ID = "full_model"
SOURCE_VARIANT = "all_time"
MINIMUM_OBSERVATION_DAY = 1095
VERSION = "paper"
OUTPUT_FILE = ROOT / "figures" / "supplement" / "fig-fixation-auc.pdf"

assert AUC_FILE.is_file()
if SAVE:
    result = f.plot_fixation_auc(
        auc_file=AUC_FILE,
        minimum_observation_day=MINIMUM_OBSERVATION_DAY,
        version=VERSION,
        output_file=OUTPUT_FILE,
        overwrite=OVERWRITE,
    )
    display(result.figure)
    AUDIT = display_figure_audit(result)
    link_to(OUTPUT_FILE)''',
        "background_selection": f'''{title}
BACKGROUND_FILE = ROOT / "data" / "plotting" / "inputs" / "background" / "population_fitness.csv"
# FIXED_RECIPE_METADATA: outcomes and initial/final times were assigned
# upstream. The renderer selects the earliest start_time record per mutation.
EMERGENCE_THRESHOLD = 0.05
FIXATION_THRESHOLD = 0.95
MUTATION_IDENTITY_FIELDS = ("subtype", "segment", "position", "state")
REPRESENTATIVE_RECORD = "earliest start_time"
MINIMUM_DURATION_DAYS = None  # Figure S13 retains every selected mutation.
VERSION = "paper"
OUTPUT_FILE = ROOT / "figures" / "supplement" / "fig-ds.pdf"

assert BACKGROUND_FILE.is_file()
if SAVE:
    result = f.plot_background_selection(
        background_file=BACKGROUND_FILE,
        version=VERSION,
        output_file=OUTPUT_FILE,
        overwrite=OVERWRITE,
    )
    display(result.figure)
    AUDIT = display_figure_audit(result)
    link_to(OUTPUT_FILE)''',
        "background_initial_advantage": f'''{title}
BACKGROUND_FILE = ROOT / "data" / "plotting" / "inputs" / "background" / "population_fitness.csv"
# FIXED_RECIPE_METADATA: outcomes and initial/final times were assigned
# upstream. The renderer applies the inclusive duration cutoff below first, then
# selects the earliest eligible start_time record per mutation.
EMERGENCE_THRESHOLD = 0.05
FIXATION_THRESHOLD = 0.95
MUTATION_IDENTITY_FIELDS = ("subtype", "segment", "position", "state")
REPRESENTATIVE_RECORD = "earliest eligible start_time"
MINIMUM_DURATION_DAYS = 90
VERSION = "paper"
OUTPUT_FILE = ROOT / "figures" / "supplement" / "fig-dw-bg.pdf"

assert BACKGROUND_FILE.is_file()
if SAVE:
    result = f.plot_background_initial_advantage(
        background_file=BACKGROUND_FILE,
        version=VERSION,
        output_file=OUTPUT_FILE,
        overwrite=OVERWRITE,
    )
    display(result.figure)
    AUDIT = display_figure_audit(result)
    link_to(OUTPUT_FILE)''',
        "background_growth": f'''{title}
BACKGROUND_FILE = ROOT / "data" / "plotting" / "inputs" / "background" / "population_fitness.csv"
# FIXED_RECIPE_METADATA: outcomes and initial/final times were assigned
# upstream. The renderer applies the inclusive duration cutoff below first, then
# selects the earliest eligible start_time record per mutation.
EMERGENCE_THRESHOLD = 0.05
FIXATION_THRESHOLD = 0.95
MUTATION_IDENTITY_FIELDS = ("subtype", "segment", "position", "state")
REPRESENTATIVE_RECORD = "earliest eligible start_time"
MINIMUM_DURATION_DAYS = 90
GROWTH_RATE_UNITS = "percent per year"
VERSION = "paper"
OUTPUT_FILE = ROOT / "figures" / "supplement" / "fig-dw-rate.pdf"

assert BACKGROUND_FILE.is_file()
if SAVE:
    result = f.plot_background_growth(
        background_file=BACKGROUND_FILE,
        version=VERSION,
        output_file=OUTPUT_FILE,
        overwrite=OVERWRITE,
    )
    display(result.figure)
    AUDIT = display_figure_audit(result)
    link_to(OUTPUT_FILE)''',
        "single_site_auc": f'''{title}
AUC_FILE = ROOT / "data" / "plotting" / "derived" / "fixation_metrics" / "auc.csv"
# FIXED_RECIPE_METADATA: model and source are enforced by f.plot_single_site_auc.
MODEL_ID = "slm"
SOURCE_VARIANT = "paper_date_filtered"
MINIMUM_OBSERVATION_DAY = 1095
VERSION = "paper"
OUTPUT_FILE = ROOT / "figures" / "supplement" / "fig-auc-ss.pdf"

assert AUC_FILE.is_file()
if SAVE:
    result = f.plot_single_site_auc(
        auc_file=AUC_FILE,
        minimum_observation_day=MINIMUM_OBSERVATION_DAY,
        version=VERSION,
        output_file=OUTPUT_FILE,
        overwrite=OVERWRITE,
    )
    display(result.figure)
    AUDIT = display_figure_audit(result)
    link_to(OUTPUT_FILE)''',
    }
    try:
        return recipes[recipe]
    except KeyError as error:  # pragma: no cover - template programming error
        raise ValueError(f"Unknown manuscript recipe: {recipe}") from error


def _baseline_recipe_source(horizon_months: int) -> str:
    """Return one explicit best-model-versus-baseline diagnostic cell."""

    return f'''# Diagnostic: best GenModel versus baseline, {horizon_months} months
INPUT_DIRECTORY = ROOT / "data" / "plotting" / "inputs" / "emd"
INPUTS = [INPUT_DIRECTORY / name for name in ("evaluations.csv", "values.csv", "methods.csv", "sources.csv")]
HORIZON_MONTHS = {horizon_months}
# FIXED_RECIPE_METADATA: enforced inside f.plot_emd_best_vs_baseline; shown
# here so the reviewed protocol and warmup remain visible.
EVALUATION_PROTOCOL = "results_grid"
WARMUP_DAYS = 1095
DELTA_EMD = "baseline EMD - best-model EMD"  # positive favors best GenModel
VERSION = "paper"
OUTPUT_FILE = ROOT / "figures" / "diagnostic" / "fig-emd-best-vs-baseline-{horizon_months}m.pdf"

assert all(path.is_file() for path in INPUTS)
if SAVE:
    result = f.plot_emd_best_vs_baseline(
        horizon_months=HORIZON_MONTHS,
        input_directory=INPUT_DIRECTORY,
        version=VERSION,
        output_file=OUTPUT_FILE,
        overwrite=OVERWRITE,
    )
    display(result.figure)
    STATISTICS = display_emd_statistics(result)
    link_to(OUTPUT_FILE)'''


def build_notebook() -> dict[str, Any]:
    """Return the initial manuscript-ordered notebook without writing it."""

    cells = [
        _markdown(
            "intro001",
            """
# Influenza manuscript figure workflow

This is the working, editable recipe book for the current paper draft. It
follows the manuscript itself: Main Figures 1-5, Supplementary Figures S1-S19,
then diagnostics and alternatives. The original exploratory notebook is
preserved at `archive/pre_plot_reorg_2026-08-26/original/figures.ipynb`.

Each figure has one visible status:

- **RUNNABLE** - the cell uses canonical data and a reviewed wrapper in
  `figures.py`.
- **EDITABLE ARTWORK** - the figure is a conceptual drawing with an active
  editable source and an intentionally manual PDF export.
- **OUT OF SCOPE** - the figure is not part of the supplied final-figure set
  being migrated in this pass.
- **MISSING SOURCE** - the draft contains the figure, but the active repository
  does not currently contain enough source material to reproduce it.

Run the setup cell once, then run only the figure cell you need. `SAVE = True`
writes the PDF; `OVERWRITE = True` intentionally replaces an existing PDF.
Every runnable cell exposes its input, scientific choices, output, and direct
function call.

All manuscript EMD cells use canonical release `emd_v6`, produced by
`emd_migration_v6` from the definitive 2026-10-13 panel set. They use
cross-validation-selected `best GenModel`, exact `evaluation_id` pairing, and
the inclusive 1,095-day warmup (2012-01-02 for H3N2; 2012-05-01 for H1N1).
The release has no lower-bound inversions. MLR means **multinomial logistic
regression**. Each EMD cell displays the exact audit statistics behind its
figure so the manuscript text can be updated from the same run. Plot labels
omit non-significant p-values (`p > 0.05`), show exact two-decimal p-values for
`0.01 < p <= 0.05`, retain the compact order-of-magnitude label for
`p <= 0.01`, and mark a significant result with `*` when the full model has
higher EMD than its comparator.
""",
        ),
        _code(
            "setup001",
            """
from dataclasses import asdict
from importlib import reload
from pathlib import Path
import sys

import pandas as pd
from IPython.display import FileLink, display


def find_repository_root(start=Path.cwd()):
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "figures.py").is_file() and (candidate / "data" / "plotting").is_dir():
            return candidate
    raise RuntimeError("Could not locate the influenza figure repository")


ROOT = find_repository_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import figures as f
f = reload(f)

# User switches: a figure cell writes only when SAVE is True.
SAVE = True
OVERWRITE = True

# Visible presentation choices used below.
PAPER_STYLE = f.PAPER_STYLE
SLIDES_STYLE = f.SLIDES_STYLE
PROTEIN_COLORS = f.PROTEIN_COLORS


def link_to(path):
    display(FileLink(str(path)))


def display_emd_statistics(result):
    # Show the paired t-test inputs and results returned by the plot call.

    rows = []
    for panel in result.audit.panels:
        significant = bool(pd.notna(panel.p_value) and panel.p_value <= 0.05)
        rows.append(
            {
                "protein": panel.label,
                "subtype": panel.subtype,
                "segment": panel.segment,
                "paired_n": panel.paired_count,
                "first_observation": panel.first_observation_date,
                "last_observation": panel.last_observation_date,
                "mean_delta_emd": panel.mean_delta_emd,
                "median_delta_emd": panel.median_delta_emd,
                "t_statistic": panel.t_statistic,
                "p_value": panel.p_value,
                "significant_at_0.05": significant,
                "full_model_worse": significant and panel.mean_delta_emd < 0,
            }
        )
    report = pd.DataFrame(rows)
    display(report)
    return report


def display_figure_audit(result):
    # Show the immutable provenance summary and each nested panel table.

    audit = result.audit
    record = asdict(audit)
    nested = {
        key: value
        for key, value in record.items()
        if isinstance(value, (list, tuple))
        and value
        and isinstance(value[0], dict)
    }
    summary = {key: value for key, value in record.items() if key not in nested}
    display(pd.Series(summary, name="value").to_frame())
    for name, rows in nested.items():
        print(name.replace("_", " ").title())
        display(pd.DataFrame(rows))
    return audit


print("Repository:", ROOT)
print("Paper style:", PAPER_STYLE)
print("Slide style:", SLIDES_STYLE)
""",
        ),
        _markdown("main000", "# Main-text figures"),
    ]

    for figure in MANUSCRIPT_FIGURES:
        if figure["label"] == "Supplementary Figure S1":
            cells.append(_markdown("supp000", "# Supplementary figures"))
        cells.append(
            _markdown(
                figure["cell_id"],
                _figure_inventory_markdown(figure),
            )
        )
        if "emd" in figure:
            cells.append(
                _code(
                    f"{figure['cell_id']}run",
                    _emd_recipe_source(figure),
                )
            )
        elif figure.get("recipe") == "model_overview_artwork":
            cells.append(
                _code(
                    f"{figure['cell_id']}run",
                    _artwork_recipe_source(figure),
                )
            )
        elif "recipe" in figure:
            cells.append(
                _code(
                    f"{figure['cell_id']}run",
                    _manuscript_recipe_source(figure),
                )
            )

    cells.extend(
        [
            _markdown(
                "diag000",
                """
# Diagnostics and alternatives

These cells are deliberately outside the manuscript sequence. They are useful
for checking model choices or regenerating reusable components, but they are
not separate figures in the attached draft.
""",
            ),
            _markdown(
                "diagbase",
                """
## Diagnostic - Cross-validated best model versus baseline

**Status: RUNNABLE**

The cells below pair `best GenModel` with `GenModel (baseline)` by exact
`evaluation_id` after the same 1,095-day warmup. Delta EMD is baseline minus
best, so positive values favor the cross-validation-selected model. Outputs
belong under `figures/diagnostic/`; each cell displays its audit table.
""",
            ),
            _code("diagbase6", _baseline_recipe_source(6)),
            _code("diagbase9", _baseline_recipe_source(9)),
            _code("diagbase12", _baseline_recipe_source(12)),
            _markdown(
                "diagclade",
                """
## Alternative component - Standalone H3N2 HA clade forecast

**Status: RUNNABLE**

- Output: `figures/paper/fig-clade.pdf`
- Entry point: `figures.plot_clade_figure`
- Relationship to manuscript: this is the reusable clade component of Main
  Figure 5, not the final combined figure.
""",
            ),
            _code(
                "diagcladerun",
                """
INPUT_FILE = ROOT / "data" / "plotting" / "inputs" / "clades" / "h3n2_ha_frequencies.csv"
OUTPUT_FILE = ROOT / "figures" / "paper" / "fig-clade.pdf"
VERSION = "paper"
MINIMUM_TOTAL_FREQUENCY = 1.0
START_DATE = "2011-08-01"
END_DATE = "2015-06-01"
FORECAST_BASELINE_DATE = "2014-06-01"
FIGSIZE_INCHES = f.CLADE_PAPER_CANVAS_INCHES

assert INPUT_FILE.is_file()
if SAVE:
    clade_audit = f.plot_clade_figure(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        version=VERSION,
        minimum_total_frequency=MINIMUM_TOTAL_FREQUENCY,
        start_date=START_DATE,
        end_date=END_DATE,
        forecast_baseline_date=FORECAST_BASELINE_DATE,
        overwrite=OVERWRITE,
    )
    display(clade_audit)
    link_to(OUTPUT_FILE)
""",
            ),
            _markdown(
                "diagpfix",
                """
## Alternative component - Standalone fixation-probability plot

**Status: PENDING ADAPTER**

- Intended output: `figures/paper/fig-pfix.pdf`
- Planned entry point: `figures.plot_pfix_single`; historical implementation
  `figures.legacy.plot_pfix_single`.
- Relationship to manuscript: useful as a check, but the draft uses the
  combined Main Figure 3 layout.
""",
            ),
            _markdown(
                "diagemdclade",
                """
## Alternative assembly - Combined 9-month EMD and clade plot

**Status: PENDING ADAPTER**

- Intended output: `figures/paper/fig-emd-clade.pdf`
- Planned entry point: a canonical replacement for
  `figures.legacy.plot_emd_clade`.
- Canonical EMD and clade preparation are independently available, but the old
  combined renderer expects retired wide clade tables.
""",
            ),
            _markdown(
                "diagfreq",
                """
## Presentation alternative - Allele-frequency slide panels

**Status: RUNNABLE**

This preserved recipe is not a manuscript figure. It keeps the explicit H1
time shift, protein colors, slide canvas, and one canonical input per panel.
""",
            ),
            _code(
                "diagfreqrun",
                """
VERSION = "slides"
FIGSIZE_INCHES = f.FREQUENCY_SLIDE_CANVAS_INCHES
FREQUENCY_SLIDES = [
    {
        "input_file": ROOT / "data" / "plotting" / "inputs" / "frequencies" / "h3n2_ha_allele_frequencies.txt",
        "subtype": "h3n2", "segment": "ha", "time_shift_days": 0, "color": "#C35999",
        "output_file": ROOT / "figures" / "slides" / "fig-H3N2-HA-frequency-slides.pdf",
    },
    {
        "input_file": ROOT / "data" / "plotting" / "inputs" / "frequencies" / "h3n2_na_allele_frequencies.txt",
        "subtype": "h3n2", "segment": "na", "time_shift_days": 0, "color": "#DDA0C3",
        "output_file": ROOT / "figures" / "slides" / "fig-H3N2-NA-frequency-slides.pdf",
    },
    {
        "input_file": ROOT / "data" / "plotting" / "inputs" / "frequencies" / "h1n1_ha_allele_frequencies.txt",
        "subtype": "h1n1", "segment": "ha", "time_shift_days": 120, "color": "#81AB5B",
        "output_file": ROOT / "figures" / "slides" / "fig-H1N1-HA-frequency-slides.pdf",
    },
    {
        "input_file": ROOT / "data" / "plotting" / "inputs" / "frequencies" / "h1n1_na_allele_frequencies.txt",
        "subtype": "h1n1", "segment": "na", "time_shift_days": 120, "color": "#ABCE7D",
        "output_file": ROOT / "figures" / "slides" / "fig-H1N1-NA-frequency-slides.pdf",
    },
]

assert all(recipe["input_file"].is_file() for recipe in FREQUENCY_SLIDES)
if SAVE:
    for recipe in FREQUENCY_SLIDES:
        output = f.plot_frequency_slide(
            **recipe,
            version=VERSION,
            overwrite=OVERWRITE,
        )
        link_to(output)
""",
            ),
            _markdown(
                "modify01",
                """
## Modifying a plot

Start with the manuscript-numbered cell above. Change the visible data,
scientific, and output choices in that cell. A named wrapper in `figures.py`
keeps preparation and presentation together and prints the exact source file
containing the axes and marks. Historical functions remain available through
`figures.legacy`, but a **PENDING ADAPTER** entry should not be turned into a
run cell merely by pointing it at archived paths or retired schemas.
""",
            ),
        ]
    )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (influenza-figures)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def notebook_bytes() -> bytes:
    """Serialize the initial notebook template deterministically."""

    return (json.dumps(build_notebook(), indent=1, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def notebook_source_sha256(path: Path) -> str:
    """Hash ordered cell types and source text, ignoring rerun state."""

    notebook = json.loads(path.read_text(encoding="utf-8"))
    source_projection = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        source_projection.append(
            {
                "cell_type": cell.get("cell_type", ""),
                "source": source,
            }
        )
    serialized = json.dumps(
        source_projection,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256_bytes(serialized)


def replace_notebook(
    notebook_path: Path,
    archived_original: Path,
    *,
    force: bool = False,
) -> bool:
    """Install the template safely; return ``True`` only when it changed."""

    if not archived_original.is_file():
        raise NotebookReplacementRefused(
            f"Archived original is missing: {archived_original}"
        )
    archived_hash = sha256_file(archived_original)
    archived_source_hash = notebook_source_sha256(archived_original)
    if archived_source_hash != ARCHIVED_SOURCE_SHA256:
        raise NotebookReplacementRefused(
            "Archived notebook cell sources failed their pinned SHA-256 check: "
            f"expected {ARCHIVED_SOURCE_SHA256}, found {archived_source_hash}. "
            "Rerun outputs and metadata may change, but cell sources must be "
            "reviewed before this pin is updated."
        )

    generated = notebook_bytes()
    if notebook_path.is_file():
        current = notebook_path.read_bytes()
        if current == generated:
            return False
        if not force and sha256_bytes(current) != archived_hash:
            raise NotebookReplacementRefused(
                "Refusing to replace figures.ipynb because it contains working "
                "content. Preserve it or pass --force explicitly."
            )

    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{notebook_path.name}.", suffix=".tmp", dir=notebook_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(generated)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, notebook_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository containing figures.ipynb and the dated archive",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing working notebook (normally refused)",
    )
    arguments = parser.parse_args()
    repo_root = arguments.repo_root.resolve()
    notebook_path = repo_root / CURRENT_NOTEBOOK
    archived_path = repo_root / ARCHIVED_NOTEBOOK

    try:
        changed = replace_notebook(
            notebook_path,
            archived_path,
            force=arguments.force,
        )
    except NotebookReplacementRefused as error:
        parser.error(str(error))
    action = "created" if changed else "already current"
    print(f"Notebook {action}: {notebook_path} (sha256={sha256_file(notebook_path)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
