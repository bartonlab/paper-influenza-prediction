"""Readable plotting entry points for the influenza figures.

This is the scientist-facing plotting file. ``figures.ipynb`` calls the
functions defined here with explicit input and output paths. The more formal
registry and provenance records remain underneath these functions as safety
checks; a notebook user does not need to work with registry keys or canonical
method identifiers directly.

The historical plotting module is still available as ``figures.legacy``.
Names that have not yet been given a canonical wrapper are also looked up in
that module, so ordinary calls such as ``figures.plot_AUC(...)`` continue to
work. New work should prefer the small functions in this file because their
data, presentation version, and destination are explicit.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys
from typing import Any, Callable


# Make the local package importable when this repository has not been installed.
ROOT = Path(__file__).resolve().parent
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


from influenza_figures.emd_plot import (  # noqa: E402
    BASELINE_FULL_MODEL_METHOD,
    COMPARATOR_DISPLAY_LABELS,
    MLR_METHOD,
    NULL_METHOD,
    PROTOCOL_BY_HORIZON,
    SLM_METHOD,
    TARGET_METHOD_BY_HORIZON,
    render_emd_comparison_figure,
)
from influenza_figures.background_plot import (  # noqa: E402
    BackgroundRenderResult,
    INITIAL_ADVANTAGE_METRIC,
    render_background_clade_figure,
    render_background_growth_figure,
    render_background_initial_advantage_figure,
    render_background_selection_figure,
)
from influenza_figures.fixation_plot import (  # noqa: E402
    DEFAULT_MINIMUM_OBSERVATION_DAY,
    DEFAULT_MODEL_ID,
    DEFAULT_SOURCE_VARIANT,
    FixationRenderResult,
    SLM_MODEL_ID,
    SLM_SOURCE_VARIANT,
    render_fixation_auc_figure,
    render_pfix_dauc_figure,
)
from influenza_figures.registry import FIGURE_REGISTRY, FigureSpec  # noqa: E402
from influenza_figures.selection_plot import (  # noqa: E402
    HA_SELECTION_LIMIT_MAX,
    HA_SELECTION_TICK_MAX,
    MIN_MUTATION_COUNT,
    NA_SELECTION_LIMIT_MAX,
    NA_SELECTION_TICK_MAX,
    SASA_LIMIT_MAX,
    SASA_TICK_MAX,
    SelectionFigureRenderResult,
    render_frequency_selection_figure,
    render_selection_sasa_figure,
    render_selection_structure_space_figure,
)


PLOTTING_DATA = ROOT / "data" / "plotting"
FIGURE_DIR = ROOT / "figures"
PLOT_SOURCE_FILES = {
    "background": ROOT / "src" / "influenza_figures" / "background_plot.py",
    "emd": ROOT / "src" / "influenza_figures" / "emd_plot.py",
    "fixation": ROOT / "src" / "influenza_figures" / "fixation_plot.py",
    "legacy_figures": ROOT / "src" / "influenza_figures" / "legacy" / "figures.py",
    "selection": ROOT / "src" / "influenza_figures" / "selection_plot.py",
}

FIXATION_METRICS = PLOTTING_DATA / "derived" / "fixation_metrics"
FIXATION_PROBABILITY_DATA = FIXATION_METRICS / "fixation_probability.csv"
FIXATION_AUC_DATA = FIXATION_METRICS / "auc.csv"
BACKGROUND_DATA = PLOTTING_DATA / "inputs" / "background" / "population_fitness.csv"
CLADE_DATA = PLOTTING_DATA / "inputs" / "clades" / "h3n2_ha_frequencies.csv"

# These are deliberately visible here and again in the notebook. They encode
# the historical H3/N2/H1/N1 color assignment rather than an arbitrary cycle.
PROTEIN_COLORS = {
    ("h3n2", "ha"): "#C35999",
    ("h3n2", "na"): "#DDA0C3",
    ("h1n1", "ha"): "#81AB5B",
    ("h1n1", "na"): "#ABCE7D",
}

PAPER_STYLE = "paper_v1"
SLIDES_STYLE = "slides_v1"
CLADE_PAPER_CANVAS_INCHES = (3.4646, 4.1575)
FREQUENCY_SLIDE_CANVAS_INCHES = (8.5, 1.6)
FULL_MODEL_SOURCE_BY_HORIZON = {
    6: "cross-validation-selected best GenModel",
    9: "cross-validation-selected best GenModel",
    12: "cross-validation-selected best GenModel",
}


def _load_legacy() -> Any:
    """Load the archived-compatible plotting implementation only when needed."""

    module = import_module("influenza_figures.legacy.figures")
    globals()["legacy"] = module
    return module


def __getattr__(name: str) -> Any:
    """Expose ``legacy`` and preserve ordinary access to historical functions."""

    # Introspection tools probe module dunder attributes. Those are not legacy
    # plotting names and must not trigger heavyweight scientific imports.
    if name.startswith("__"):
        raise AttributeError(name)
    module = _load_legacy()
    if name == "legacy":
        return module
    try:
        return getattr(module, name)
    except AttributeError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error


def __dir__() -> list[str]:
    names = set(globals()) | {"legacy"}
    try:
        names.update(dir(_load_legacy()))
    except ModuleNotFoundError:
        pass
    return sorted(names)


def _output_path(output_file: str | Path | None, default: Path) -> Path:
    """Resolve an explicit path, or use a plainly stated repository default."""

    return default if output_file is None else Path(output_file).expanduser().resolve()


def _prepare_destination(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to replace {path}; pass overwrite=True to regenerate it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def _paper_recipe(
    figure_key: str,
    *,
    version: str,
    output_file: str | Path | None,
    overwrite: bool,
) -> tuple[FigureSpec, Path]:
    """Resolve one registered paper recipe and its requested destination."""

    if version != "paper":
        raise ValueError(
            "Canonical manuscript figures currently support version='paper' only"
        )
    figure_spec = FIGURE_REGISTRY.get(figure_key)
    destination = _output_path(output_file, FIGURE_DIR / figure_spec.output_name)
    _prepare_destination(destination, overwrite=overwrite)
    return figure_spec, destination


def _print_recipe(
    *,
    inputs: tuple[tuple[str, Path], ...],
    details: tuple[tuple[str, object], ...],
    figure_spec: FigureSpec,
    source_key: str,
    destination: Path,
) -> None:
    """Print the compact provenance needed to understand a notebook call."""

    label_width = 15
    for label, path in inputs:
        print(f"{label + ':':<{label_width}}{path}")
    for label, value in details:
        print(f"{label + ':':<{label_width}}{value}")
    print(
        f"{'Appearance:':<{label_width}}{figure_spec.profile}; "
        f"{figure_spec.canvas.name}; "
        f"{figure_spec.canvas.width_in:.3f} x "
        f"{figure_spec.canvas.height_in:.3f} inches"
    )
    print(f"{'Plot code:':<{label_width}}{PLOT_SOURCE_FILES[source_key]}")
    print(f"{'Output:':<{label_width}}{destination}")


def manuscript_statistics():
    """Return manuscript statistics as a readable pandas table.

    This is intentionally independent of figure generation: use it in the
    notebook whenever manuscript text or plot annotations need updating. Each
    row names the manuscript figure/panel, analysis, sample sizes, estimate,
    test statistic, p value, canonical data release, and source table.
    """

    try:
        import pandas as pd
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "pandas is required to display the manuscript statistics table"
        ) from error

    from influenza_figures.manuscript_stats import collect_manuscript_statistics

    rows = collect_manuscript_statistics(PLOTTING_DATA)
    return pd.DataFrame(row.as_record() for row in rows)


def plot_selection_structure_space(
    *,
    input_directory: str | Path = PLOTTING_DATA,
    minimum_mutation_count: int = MIN_MUTATION_COUNT,
    version: str = "paper",
    output_file: str | Path | None = None,
    overwrite: bool = True,
) -> SelectionFigureRenderResult:
    """Render Main Figure 2 from the canonical selection and structure data."""

    figure_spec, destination = _paper_recipe(
        "paper.selection_structure_space",
        version=version,
        output_file=output_file,
        overwrite=overwrite,
    )
    input_path = Path(input_directory).expanduser().resolve()
    _print_recipe(
        inputs=(("Input", input_path),),
        details=(
            ("Filter", f"mutation_count >= {minimum_mutation_count}"),
            (
                "Selection y",
                f"HA 0-{HA_SELECTION_LIMIT_MAX:g} "
                f"(top tick {HA_SELECTION_TICK_MAX:g}); "
                f"NA 0-{NA_SELECTION_LIMIT_MAX:g} "
                f"(top tick {NA_SELECTION_TICK_MAX:g})",
            ),
        ),
        figure_spec=figure_spec,
        source_key="selection",
        destination=destination,
    )
    return render_selection_structure_space_figure(
        input_path,
        figure_spec=figure_spec,
        minimum_mutation_count=minimum_mutation_count,
        output_path=destination,
    )


def plot_pfix_dauc(
    *,
    fixation_probability_file: str | Path = FIXATION_PROBABILITY_DATA,
    auc_file: str | Path = FIXATION_AUC_DATA,
    minimum_observation_day: int = DEFAULT_MINIMUM_OBSERVATION_DAY,
    version: str = "paper",
    output_file: str | Path | None = None,
    overwrite: bool = True,
) -> FixationRenderResult:
    """Render Main Figure 3 using full-model fixation and AUROC summaries."""

    figure_spec, destination = _paper_recipe(
        "paper.pfix_dauc",
        version=version,
        output_file=output_file,
        overwrite=overwrite,
    )
    fixation_path = Path(fixation_probability_file).expanduser().resolve()
    auc_path = Path(auc_file).expanduser().resolve()
    _print_recipe(
        inputs=(("Fixation", fixation_path), ("AUROC", auc_path)),
        details=(
            ("Model", DEFAULT_MODEL_ID),
            ("Source", DEFAULT_SOURCE_VARIANT),
            ("Warmup", f">= {minimum_observation_day} days"),
        ),
        figure_spec=figure_spec,
        source_key="fixation",
        destination=destination,
    )
    return render_pfix_dauc_figure(
        fixation_path,
        auc_path,
        figure_spec=figure_spec,
        model_id=DEFAULT_MODEL_ID,
        source_variant=DEFAULT_SOURCE_VARIANT,
        minimum_observation_day=minimum_observation_day,
        output_path=destination,
    )


def plot_background_clade(
    *,
    background_file: str | Path = BACKGROUND_DATA,
    clade_file: str | Path = CLADE_DATA,
    minimum_total_frequency: float = 1.0,
    start_date: str = "2011-08-01",
    end_date: str = "2015-06-01",
    forecast_baseline_date: str = "2014-06-01",
    version: str = "paper",
    output_file: str | Path | None = None,
    overwrite: bool = True,
) -> BackgroundRenderResult:
    """Render Main Figure 5 with one earliest eligible record per mutation."""

    figure_spec, destination = _paper_recipe(
        "paper.dw_clade",
        version=version,
        output_file=output_file,
        overwrite=overwrite,
    )
    background_path = Path(background_file).expanduser().resolve()
    clade_path = Path(clade_file).expanduser().resolve()
    advantage_y_min, advantage_y_max = (
        100.0 * value for value in INITIAL_ADVANTAGE_METRIC.y_limits
    )
    _print_recipe(
        inputs=(("Background", background_path), ("Clades", clade_path)),
        details=(
            (
                "Mutations",
                "duration >= 90 days; earliest eligible record per identity",
            ),
            (
                "Panel B range",
                f"{advantage_y_min:g}% to {advantage_y_max:g}%",
            ),
            ("Y-axis lines", "spines end at outer major ticks"),
            ("Clade filter", f"total observed frequency > {minimum_total_frequency}"),
            ("Dates", f"{start_date} to {end_date}"),
        ),
        figure_spec=figure_spec,
        source_key="background",
        destination=destination,
    )
    return render_background_clade_figure(
        background_path,
        clade_path,
        figure_spec=figure_spec,
        output_path=destination,
        minimum_total_frequency=minimum_total_frequency,
        start_date=start_date,
        end_date=end_date,
        forecast_baseline_date=forecast_baseline_date,
    )


def plot_frequency_selection(
    *,
    input_directory: str | Path = PLOTTING_DATA,
    minimum_mutation_count: int = MIN_MUTATION_COUNT,
    version: str = "paper",
    output_file: str | Path | None = None,
    overwrite: bool = True,
) -> SelectionFigureRenderResult:
    """Render Supplementary Figure S5 from canonical frequency/selection data."""

    figure_spec, destination = _paper_recipe(
        "paper.frequency_selection",
        version=version,
        output_file=output_file,
        overwrite=overwrite,
    )
    input_path = Path(input_directory).expanduser().resolve()
    _print_recipe(
        inputs=(("Input", input_path),),
        details=(("Filter", f"mutation_count >= {minimum_mutation_count}"),),
        figure_spec=figure_spec,
        source_key="selection",
        destination=destination,
    )
    return render_frequency_selection_figure(
        input_path,
        figure_spec=figure_spec,
        minimum_mutation_count=minimum_mutation_count,
        output_path=destination,
    )


def plot_selection_sasa(
    *,
    input_directory: str | Path = PLOTTING_DATA,
    version: str = "paper",
    output_file: str | Path | None = None,
    overwrite: bool = True,
) -> SelectionFigureRenderResult:
    """Render Supplementary Figure S6 from the consolidated SASA table."""

    figure_spec, destination = _paper_recipe(
        "paper.selection_sasa",
        version=version,
        output_file=output_file,
        overwrite=overwrite,
    )
    input_path = Path(input_directory).expanduser().resolve()
    _print_recipe(
        inputs=(("Input", input_path),),
        details=(
            ("Data", "smoothed selection and solvent accessibility"),
            (
                "Selection y",
                f"HA 0-{HA_SELECTION_LIMIT_MAX:g} "
                f"(top tick {HA_SELECTION_TICK_MAX:g}); "
                f"NA 0-{NA_SELECTION_LIMIT_MAX:g} "
                f"(top tick {NA_SELECTION_TICK_MAX:g})",
            ),
            (
                "SASA y",
                f"0-{SASA_LIMIT_MAX:g} (top tick {SASA_TICK_MAX:g})",
            ),
        ),
        figure_spec=figure_spec,
        source_key="selection",
        destination=destination,
    )
    return render_selection_sasa_figure(
        input_path,
        figure_spec=figure_spec,
        output_path=destination,
    )


def plot_fixation_auc(
    *,
    auc_file: str | Path = FIXATION_AUC_DATA,
    minimum_observation_day: int = DEFAULT_MINIMUM_OBSERVATION_DAY,
    version: str = "paper",
    output_file: str | Path | None = None,
    overwrite: bool = True,
) -> FixationRenderResult:
    """Render Supplementary Figure S7: full model versus neutral AUROC."""

    figure_spec, destination = _paper_recipe(
        "paper.auc",
        version=version,
        output_file=output_file,
        overwrite=overwrite,
    )
    input_path = Path(auc_file).expanduser().resolve()
    _print_recipe(
        inputs=(("Input", input_path),),
        details=(
            ("Model", DEFAULT_MODEL_ID),
            ("Source", DEFAULT_SOURCE_VARIANT),
            ("Warmup", f">= {minimum_observation_day} days"),
        ),
        figure_spec=figure_spec,
        source_key="fixation",
        destination=destination,
    )
    return render_fixation_auc_figure(
        input_path,
        figure_spec=figure_spec,
        model_id=DEFAULT_MODEL_ID,
        source_variant=DEFAULT_SOURCE_VARIANT,
        minimum_observation_day=minimum_observation_day,
        model_display_label="model",
        output_path=destination,
    )


def _plot_background_summary(
    *,
    figure_key: str,
    metric: str,
    duration: str,
    renderer: Callable[..., BackgroundRenderResult],
    background_file: str | Path,
    version: str,
    output_file: str | Path | None,
    overwrite: bool,
) -> BackgroundRenderResult:
    """Shared transparent adapter for the three background-only figures."""

    figure_spec, destination = _paper_recipe(
        figure_key,
        version=version,
        output_file=output_file,
        overwrite=overwrite,
    )
    input_path = Path(background_file).expanduser().resolve()
    _print_recipe(
        inputs=(("Input", input_path),),
        details=(("Metric", metric), ("Mutations", duration)),
        figure_spec=figure_spec,
        source_key="background",
        destination=destination,
    )
    return renderer(
        input_path,
        figure_spec=figure_spec,
        output_path=destination,
    )


def plot_background_selection(
    *,
    background_file: str | Path = BACKGROUND_DATA,
    version: str = "paper",
    output_file: str | Path | None = None,
    overwrite: bool = True,
) -> BackgroundRenderResult:
    """Render Supplementary Figure S13: mutation selection by outcome."""

    return _plot_background_summary(
        figure_key="paper.ds",
        metric="selection coefficient",
        duration="earliest record per identity; no duration filter",
        renderer=render_background_selection_figure,
        background_file=background_file,
        version=version,
        output_file=output_file,
        overwrite=overwrite,
    )


def plot_background_initial_advantage(
    *,
    background_file: str | Path = BACKGROUND_DATA,
    version: str = "paper",
    output_file: str | Path | None = None,
    overwrite: bool = True,
) -> BackgroundRenderResult:
    """Render Supplementary Figure S14: initial background advantage."""

    return _plot_background_summary(
        figure_key="paper.dw_bg",
        metric="initial background fitness minus population average",
        duration="duration >= 90 days; earliest eligible record per identity",
        renderer=render_background_initial_advantage_figure,
        background_file=background_file,
        version=version,
        output_file=output_file,
        overwrite=overwrite,
    )


def plot_background_growth(
    *,
    background_file: str | Path = BACKGROUND_DATA,
    version: str = "paper",
    output_file: str | Path | None = None,
    overwrite: bool = True,
) -> BackgroundRenderResult:
    """Render Supplementary Figure S15: annualized background-fitness growth."""

    return _plot_background_summary(
        figure_key="paper.dw_rate",
        metric="annualized background-fitness change",
        duration="duration >= 90 days; earliest eligible record per identity",
        renderer=render_background_growth_figure,
        background_file=background_file,
        version=version,
        output_file=output_file,
        overwrite=overwrite,
    )


def plot_single_site_auc(
    *,
    auc_file: str | Path = FIXATION_AUC_DATA,
    minimum_observation_day: int = DEFAULT_MINIMUM_OBSERVATION_DAY,
    version: str = "paper",
    output_file: str | Path | None = None,
    overwrite: bool = True,
) -> FixationRenderResult:
    """Render Supplementary Figure S16: single-site model versus neutral AUROC."""

    figure_spec, destination = _paper_recipe(
        "paper.auc_ss",
        version=version,
        output_file=output_file,
        overwrite=overwrite,
    )
    input_path = Path(auc_file).expanduser().resolve()
    _print_recipe(
        inputs=(("Input", input_path),),
        details=(
            ("Model", SLM_MODEL_ID),
            ("Source", SLM_SOURCE_VARIANT),
            ("Warmup", f">= {minimum_observation_day} days"),
        ),
        figure_spec=figure_spec,
        source_key="fixation",
        destination=destination,
    )
    return render_fixation_auc_figure(
        input_path,
        figure_spec=figure_spec,
        model_id=SLM_MODEL_ID,
        source_variant=SLM_SOURCE_VARIANT,
        minimum_observation_day=minimum_observation_day,
        model_display_label="single site",
        output_path=destination,
    )


def _emd_method(comparator: str) -> tuple[str, str]:
    """Translate the three human comparator names accepted by the notebook."""

    choices = {
        "null": ("null", NULL_METHOD),
        "neutral": ("null", NULL_METHOD),
        "slm": ("slm", SLM_METHOD),
        "single locus": ("slm", SLM_METHOD),
        "single-locus": ("slm", SLM_METHOD),
        "mlr": ("mlr", MLR_METHOD),
        "multinomial logistic regression": ("mlr", MLR_METHOD),
    }
    normalized = comparator.strip().lower()
    try:
        return choices[normalized]
    except KeyError as error:
        raise ValueError(
            "comparator must be 'null', 'SLM', or 'MLR' "
            "(multinomial logistic regression)"
        ) from error


def plot_emd_comparison(
    comparator: str,
    horizon_months: int,
    *,
    input_directory: str | Path = PLOTTING_DATA / "inputs" / "emd",
    version: str = "paper",
    output_file: str | Path | None = None,
    overwrite: bool = True,
):
    """Plot full-model EMD against null, SLM, or MLR.

    The comparison is paired by exact evaluation ID. All three horizons use
    complete ``results_grid`` evaluations and ``best GenModel`` with
    cross-validation-selected hyperparameters. The canonical renderer applies
    the inclusive 1,095-day warmup separately from the H3N2 and H1N1 inference
    start dates and prints both the relative and calendar-date cutoffs.

    ``version='paper'`` selects the paper typography and the canvas registered
    for this particular comparator. A canonical slide-specific EMD layout is
    not yet implemented; the function refuses that request instead of silently
    stretching a paper figure.
    """

    if version != "paper":
        raise ValueError(
            "Only version='paper' is currently validated for canonical EMD plots; "
            "the slide layout is still pending."
        )

    short_name, method_id = _emd_method(comparator)
    if horizon_months not in PROTOCOL_BY_HORIZON:
        raise ValueError("horizon_months must be 6, 9, or 12")

    figure_key = f"paper.emd_vs_{short_name}.{horizon_months}m"
    figure_spec = FIGURE_REGISTRY.get(figure_key)
    input_path = Path(input_directory).expanduser().resolve()
    destination = _output_path(
        output_file,
        FIGURE_DIR / figure_spec.output_name,
    )
    _prepare_destination(destination, overwrite=overwrite)

    print(f"Input:      {input_path}")
    print(f"Comparison: full model vs {COMPARATOR_DISPLAY_LABELS[method_id]}")
    print(f"Horizon:    {horizon_months} months ({PROTOCOL_BY_HORIZON[horizon_months]})")
    print(
        f"Full model: {FULL_MODEL_SOURCE_BY_HORIZON[horizon_months]} "
        f"[{TARGET_METHOD_BY_HORIZON[horizon_months]}]"
    )
    print(
        f"Appearance: {figure_spec.profile}; {figure_spec.canvas.name}; "
        f"{figure_spec.canvas.width_in:.3f} x {figure_spec.canvas.height_in:.3f} inches"
    )
    print(f"Plot code:  {PLOT_SOURCE_FILES['emd']}")
    print(f"Output:     {destination}")

    return render_emd_comparison_figure(
        input_path,
        horizon_months=horizon_months,
        comparator_method=method_id,
        figure_spec=figure_spec,
        output_path=destination,
    )


def plot_emd_best_vs_baseline(
    horizon_months: int,
    *,
    input_directory: str | Path = PLOTTING_DATA / "inputs" / "emd",
    version: str = "paper",
    output_file: str | Path | None = None,
    overwrite: bool = True,
):
    """Plot a cross-validated full model against its same-run baseline.

    This is a diagnostic rather than one of the three primary paper comparator
    figures. It uses exact ``results_grid`` evaluation pairs after the same
    inclusive 1,095-day warmup. Delta EMD is baseline minus best, so positive
    values favor the cross-validation-selected ``best GenModel``.
    """

    if version != "paper":
        raise ValueError(
            "Only version='paper' is currently validated for the best-vs-baseline "
            "EMD diagnostic."
        )

    if horizon_months not in PROTOCOL_BY_HORIZON:
        raise ValueError("horizon_months must be 6, 9, or 12")

    figure_spec = FIGURE_REGISTRY.get(
        f"diagnostic.emd_best_vs_baseline.{horizon_months}m"
    )
    input_path = Path(input_directory).expanduser().resolve()
    destination = _output_path(
        output_file,
        FIGURE_DIR / figure_spec.output_name,
    )
    _prepare_destination(destination, overwrite=overwrite)

    print(f"Input:      {input_path}")
    print("Comparison: cross-validation-selected best GenModel vs baseline GenModel")
    print("Delta:      baseline EMD - best EMD (positive favors best GenModel)")
    print(f"Horizon:    {horizon_months} months ({PROTOCOL_BY_HORIZON[horizon_months]})")
    print(
        f"Full model: {FULL_MODEL_SOURCE_BY_HORIZON[horizon_months]} "
        f"[{TARGET_METHOD_BY_HORIZON[horizon_months]}]"
    )
    print(f"Baseline:   GenModel (baseline) [{BASELINE_FULL_MODEL_METHOD}]")
    print(
        f"Appearance: {figure_spec.profile}; {figure_spec.canvas.name}; "
        f"{figure_spec.canvas.width_in:.3f} x {figure_spec.canvas.height_in:.3f} inches"
    )
    print(f"Plot code:  {PLOT_SOURCE_FILES['emd']}")
    print(f"Output:     {destination}")

    return render_emd_comparison_figure(
        input_path,
        horizon_months=horizon_months,
        comparator_method=BASELINE_FULL_MODEL_METHOD,
        figure_spec=figure_spec,
        output_path=destination,
    )


def plot_emd_best_vs_baseline_9m(
    *,
    input_directory: str | Path = PLOTTING_DATA / "inputs" / "emd",
    version: str = "paper",
    output_file: str | Path | None = None,
    overwrite: bool = True,
):
    """Compatibility wrapper for the nine-month best-vs-baseline diagnostic."""

    return plot_emd_best_vs_baseline(
        9,
        input_directory=input_directory,
        version=version,
        output_file=output_file,
        overwrite=overwrite,
    )


def plot_clade_figure(
    *,
    input_file: str | Path = (
        PLOTTING_DATA / "inputs" / "clades" / "h3n2_ha_frequencies.csv"
    ),
    output_file: str | Path = FIGURE_DIR / "paper" / "fig-clade.pdf",
    version: str = "paper",
    minimum_total_frequency: float = 1.0,
    start_date: str = "2011-08-01",
    end_date: str = "2015-06-01",
    forecast_baseline_date: str = "2014-06-01",
    overwrite: bool = True,
) -> dict[str, Any]:
    """Prepare canonical clade frequencies and render the paper clade figure.

    Clades are selected from the complete observed series before date filtering
    and per-date normalization. The forecast baseline date is omitted here
    because the historical renderer explicitly prepends the last observed
    history row as its baseline; retaining it twice would duplicate that date.
    """

    if version != "paper":
        raise ValueError("plot_clade_figure currently supports version='paper' only")

    try:
        import pandas as pd
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("pandas is required to plot the clade figure") from error

    from influenza_figures.clades import prepare_clade_frequencies
    from influenza_figures.render import legacy_style_context

    source_path = Path(input_file).expanduser().resolve()
    destination = Path(output_file).expanduser().resolve()
    _prepare_destination(destination, overwrite=overwrite)

    source = pd.read_csv(source_path)
    prepared = prepare_clade_frequencies(
        source,
        reference_series="observed_full",
        minimum_total_frequency=minimum_total_frequency,
        reference_start_date=start_date,
        reference_end_date=end_date,
    )
    data = prepared.data
    past = data[
        (data["series"] == "observed_history") & (data["date"] > start_date)
    ].copy()
    full = data[
        (data["series"] == "observed_full")
        & (data["date"] > start_date)
        & (data["date"] < end_date)
    ].copy()
    forecast = data[
        (data["series"] == "forecast")
        & (data["date"] != forecast_baseline_date)
    ].copy()
    forecast_frames = [
        group.copy() for _, group in forecast.groupby("date", sort=True)
    ]
    if len(forecast_frames) != 4:
        raise ValueError(
            "The clade figure expects four forecast dates (3, 6, 9, and 12 months); "
            f"found {len(forecast_frames)}."
        )

    print(f"Input:      {source_path}")
    print(f"Selection:  total observed frequency > {minimum_total_frequency}")
    print(f"Dates:      {start_date} < date < {end_date}")
    print(f"Clades:     {', '.join(prepared.selected_clades)}")
    print(
        f"Appearance: {PAPER_STYLE}; legacy single-column canvas; "
        f"{CLADE_PAPER_CANVAS_INCHES[0]:.3f} x "
        f"{CLADE_PAPER_CANVAS_INCHES[1]:.3f} inches"
    )
    print(f"Plot code:  {PLOT_SOURCE_FILES['legacy_figures']} (plot_clade)")
    print(f"Output:     {destination}")

    legacy_module = _load_legacy()
    old_figure_directory = legacy_module.FIG_DIR
    legacy_module.FIG_DIR = str(destination.parent)
    try:
        with legacy_style_context(PAPER_STYLE):
            # The historical renderer mutates its data frames, hence the copies.
            legacy_module.plot_clade(
                past.copy(),
                full.copy(),
                [frame.copy() for frame in forecast_frames],
                fig_name=destination.name,
            )
    finally:
        legacy_module.FIG_DIR = old_figure_directory

    if not destination.is_file():
        raise RuntimeError(f"The clade renderer did not create {destination}")
    return {
        "output_file": destination,
        "selected_clades": prepared.selected_clades,
        "row_counts": {
            "observed_history": len(past),
            "observed_full": len(full),
            "forecast": len(forecast),
        },
    }


def plot_frequency_slide(
    input_file: str | Path,
    *,
    subtype: str,
    segment: str,
    time_shift_days: int,
    output_file: str | Path,
    color: str | None = None,
    version: str = "slides",
    overwrite: bool = True,
) -> Path:
    """Render one allele-frequency slide panel from a canonical matrix.

    The H1N1 series in the historical figure use a 120-day time shift, while
    H3N2 uses zero. The shift is required explicitly so that this scientific
    alignment choice remains visible in every notebook call.
    """

    if version != "slides":
        raise ValueError("plot_frequency_slide currently supports version='slides' only")

    normalized_key = (subtype.strip().lower(), segment.strip().lower())
    try:
        plot_color = PROTEIN_COLORS[normalized_key] if color is None else color
    except KeyError as error:
        raise ValueError("subtype/segment must be H3N2 or H1N1 and HA or NA") from error

    from influenza_figures.render import legacy_style_context

    source_path = Path(input_file).expanduser().resolve()
    destination = Path(output_file).expanduser().resolve()
    _prepare_destination(destination, overwrite=overwrite)

    print(f"Input:      {source_path}")
    print(f"Protein:    {normalized_key[0].upper()} {normalized_key[1].upper()}")
    print(f"Time shift: {time_shift_days} days")
    print(
        f"Appearance: {SLIDES_STYLE}; "
        f"{FREQUENCY_SLIDE_CANVAS_INCHES[0]:.1f} x "
        f"{FREQUENCY_SLIDE_CANVAS_INCHES[1]:.1f} inches"
    )
    print(
        f"Plot code:  {PLOT_SOURCE_FILES['legacy_figures']} "
        "(plot_frequency_slides)"
    )
    print(f"Output:     {destination}")

    legacy_module = _load_legacy()
    old_figure_directory = legacy_module.FIG_DIR
    legacy_module.FIG_DIR = str(destination.parent)
    try:
        with legacy_style_context(SLIDES_STYLE):
            legacy_module.plot_frequency_slides(
                str(source_path),
                deltat=time_shift_days,
                color=plot_color,
                fig_name=destination.name,
            )
    finally:
        legacy_module.FIG_DIR = old_figure_directory

    if not destination.is_file():
        raise RuntimeError(f"The frequency renderer did not create {destination}")
    return destination


__all__ = [
    "BACKGROUND_DATA",
    "CLADE_PAPER_CANVAS_INCHES",
    "CLADE_DATA",
    "FIGURE_DIR",
    "FIXATION_AUC_DATA",
    "FIXATION_METRICS",
    "FIXATION_PROBABILITY_DATA",
    "FREQUENCY_SLIDE_CANVAS_INCHES",
    "PAPER_STYLE",
    "PLOT_SOURCE_FILES",
    "PLOTTING_DATA",
    "PROTEIN_COLORS",
    "ROOT",
    "SLIDES_STYLE",
    "legacy",
    "plot_background_clade",
    "plot_background_growth",
    "plot_background_initial_advantage",
    "plot_background_selection",
    "plot_clade_figure",
    "plot_emd_best_vs_baseline",
    "plot_emd_best_vs_baseline_9m",
    "plot_emd_comparison",
    "plot_fixation_auc",
    "plot_frequency_selection",
    "plot_frequency_slide",
    "plot_pfix_dauc",
    "plot_selection_sasa",
    "plot_selection_structure_space",
    "plot_single_site_auc",
    "manuscript_statistics",
]
