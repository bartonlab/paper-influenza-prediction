"""Explicit canvas and output ownership registry for figure recipes."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping

from .styles import STYLES


def cm_to_inches(value: float) -> float:
    return value / 2.54


SINGLE_COLUMN = cm_to_inches(8.8)
ONE_FIVE_COLUMN = cm_to_inches(12.1)
DOUBLE_COLUMN = cm_to_inches(18.4)
SLIDE_WIDTH = 10.5
GOLDEN_RATIO = (1.0 + sqrt(5.0)) / 2.0


@dataclass(frozen=True, slots=True)
class Canvas:
    """A named physical canvas measured in inches."""

    name: str
    width_in: float
    height_in: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("canvas name must not be empty")
        for label, value in (("width_in", self.width_in), ("height_in", self.height_in)):
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be finite and positive; got {value!r}")

    @property
    def size_inches(self) -> tuple[float, float]:
        return (self.width_in, self.height_in)

    @property
    def size_points(self) -> tuple[float, float]:
        return (self.width_in * 72.0, self.height_in * 72.0)


def _canvas(name: str, width: float, height: float) -> Canvas:
    return Canvas(name=name, width_in=width, height_in=height)


_CANVAS_LIST = (
    _canvas("model_overview_artwork", 250.0 / 72.0, 194.0 / 72.0),
    _canvas("overview_strip", 1.6, 0.7),
    _canvas("slide_overview_fixloss", SLIDE_WIDTH, 0.3 * SLIDE_WIDTH),
    _canvas("double_half", DOUBLE_COLUMN, DOUBLE_COLUMN / 2.0),
    _canvas("double_two_thirds", DOUBLE_COLUMN, DOUBLE_COLUMN / 1.5),
    _canvas("double_over_1_8", DOUBLE_COLUMN, DOUBLE_COLUMN / 1.8),
    _canvas("single_over_1_5", SINGLE_COLUMN, SINGLE_COLUMN / 1.5),
    _canvas("double_over_1_9", DOUBLE_COLUMN, DOUBLE_COLUMN / 1.9),
    _canvas("double_over_2_2", DOUBLE_COLUMN, DOUBLE_COLUMN / 2.2),
    _canvas("double_over_1_6", DOUBLE_COLUMN, DOUBLE_COLUMN / 1.6),
    _canvas("one_five_over_1_6", ONE_FIVE_COLUMN, ONE_FIVE_COLUMN / 1.6),
    _canvas("one_five_square", ONE_FIVE_COLUMN, ONE_FIVE_COLUMN),
    _canvas("double_over_2_8", DOUBLE_COLUMN, DOUBLE_COLUMN / 2.8),
    _canvas("single_square", SINGLE_COLUMN, SINGLE_COLUMN),
    _canvas("single_tall_1_5", SINGLE_COLUMN, SINGLE_COLUMN * 1.5),
    _canvas("single_tall_1_2", SINGLE_COLUMN, SINGLE_COLUMN * 1.2),
    _canvas("one_five_over_1_4", ONE_FIVE_COLUMN, ONE_FIVE_COLUMN / 1.4),
    _canvas("slide_distribution", 5.0, 3.0),
    _canvas("grant_auc", 3.35, 3.35 * 0.60),
    _canvas("grant_overview", 3.35, 3.35 * 1.2),
    _canvas("slide_wide_short", SLIDE_WIDTH, SLIDE_WIDTH / 2.5),
    _canvas("slide_overview_true", SLIDE_WIDTH / 3.5, 2.0),
    _canvas("slide_overview_sim", SLIDE_WIDTH / 2.5, 2.0),
    _canvas("slide_overview_combined", SLIDE_WIDTH, 3.0),
    _canvas("slide_frequency", 8.5, 1.6),
    _canvas("slide_selection", 3.4, 1.6),
    _canvas("slide_selection_subset", 5.25, 5.25 / GOLDEN_RATIO),
    _canvas("slide_example_frequency", SLIDE_WIDTH, 3.5),
    _canvas("slide_auc", SLIDE_WIDTH / 4.0, SLIDE_WIDTH / 2.0),
    _canvas("slide_pfix", SLIDE_WIDTH / 2.0, SLIDE_WIDTH / 2.0),
    _canvas("slide_emd", SLIDE_WIDTH, SLIDE_WIDTH / 1.8),
    _canvas("slide_emd_joint", SLIDE_WIDTH, SLIDE_WIDTH / 2.0),
)

CANVASES: Mapping[str, Canvas] = MappingProxyType(
    {canvas.name: canvas for canvas in _CANVAS_LIST}
)


def _safe_output_name(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.name or ".." in path.parts:
        raise ValueError(f"output_name must be a safe relative path; got {value!r}")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class FigureSpec:
    """Immutable ownership and presentation metadata for one figure recipe."""

    key: str
    renderer: str
    profile: str
    canvas: Canvas
    output_name: str
    category: str
    layout: str = "custom"
    rc_overrides: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""
    recipe_parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key or not self.renderer or not self.category:
            raise ValueError("key, renderer, and category must not be empty")
        if self.profile not in STYLES:
            raise ValueError(f"unknown style profile {self.profile!r}")
        object.__setattr__(self, "output_name", _safe_output_name(self.output_name))
        object.__setattr__(self, "rc_overrides", MappingProxyType(dict(self.rc_overrides)))
        object.__setattr__(
            self,
            "recipe_parameters",
            MappingProxyType(dict(self.recipe_parameters)),
        )


class RegistryCollisionError(ValueError):
    """Raised when two figure specs claim the same key or output path."""


class FigureRegistry:
    """Read-only registry with case-insensitive output ownership checks."""

    def __init__(self, specs: Iterable[FigureSpec]):
        by_key: dict[str, FigureSpec] = {}
        owners: dict[str, str] = {}
        canonical_outputs: dict[str, str] = {}

        for spec in specs:
            if spec.key in by_key:
                raise RegistryCollisionError(f"duplicate figure key: {spec.key!r}")
            folded = spec.output_name.casefold()
            if folded in owners:
                first = owners[folded]
                raise RegistryCollisionError(
                    f"output {spec.output_name!r} is claimed by both {first!r} and {spec.key!r}"
                )
            by_key[spec.key] = spec
            owners[folded] = spec.key
            canonical_outputs[folded] = spec.output_name

        self._by_key = MappingProxyType(by_key)
        self._owners = MappingProxyType(owners)
        self._canonical_outputs = MappingProxyType(canonical_outputs)

    def __len__(self) -> int:
        return len(self._by_key)

    def __iter__(self) -> Iterator[FigureSpec]:
        return iter(self._by_key.values())

    def __contains__(self, key: object) -> bool:
        return key in self._by_key

    def get(self, key: str) -> FigureSpec:
        try:
            return self._by_key[key]
        except KeyError as exc:
            raise KeyError(f"unknown figure key: {key!r}") from exc

    def owner_of(self, output_name: str) -> str:
        safe_name = _safe_output_name(output_name)
        try:
            return self._owners[safe_name.casefold()]
        except KeyError as exc:
            raise KeyError(f"unregistered output: {safe_name!r}") from exc

    @property
    def specs(self) -> tuple[FigureSpec, ...]:
        return tuple(self._by_key.values())


def _spec(
    key: str,
    renderer: str,
    profile: str,
    canvas: str,
    output_name: str,
    category: str,
    *,
    layout: str = "custom",
    notes: str = "",
    recipe_parameters: Mapping[str, Any] | None = None,
) -> FigureSpec:
    output_directory = {
        "paper": "paper",
        "supplement": "supplement",
        "slides": "slides",
        "grant": "grant",
        "diagnostic": "diagnostic",
        "legacy": "diagnostic",
    }[category]
    if "/" not in output_name:
        output_name = f"{output_directory}/{output_name}"
    return FigureSpec(
        key=key,
        renderer=renderer,
        profile=profile,
        canvas=CANVASES[canvas],
        output_name=output_name,
        category=category,
        layout=layout,
        notes=notes,
        recipe_parameters={} if recipe_parameters is None else recipe_parameters,
    )


_FIGURE_SPECS = (
    _spec(
        "paper.model_overview",
        "manual_export_powerpoint",
        "paper_v1",
        "model_overview_artwork",
        "fig-model-overview.pdf",
        "paper",
        layout="externally_assembled_artwork",
        notes=(
            "Main Fig. 1 is maintained as editable PowerPoint artwork under "
            "data/plotting/inputs/artwork; export it to PDF without resizing."
        ),
    ),
    _spec("paper.overview_true", "plot_overview_schematic_true", "paper_v1", "overview_strip", "fig-overview-true.pdf", "paper", layout="single_axis"),
    _spec("paper.overview_sim", "plot_overview_schematic_sim", "paper_v1", "overview_strip", "fig-overview-sim.pdf", "paper", layout="single_axis"),
    _spec("slides.overview_fixloss", "plot_overview_schematic_fixloss", "slides_v1", "slide_overview_fixloss", "fig-overview-fixloss.pdf", "slides", layout="single_axis", notes="Legacy function lacks a _slides suffix but uses SLIDE_WIDTH."),
    _spec("paper.selection_structure_space", "plot_selection_structure_space", "paper_v1", "double_half", "fig-selection-structure-space.pdf", "paper"),
    _spec("paper.selection_sasa", "plot_selection_sasa", "paper_v1", "double_half", "fig-selection-sasa.pdf", "supplement", notes="Supplementary Fig. S6 in the 2026-10-13 manuscript draft; key retained for compatibility."),
    _spec("paper.frequency_selection_structure", "plot_frequency_selection_structure", "paper_v1", "double_two_thirds", "fig-frequency-selection-structure.pdf", "paper"),
    _spec("paper.frequency_selection", "plot_frequency_selection", "paper_v1", "double_over_1_8", "fig-frequency-selection.pdf", "supplement", notes="Supplementary Fig. S5 in the 2026-10-13 manuscript draft; key retained for compatibility."),
    _spec("paper.emd_vs_mlr.6m", "render_emd_comparison_figure", "paper_v1", "double_over_1_9", "fig-emd-vs-mlr-6m.pdf", "supplement", layout="four_by_two_trajectory_and_distribution", notes="Supplementary Fig. S10. Canonical comparator: multinomial_logistic_regression_rare_filtered; results_grid. Key retained for compatibility." , recipe_parameters={"horizon_months": 6, "comparator_method": "multinomial_logistic_regression_rare_filtered"}),
    _spec("paper.emd_vs_mlr.9m", "render_emd_comparison_figure", "paper_v1", "double_over_1_9", "fig-emd-vs-mlr-9m.pdf", "supplement", layout="four_by_two_trajectory_and_distribution", notes="Supplementary Fig. S11. Canonical comparator: multinomial_logistic_regression_rare_filtered; results_grid. Key retained for compatibility.", recipe_parameters={"horizon_months": 9, "comparator_method": "multinomial_logistic_regression_rare_filtered"}),
    _spec("paper.emd_vs_mlr.12m", "render_emd_comparison_figure", "paper_v1", "double_over_1_9", "fig-emd-vs-mlr-12m.pdf", "supplement", layout="four_by_two_trajectory_and_distribution", notes="Supplementary Fig. S12. Canonical comparator: multinomial_logistic_regression_rare_filtered; results_grid. Key retained for compatibility.", recipe_parameters={"horizon_months": 12, "comparator_method": "multinomial_logistic_regression_rare_filtered"}),
    _spec("paper.emd_vs_slm.6m", "render_emd_comparison_figure", "paper_v1", "double_over_1_9", "fig-emd-vs-slm-6m.pdf", "supplement", layout="four_by_two_trajectory_and_distribution", notes="Supplementary Fig. S17. Canonical comparator: single_locus_model_rare_filtered; results_grid.", recipe_parameters={"horizon_months": 6, "comparator_method": "single_locus_model_rare_filtered"}),
    _spec("paper.emd_vs_slm.9m", "render_emd_comparison_figure", "paper_v1", "double_over_1_9", "fig-emd-vs-slm-9m.pdf", "supplement", layout="four_by_two_trajectory_and_distribution", notes="Supplementary Fig. S18. Canonical comparator: single_locus_model_rare_filtered; results_grid.", recipe_parameters={"horizon_months": 9, "comparator_method": "single_locus_model_rare_filtered"}),
    _spec("paper.emd_vs_slm.12m", "render_emd_comparison_figure", "paper_v1", "double_over_1_9", "fig-emd-vs-slm-12m.pdf", "supplement", layout="four_by_two_trajectory_and_distribution", notes="Supplementary Fig. S19. Canonical comparator: single_locus_model_rare_filtered; results_grid.", recipe_parameters={"horizon_months": 12, "comparator_method": "single_locus_model_rare_filtered"}),
    _spec("paper.emd_vs_null.6m", "render_emd_comparison_figure", "paper_v1", "double_over_2_2", "fig-emd-vs-null-6m.pdf", "supplement", layout="four_by_two_trajectory_and_distribution", notes="Supplementary Fig. S8. Canonical comparator: null; results_grid. Key retained for compatibility.", recipe_parameters={"horizon_months": 6, "comparator_method": "null"}),
    _spec("paper.emd_vs_null.9m", "render_emd_comparison_figure", "paper_v1", "double_over_2_2", "fig-emd-vs-null-9m.pdf", "paper", layout="four_by_two_trajectory_and_distribution", notes="Main Fig. 4. Canonical comparator: null; results_grid.", recipe_parameters={"horizon_months": 9, "comparator_method": "null"}),
    _spec("paper.emd_vs_null.12m", "render_emd_comparison_figure", "paper_v1", "double_over_2_2", "fig-emd-vs-null-12m.pdf", "supplement", layout="four_by_two_trajectory_and_distribution", notes="Supplementary Fig. S9. Canonical comparator: null; results_grid. Key retained for compatibility.", recipe_parameters={"horizon_months": 12, "comparator_method": "null"}),
    _spec("diagnostic.emd_best_vs_baseline.6m", "render_emd_comparison_figure", "paper_v1", "double_over_1_9", "fig-emd-best-vs-baseline-6m.pdf", "diagnostic", layout="four_by_two_trajectory_and_distribution", notes="Diagnostic comparison of the cross-validation-selected full model against its same-run baseline; plotted delta is baseline minus best, so positive values favor the selected model.", recipe_parameters={"horizon_months": 6, "target_method": "full_model_cross_validated_rare_filtered", "comparator_method": "full_model_rare_filtered", "difference_sign": "comparator_minus_target"}),
    _spec("diagnostic.emd_best_vs_baseline.9m", "render_emd_comparison_figure", "paper_v1", "double_over_1_9", "fig-emd-best-vs-baseline-9m.pdf", "diagnostic", layout="four_by_two_trajectory_and_distribution", notes="Diagnostic comparison of the cross-validation-selected full model against its same-run baseline; plotted delta is baseline minus best, so positive values favor the selected model.", recipe_parameters={"horizon_months": 9, "target_method": "full_model_cross_validated_rare_filtered", "comparator_method": "full_model_rare_filtered", "difference_sign": "comparator_minus_target"}),
    _spec("diagnostic.emd_best_vs_baseline.12m", "render_emd_comparison_figure", "paper_v1", "double_over_1_9", "fig-emd-best-vs-baseline-12m.pdf", "diagnostic", layout="four_by_two_trajectory_and_distribution", notes="Diagnostic comparison of the cross-validation-selected full model against its same-run baseline; plotted delta is baseline minus best, so positive values favor the selected model.", recipe_parameters={"horizon_months": 12, "target_method": "full_model_cross_validated_rare_filtered", "comparator_method": "full_model_rare_filtered", "difference_sign": "comparator_minus_target"}),
    _spec("paper.emd_clade", "plot_emd_clade", "paper_v1", "double_over_1_6", "fig-emd-clade.pdf", "paper"),
    _spec("legacy.clade_old", "plot_clade_old", "paper_v1", "one_five_over_1_6", "fig-clade-old.pdf", "legacy"),
    _spec("legacy.pfix_grid", "plot_pfix", "paper_v1", "one_five_square", "fig-pfix-grid.pdf", "legacy"),
    _spec("paper.pfix_dauc", "plot_pfix_dauc", "paper_v1", "double_over_2_8", "fig-pfix-dauc.pdf", "paper"),
    _spec("paper.pfix", "plot_pfix_single", "paper_v1", "single_square", "fig-pfix.pdf", "paper"),
    _spec("paper.auc", "plot_fixation_auc", "paper_v1", "single_tall_1_5", "fig-fixation-auc.pdf", "supplement", layout="two_by_two", notes="Supplementary Fig. S7; full-model versus neutral AUROC."),
    _spec("paper.auc_ss", "plot_single_site_auc", "paper_v1", "single_tall_1_5", "fig-auc-ss.pdf", "supplement", layout="two_by_two", notes="Supplementary Fig. S16; rare-filtered single-locus model versus neutral AUROC."),
    _spec("paper.dw", "plot_dw", "paper_v1", "single_square", "fig-dw.pdf", "paper"),
    _spec("paper.clade", "plot_clade", "paper_v1", "single_tall_1_2", "fig-clade.pdf", "paper"),
    _spec("paper.dw_clade", "plot_background_clade", "paper_v1", "one_five_over_1_4", "fig-dw-clade.pdf", "paper"),
    _spec("paper.dw_rate", "plot_background_growth", "paper_v1", "one_five_square", "fig-dw-rate.pdf", "supplement", notes="Supplementary Fig. S15."),
    _spec("paper.dw_bg", "plot_background_initial_advantage", "paper_v1", "one_five_square", "fig-dw-bg.pdf", "supplement", notes="Supplementary Fig. S14."),
    _spec("paper.ds", "plot_background_selection", "paper_v1", "one_five_square", "fig-ds.pdf", "supplement", notes="Supplementary Fig. S13."),
    _spec("slides.dh_distribution", "plot_dH_distribution_slides", "slides_v1", "slide_distribution", "fig-dh-distribution-slides.pdf", "slides"),
    _spec("paper.ha_circle", "plot_figure_HA_circle", "paper_v1", "single_square", "fig-ha-circle.pdf", "paper", layout="axis_off"),
    _spec("grant.auc", "plot_AUC_grant", "grant_v1", "grant_auc", "fig-auc-grant.pdf", "grant", layout="one_by_two"),
    _spec("grant.overview", "plot_overview_grant", "grant_v1", "grant_overview", "fig-overview-grant.pdf", "grant"),
    _spec("grant.prediction", "plot_prediction_grant", "grant_v1", "grant_overview", "fig-prediction-grant.pdf", "grant", notes="Registry records the final canvas; the legacy function first creates an unused 7.5-inch square."),
    _spec("slides.surveillance", "plot_surveillance_slides", "slides_v1", "slide_wide_short", "fig-surveillance-slides.pdf", "slides"),
    _spec("slides.overview_true", "plot_overview_schematic_true_slides", "slides_v1", "slide_overview_true", "fig-overview-true-slides.pdf", "slides"),
    _spec("slides.overview_sim", "plot_overview_schematic_sim_slides", "slides_v1", "slide_overview_sim", "fig-overview-sim-slides.pdf", "slides"),
    _spec("slides.overview_combined", "plot_overview_schematic_true_and_sim_slides", "slides_v1", "slide_overview_combined", "fig-overview-sim-prediction-slides.pdf", "slides"),
    _spec("slides.fix_loss_example", "plot_fix_loss_example_slides", "slides_v1", "slide_wide_short", "fig-fix-loss-example-slides.pdf", "slides"),
    _spec("slides.frequency", "plot_frequency_slides", "slides_v1", "slide_frequency", "fig-frequency-slides.pdf", "slides"),
    _spec("slides.selection", "plot_selection_slides", "slides_v1", "slide_selection", "fig-selection-slides.pdf", "slides"),
    _spec("slides.selection_subset", "plot_selection_subset_slides", "slides_v1", "slide_selection_subset", "fig-selection-subset-slides.pdf", "slides"),
    _spec("slides.emd_distribution", "plot_emd_distribution_slides", "slides_v1", "slide_distribution", "fig-emd-distribution-slides.pdf", "slides"),
    _spec("slides.example_frequency", "plot_example_frequency_slides", "slides_v1", "slide_example_frequency", "fig-example-frequency-slides.pdf", "slides"),
    _spec("slides.simulation", "plot_simulation_slides", "slides_v1", "slide_wide_short", "fig-simulation-slides.pdf", "slides"),
    _spec("slides.auc", "plot_AUC_slides", "slides_v1", "slide_auc", "fig-auc-slides.pdf", "slides"),
    _spec("slides.pfix", "plot_pfix_slides", "slides_v1", "slide_pfix", "fig-pfix-slides.pdf", "slides"),
    _spec("slides.emd", "plot_emd_slides", "slides_v1", "slide_emd", "fig-emd-slides.pdf", "slides"),
    _spec("slides.emd_joint", "plot_emd_joint_slides", "slides_v1", "slide_emd_joint", "fig-emd-joint-slides.pdf", "slides"),
)

FIGURE_REGISTRY = FigureRegistry(_FIGURE_SPECS)
