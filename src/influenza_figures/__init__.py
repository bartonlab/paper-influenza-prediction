"""Reproducible data preparation and plotting helpers for the influenza paper.

Legacy plotting scripts remain separate from this package.  The public exports
below are dependency-light configuration and orchestration primitives; focused
data-preparation modules live under :mod:`influenza_figures.prepare`.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from .background import (
    EVENT_KEY_FIELDS,
    MUTATION_KEY_FIELDS,
    BackgroundEventError,
    BackgroundEventExtraction,
    BackgroundMutationExtraction,
    extract_background_events,
    extract_background_mutations,
)
from .outputs import (
    OutputBusyError,
    OutputExistsError,
    OutputManager,
    OutputNotProducedError,
    OutputOwnershipError,
)
from .emd import (
    SUBTYPE_INITIAL_INFERENCE_DATES,
    EmdComparison,
    EmdComparisonError,
    EmdCutoff,
    compare_emd,
)
from .paths import PathOutsideRootError, ProjectPaths, ProjectRootNotFoundError
from .registry import (
    CANVASES,
    FIGURE_REGISTRY,
    Canvas,
    FigureRegistry,
    FigureSpec,
    RegistryCollisionError,
)
from .render import figure_style_context, legacy_style_context
from .styles import (
    STYLES,
    LegacyFigureConstants,
    LegacyMplotConstants,
    MarkerStyle,
    OutputStyle,
    Palette,
    StrokeStyle,
    StyleProfile,
    Typography,
    get_style,
    scoped_rc_context,
)

if TYPE_CHECKING:
    from .clades import (
        CladePreparationError,
        PreparedCladeFrequencies,
        filter_and_normalize_clade_frequencies,
        prepare_clade_frequencies,
        select_clades_by_total_frequency,
    )


_CLADE_EXPORTS = frozenset(
    {
        "CladePreparationError",
        "PreparedCladeFrequencies",
        "filter_and_normalize_clade_frequencies",
        "prepare_clade_frequencies",
        "select_clades_by_total_frequency",
    }
)


def __getattr__(name: str) -> object:
    """Load pandas-backed clade helpers only when they are requested."""

    if name not in _CLADE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = import_module(".clades", __name__)
    except ModuleNotFoundError as error:
        if error.name in {"numpy", "pandas"}:
            raise ModuleNotFoundError(
                "Canonical clade preparation requires the scientific Python "
                "dependencies; install the project environment before using "
                "the clade helpers.",
                name=error.name,
            ) from error
        raise
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _CLADE_EXPORTS)

__all__ = [
    "CANVASES",
    "EVENT_KEY_FIELDS",
    "MUTATION_KEY_FIELDS",
    "FIGURE_REGISTRY",
    "STYLES",
    "SUBTYPE_INITIAL_INFERENCE_DATES",
    "BackgroundEventError",
    "BackgroundEventExtraction",
    "BackgroundMutationExtraction",
    "Canvas",
    "CladePreparationError",
    "EmdComparison",
    "EmdComparisonError",
    "EmdCutoff",
    "FigureRegistry",
    "FigureSpec",
    "LegacyFigureConstants",
    "LegacyMplotConstants",
    "MarkerStyle",
    "OutputBusyError",
    "OutputExistsError",
    "OutputManager",
    "OutputNotProducedError",
    "OutputOwnershipError",
    "OutputStyle",
    "Palette",
    "PathOutsideRootError",
    "ProjectPaths",
    "ProjectRootNotFoundError",
    "PreparedCladeFrequencies",
    "RegistryCollisionError",
    "StrokeStyle",
    "StyleProfile",
    "Typography",
    "extract_background_events",
    "extract_background_mutations",
    "get_style",
    "figure_style_context",
    "legacy_style_context",
    "compare_emd",
    "filter_and_normalize_clade_frequencies",
    "prepare_clade_frequencies",
    "scoped_rc_context",
    "select_clades_by_total_frequency",
]
