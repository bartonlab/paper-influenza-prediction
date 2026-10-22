#!/usr/bin/env python3
"""Render three representative PDFs from archived inputs for migration parity."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "archive" / "pre_plot_reorg_2026-08-26" / "original"
sys.path.insert(0, str(REPO / "src"))

from influenza_figures.legacy import figures as legacy  # noqa: E402
from influenza_figures.render import legacy_style_context  # noqa: E402


COLORS = ["#C35999", "#DDA0C3", "#81AB5B", "#ABCE7D"]


def render(output_dir: Path) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy.FIG_DIR = str(output_dir)

    with legacy_style_context("paper_v1"):
        legacy.plot_frequency_selection(
            freq_files=[
                str(ARCHIVE / "data/allele_frequency_H3N2_HA_2009_to_2020_aa_ts_60_filt.txt"),
                str(ARCHIVE / "data/allele_frequency_H3N2_NA_2009_to_2020_aa_ts_60_filt.txt"),
                str(ARCHIVE / "data/allele_frequency_H1N1_HA_2009_to_2020_aa_ts_60_filt.txt"),
                str(ARCHIVE / "data/allele_frequency_H1N1_NA_2009_to_2020_aa_ts_60_filt.txt"),
            ],
            s_files=[
                str(ARCHIVE / "data/selection_coeff_H3N2_HA_since_01_2009_to_03_2020_filt_observed_add_counts.csv"),
                str(ARCHIVE / "data/selection_coeff_H3N2_NA_since_01_2009_to_03_2020_filt_observed_add_counts.csv"),
                str(ARCHIVE / "data/selection_coeff_H1N1_HA_since_01_2009_to_03_2020_filt_observed_add_counts.csv"),
                str(ARCHIVE / "data/selection_coeff_H1N1_NA_since_01_2009_to_03_2020_filt_observed_add_counts.csv"),
            ],
            deltats=[0, 0, 120, 120],
            colors=COLORS,
            min_counts=20,
            fig_name="fig-frequency-selection.pdf",
        )

        legacy.plot_emd(
            emd_files=[
                str(ARCHIVE / "data/emd/EMD_9m_FullModel_H3N2_HA_full_plus_MLR.csv"),
                str(ARCHIVE / "data/emd/EMD_9m_FullModel_H3N2_NA_full_plus_MLR.csv"),
                str(ARCHIVE / "data/emd/EMD_9m_FullModel_H1N1_HA_full_plus_MLR.csv"),
                str(ARCHIVE / "data/emd/EMD_9m_FullModel_H1N1_NA_full_plus_MLR.csv"),
            ],
            t_min=365 * 3,
            extend_limits=False,
            deltats=[0, 0, 120, 120],
            null_col="Null",
            sim_col="FM_remove_rare_mut",
            colors=COLORS,
            label_reassortment=False,
            fig_name="fig-emd-vs-null-9m.pdf",
        )

    with legacy_style_context("slides_v1"):
        legacy.plot_simulation_slides(seed=6)

    expected = (
        output_dir / "fig-frequency-selection.pdf",
        output_dir / "fig-emd-vs-null-9m.pdf",
        output_dir / "fig-simulation-slides.pdf",
    )
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"Expected parity PDFs were not created: {missing}")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in render(args.output_dir.resolve()):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
