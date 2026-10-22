# Generated figures

Figure outputs are grouped by their intended use:

- `paper/`
- `supplement/`
- `slides/`
- `poster/`
- `grant/`
- `diagnostic/`

These are generated artifacts. Their recipes, inputs, style profiles, and
canonical output paths are recorded in `../figure_manifest.yaml`.

The current manuscript set is:

- `paper/`: Main Figures 1–5;
- `supplement/`: S5–S19, including null EMD at 6/12 months (S8/S9), MLR EMD at
  6/9/12 months (S10–S12), and SLM EMD at 6/9/12 months (S17–S19); and
- `diagnostic/`: comparisons such as best GenModel versus its same-run baseline,
  which are intentionally outside the manuscript numbering.

Main Figure 1 is exported from the editable one-slide source at
`../data/plotting/inputs/artwork/fig-model-overview.pptx`. Every other figure in
the current Main 1–5 / S5–S19 set has a direct canonical call in
`../figures.ipynb` and `../figures.py`.

EMD pages omit nonsignificant p-value text. Significant values between 0.01 and
0.05 are printed to two decimal places; smaller values use order notation. An
asterisk plus the key `* full model worse` marks significant comparisons whose
mean comparator-minus-full-model EMD is negative.

The pre-reorganization flat outputs are preserved under
`../archive/pre_plot_reorg_2026-08-26/original/figures/` for visual comparison
and recovery.
