# Overview

This repository contains the code and data for reproducing the results accompanying the manuscript

### Sequence-structured fitness landscapes predict influenza evolution

Edwin Rodriguez-Horta<sup>1</sup> and John P. Barton<sup>1,†</sup>

<sup>1</sup> Department of Computational and Systems Biology, University of Pittsburgh School of Medicine, USA  
<sup>†</sup> Address correspondence to: [jpbarton@pitt.edu](mailto:jpbarton@pitt.edu)

The preprint is available at [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.10.27.762377).

We infer the selection acting on every amino-acid mutation in influenza A/H3N2 and A/H1N1 hemagglutinin (HA) and neuraminidase (NA) from sequence time series. We then use the inferred fitness landscape to forecast influenza evolution.

The repository has two parts:

1. **Analysis (Julia)**, in [`julia/FluGEM/`](julia/FluGEM/). This is
   the code that produces the results of the paper. It contains three analyses:
   * **MAP selection coefficients**: fitness of every amino-acid mutation over the whole study window;
   * **fixation-probability prediction**: the probability that each mutation present at a prediction date fixes or is lost, scored against its actual fate;
   * **simulation-based EMD prediction**: the predicted strain distribution months ahead, scored against the observed one by the earth mover's distance (EMD), alongside baseline predictors (null, single-locus model, multinomial logistic regression).

   Its [README](julia/FluGEM/README.md) describes the code structure, installation, input data, and how to run each analysis and the quick checks on synthetic data.
2. **Figures (Python)**, at the repository root: the `influenza_figures` package in
   `src/`, with `figures.ipynb` and `figures.py` as entry points. It draws the manuscript figures from the analysis results stored in `data/plotting/`.

The sequence data (from GISAID) are not included.

# Contents

```
paper-influenza-prediction/
├── julia/FluGEM/                # analysis code (Julia project); see its README
│   ├── Project.toml, Manifest.toml
│   ├── src/                     # shared/, map_estimation/, fixation_prediction/, emd_prediction/
│   ├── data/                    # expected inputs; GISAID sequences not tracked
│   └── notebooks/               # quick checks on synthetic data
├── src/influenza_figures/       # Python package: plotting-data preparation and figures
├── figures.ipynb                # one cell per manuscript figure
├── figures.py                   # plotting entry points called by the notebook
├── figure_manifest.yaml         # catalog of figures: inputs, style, outputs, status
├── data/plotting/               # plotting inputs (analysis results) and derived tables
├── figures/                     # rendered figures: paper/, supplement/, slides/, diagnostic/
├── scripts/                     # plotting-data preparation and validation
├── tests/                       # tests of the figure code
├── docs/                        # plotting layout and migration log
├── pyproject.toml               # Python package definition
├── environment-plotting.yml     # conda environment with the tested Python versions
└── LICENSE-GPL, LICENSE-CC0
```

## From analysis results to figures

The files in `data/plotting/inputs/` are the stored results of the analysis runs used in
the paper, normalized and with their provenance recorded (see
[data/plotting/README.md](data/plotting/README.md)). The figure code reads only these
files, not the Julia `outputs/` folder, so rerunning an analysis does not change a figure
until its results are brought into `data/plotting/inputs/`.

| Analysis | Plotting inputs | Figures |
|---|---|---|
| MAP selection coefficients | `inputs/selection/`, `inputs/frequencies/`, `inputs/structure/` | 2, S5, S6 |
| Fixation-probability prediction | `inputs/fixation/`, `derived/fixation_metrics/` | 3, S7, S16 |
| Simulation-based EMD prediction | `inputs/emd/` | 4, S8–S12, S17–S19 |
| Background fitness and clade frequencies | `inputs/background/`, `inputs/clades/` | 5, S13–S15 |

# Software dependencies

* **Analysis:** Julia 1.10 or later. The package versions tested are pinned in
  `julia/FluGEM/Manifest.toml`; see the
  [Julia README](julia/FluGEM/README.md#software-dependencies).
* **Figures:** Python 3.12 or later with numpy, scipy, pandas, scikit-learn, matplotlib,
  Pillow and seaborn (`pyproject.toml`), plus JupyterLab for the notebook.
  `environment-plotting.yml` is a conda environment with the tested versions. gemmi and
  freesasa are needed only to redo the structure preprocessing; the plotting inputs
  already contain its results.

# Installation and use

### Analysis

```bash
cd julia/FluGEM
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Then follow [Running the analyses](julia/FluGEM/README.md#running-the-analyses)
in the Julia README.

### Figures

```bash
conda env create -f environment-plotting.yml
conda activate influenza-figures
python -m pip install -e .       # optional: figures.py adds src/ to the path itself
jupyter lab figures.ipynb
```

Run the setup cell of `figures.ipynb` once, then the cell of the figure you want. Figures
are written below `figures/`; with the notebook's default `OVERWRITE = True`, an existing
PDF is replaced.

To run the tests of the figure code:

```bash
python -m unittest discover -s tests
```

The tests that check data provenance against the archived original files need a local
`archive/` folder, which is not part of the repository, and fail without it.

# License

This repository is dual licensed as [GPL-3.0](LICENSE-GPL) (source code) and
[CC0 1.0](LICENSE-CC0) (figures, documentation, and our presentation of the data).
