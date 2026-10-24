# FluGEM

**FluGEM** (Flu Generative Evolution Model) is the Julia code for the analyses in
*Sequence-structured fitness landscapes predict influenza evolution* (E. Rodriguez-Horta
and J. P. Barton). It is the analysis part of the
[paper-influenza-prediction](../../README.md) repository; the paper figures are made by the
Python code at the repository root.

# Overview

This project contains the code for inferring selection in seasonal influenza from
sequence time series and for predicting its evolution. Selection coefficients
are inferred as maximum a posteriori (MAP) estimates, and predictions come from forward
stochastic simulations seeded with the observed population. There are three analyses:

1. **MAP selection coefficients** (`src/map_estimation/`). MAP selection coefficients of
   every amino-acid mutation, inferred from the whole study window.
2. **Fixation-probability prediction** (`src/fixation_prediction/`). For a prediction
   date, estimate the probability that each mutation present in the population will fix
   or be lost, and compare with the fate it actually had (AUROC and fixation-probability
   summaries across many dates).
3. **Simulation-based EMD prediction** (`src/emd_prediction/`). Predict the sequences  distribution Δ days after a prediction date, and score the prediction against the observed future distribution by the earth mover's distance (EMD) and the consensus Hamming distance, alongside baseline predictors.

The data are aligned influenza A/H3N2 and A/H1N1 hemagglutinin (HA) and neuraminidase (NA)
amino-acid sequences from GISAID. They cannot be redistributed and are not included; see
[Input data](#input-data).

# Contents

```
julia/FluGEM/
├── Project.toml                 # Julia environment (dependencies + compat)
├── Manifest.toml                # exact package versions tested
├── README.md
├── src/
│   ├── shared/                  # used by more than one task
│   │   ├── FluStudy.jl      # FASTA -> data matrix (filtering, dedup, binning, imputation)
│   │   ├── mpl_inference.jl     #   loader for the inference kernel below
│   │   ├── mpl.jl               #   selection inference, MAPestimate
│   │   ├── gaussian_smoothing.jl, Population_dynamics.jl, misc.jl
│   │   ├── aa_encoding.jl       #   amino acid <-> integer state (q = 21)
│   │   ├── cli.jl               #   command-line options, output file names, study windows
│   │   ├── population_simulation.jl  # negative-binomial / Wright-Fisher dynamics 
│   │   └── synthetic_data.jl    #   synthetic inputs for notebooks and checks
│   ├── map_estimation/          # task 1
│   │   ├── map_selection.jl                # map_selection_table
│   │   └── run_map_estimation.jl           # entry point
│   ├── fixation_prediction/     # task 2
│   │   ├── FixationPrediction.jl        # module: run_simulation, scoring, summary curves
│   │   └── run_fixation_prediction.jl      # entry point (one prediction date)
│   └── emd_prediction/          # task 3
│       ├── mutation_simulation.jl   # causal prediction pipeline, baselines
│       ├── PredictionPipeline.jl    # config, study cache, SLM, export (save_pipeline_run)
│       ├── misc_EMD.jl, MLR.jl, sequence_distance.jl
│       └── run_emd_prediction.jl           # entry point (one prediction date)
├── data/                        # layout and formats in data/README.md; data not tracked
│   ├── raw/  derived/study/  derived/legacy/
│   └── examples/                # tiny synthetic inputs (tracked)
└── notebooks/                   # quick, reproducible checks on synthetic data
    ├── 01_map_estimation_quickcheck.ipynb
    ├── 02_fixation_prediction_quickcheck.ipynb
    └── 03_emd_prediction_quickcheck.ipynb
```

Run outputs go to `outputs/<task>/` in this folder, which is created when needed and not
tracked.

# Software dependencies

* **Julia 1.10 or later** (the lower bound set in `Project.toml`). Tested with 1.10.0.
* Julia packages, declared in `Project.toml` and pinned to the versions tested in
  `Manifest.toml`:

  | Package | Version | Used for |
  |---|---|---|
  | BioSequences | 3.1.6 | amino-acid sequences |
  | FastaIO | 1.0.2 | reading FASTA |
  | CSV, DataFrames | 0.10.14, 1.6.1 | tables |
  | Distributions | 0.25.109 | sampling in the Wright–Fisher and negative-binomial simulations |
  | StatsBase, DataStructures | 0.34.3, 0.18.20 | utilities |
  | JuMP, MathOptInterface, HiGHS | 1.23.3, 1.33.0, 1.11.0 | EMD as a linear program (task 3) |
  | Optim, LineSearches | 2.0.1, 7.6.0 | MLR baseline (task 3) |

  Standard libraries: Dates, DelimitedFiles, LinearAlgebra, Random, Statistics, TOML.
* **Jupyter with IJulia**, only for the notebooks.

Runtimes on a 10-core laptop with 16 GB of memory, for H3N2 HA (566 sites, 2009–2020):

* building the study from the FASTA takes about 30 s;
* each MAP fit takes about 10 min and about 4 GB of memory (tasks 1–3);
* task 2 takes about 45 s per simulated trajectory, so the production setting `M = 1000`
  is meant for a cluster, with one prediction date per array task;
* the notebooks take about 35 s, 2 min and 1 min on synthetic data, plus package compilation the first time.

# Installation

```bash
git clone https://github.com/bartonlab/paper-influenza-prediction.git
cd paper-influenza-prediction/julia/FluGEM
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`instantiate` installs the exact package versions recorded in `Manifest.toml`, all from
the General registry. To move to newer versions within the compat bounds of
`Project.toml`, run `julia --project=. -e 'using Pkg; Pkg.update()'`, which rewrites
`Manifest.toml`.

For the notebooks, install IJulia once in your default environment:

```bash
julia -e 'using Pkg; Pkg.add("IJulia")'
```

# Input data

The expected layout, file names, FASTA header fields and data-matrix columns are described
in [data/README.md](data/README.md). In short:

* put one FASTA per subtype and protein in `data/raw/`, named `<subtype>_<protein>.fasta`
  (`H3N2_HA.fasta`, `H3N2_NA.fasta`, `H1N1_HA.fasta`, `H1N1_NA.fasta`), or pass any file with
  `--fasta PATH`;
* FASTA headers are `strain|isolate_id|date|segment|hash|region|clade`, with dates
  `YYYY-MM-DD`; clade assignments are read from the `clade` field. Sequences are aligned,
  566 residues for HA and 469 for NA;
* every task turns the FASTA into the same **data matrix**, with rows
  `day  count  s_1 … s_L`: one row per distinct sequence and day, days counted from the
  study start (day 0), and states 1 = gap and 2..21 = `ACDEFGHIKLMNPQRSTVWY`.

## Supported subtypes and proteins

All three tasks run for **H3N2 HA, H3N2 NA, H1N1 HA and H1N1 NA**.

| Subtype | Study window (day 0 … end) | Notes |
|---|---|---|
| `H3N2` | 2009-01-01 … 2020-03-01 | default |
| `H1N1` | 2009-05-01 … 2020-03-01 | the scripts set day 0 automatically (`study_window` in `src/shared/cli.jl`); it must match the date hard-coded in `MAPestimate` and the EMD pipeline |

* **Protein** only sets the expected sequence length (HA 566, NA 469, the same for both
  subtypes) and, in task 1, whether the HA `Domain` column is added. NA tables have no
  domain column. The HA domain boundaries depend on the subtype, because the H3 signal
  peptide is one residue shorter than the H1 one:

  | Subtype | Signal peptide | HA1 | HA2 |
  |---|---|---|---|
  | `H3N2` | 1–16 | 17–345 | 346–566 |
  | `H1N1` | 1–17 | 18–344 | 345–566 |

  These are positions in the 566-column alignment (`ha_domain` in
  `src/map_estimation/map_selection.jl`), so check them against your own alignment.

# Running the analyses

Run all commands from this folder, `julia/FluGEM/`, where `--project=.`
activates its environment. The scripts find their default inputs and write their outputs
relative to this folder. A relative `--fasta` path is read from the working directory in
tasks 1 and 2, and from this folder in task 3 (the script changes to it first), so the two
agree when you run from here. Every script takes `--subtype` (`H3N2`, `H1N1`), `--protein`
(`HA`, `NA`) and `--fasta PATH`. The defaults are H3N2, HA and
`data/raw/<subtype>_<protein>.fasta`.

### 1. MAP selection coefficients

```bash
julia --project=. src/map_estimation/run_map_estimation.jl --subtype H3N2 --protein HA
julia --project=. src/map_estimation/run_map_estimation.jl --subtype H1N1 --protein NA
```

This fits the whole study window of the subtype and writes
`outputs/map_estimation/map_selection_<subtype>_<protein>.csv`,
with the columns `Position, Domain (HA only), Reference, Mutation, Selection_coefficient`.
It lists the alleles seen above 1% at two consecutive time points at least once. Settings
(`gamma`, smoothing, threshold) are in the `config` block at the top of the script.

### 2. Fixation-probability prediction

```bash
# one prediction date (the last day of data used for the fit)
julia --project=. src/fixation_prediction/run_fixation_prediction.jl --subtype H3N2 --protein HA --prediction-date 2010-01-03

# SLURM array: task k = 10-day grid of prediction dates (H3N2: k = 1..402)
julia --project=. src/fixation_prediction/run_fixation_prediction.jl --task-id $SLURM_ARRAY_TASK_ID --subtype H3N2 --protein HA
julia --project=. src/fixation_prediction/run_fixation_prediction.jl $SLURM_ARRAY_TASK_ID H3N2 HA     # positional form

# precomputed data matrix instead of the FASTA
julia --project=. src/fixation_prediction/run_fixation_prediction.jl --prediction-date 2010-01-03 --source txt --txt data/derived/legacy/data_H3N2_HA.txt
```

`--task-id` picks a date from a 10-day grid, set relative to day 0 of the subtype. The
ends of the grid are a compromise: it has to start late enough that the fit has enough
data to infer selection, and end early enough that every date still has future data to
establish each mutation's fate, while leaving enough dates to summarise. Prediction dates
outside the study window are rejected.

Each run writes
`outputs/fixation_prediction/<subtype>_<protein>_<source>/fixation_<subtype>_<protein>_<date>.csv`,
with one row per (site, amino acid). The columns are the observed frequency trajectory,
the MAP selection coefficient, the simulated P(fixation) and P(loss) with mean
times, and the true future fate. The settings (`GAMMA`, `TSMOOTH`, `NOISE`, `M`) are at the
top of the script. To summarise many dates, load the folder with `load_results` and call
`compute_summary_curves` (both in `FixationPrediction`). They return the AUROC and the
fixation probability of the top and bottom 50% of predictions per frequency bin.

### 3. Simulation-based EMD prediction

```bash
# explicit dates: fit from the initial date, predict from the prediction date, `horizon` days ahead
julia --project=. -t auto src/emd_prediction/run_emd_prediction.jl --subtype H3N2 --protein HA \
      --initial-date 2009-01-01 --prediction-date 2013-02-03 --horizon 270

# SLURM array: task k = monthly prediction dates 2012-01-01 .. 2019-01-01 (k = 1..85), horizon 270
julia --project=. -t auto src/emd_prediction/run_emd_prediction.jl --task-id $SLURM_ARRAY_TASK_ID --subtype H3N2 --protein HA
julia --project=. -t auto src/emd_prediction/run_emd_prediction.jl $SLURM_ARRAY_TASK_ID H3N2 HA        # positional form
```

* `--prediction-date` is the date the prediction is made: the last day of data the methods
  see. The predicted (target) date is `prediction date + horizon`.
* `--initial-date` is the first day of data used to fit the selection coefficients and the
  MLR model. It defaults to day 0 of the study window.
* `--horizon` is in days and defaults to 270.

The script checks that `day 0 ≤ initial date < prediction date` and that the target date
falls within the study window.

On first use the study is built from the FASTA and cached in `data/derived/study/`. Each
run writes `outputs/emd_prediction/predictions_<tag>.csv`, with one row per method
(GenModel, GenModel-Average, MLR, SLM, SLM-Average, Null, Lower Bound, Null-Emp,
Null-Emp-Denoised) and the columns `emd_pop`, `ham_cons`, dates and settings. Side-car
files with the same tag hold each method's predicted distribution, `d1`/`d2`/`dlb`, the
empirical bins and the selection-coefficient vectors. All other pipeline settings are in
the `PredictionConfig` block of the script.

## Output file names

All outputs follow one pattern, `<kind>_<subtype>_<protein>_<date(s)>_<key><value>...`. The
keys are the main parameters of the run, with values written compactly (`60.0` → `60`). It
is built by `run_name` in `src/shared/cli.jl`:

| Task | File name |
|---|---|
| 1 | `map_selection_<subtype>_<protein>.csv` |
| 2 | `fixation_<subtype>_<protein>_<prediction date>.csv` |
| 3 | `predictions_<tag>.csv` (and `d1_…`, `output_genmodel_…`, … with the same tag) |

# Notebooks and quick checks

Each notebook builds a synthetic population with known selection coefficients (no real
data needed), runs one task through the same functions as the scripts with small settings,
and checks basic properties with `@assert`. The synthetic data come from
`simulate_population` (`src/shared/synthetic_data.jl`), which uses the negative-binomial
dynamics of `src/shared/population_simulation.jl`; each notebook states the settings it
uses.

| Notebook | Checks |
|---|---|
| `01_map_estimation_quickcheck` | FASTA → data matrix round trip; MAP coefficients follow the true ones (correlation > 0.5, sign agreement > 0.6 for \|s\| > 0.05, reference = 0); the 1% / two-time-point filter keeps exactly the right alleles |
| `02_fixation_prediction_quickcheck` | P(fix), P(loss) in [0, 1] for every (site, state); prediction day; summary over several dates |
| `03_emd_prediction_quickcheck` | All methods present with finite EMD; Lower Bound ≤ Null; distributions normalised; `save_pipeline_run` output |

To run them:

```bash
julia --project=. -e 'using IJulia; notebook(dir = "notebooks")'
```

or start `jupyter lab` in this folder and pick the **Julia 1.10** kernel. The first cell
activates the project environment. Notebook 3 needs HiGHS and runs faster with several
threads. To add a multi-threaded kernel once:

```bash
julia -e 'using IJulia; IJulia.installkernel("Julia 1.10 (4 threads)", env = Dict("JULIA_NUM_THREADS" => "4"))'
```

To run a notebook without opening it:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_map_estimation_quickcheck.ipynb --output /tmp/01_check.ipynb
```

# What to track in Git

| Track | Do not track |
|---|---|
| `src/`, `notebooks/` (with outputs cleared), `README.md`, `data/README.md` | `data/raw/` (GISAID sequences, not redistributable) |
| `Project.toml`, `Manifest.toml` | `data/derived/` (study cache, large matrices) |
| `data/examples/` (tiny synthetic files) and the `.gitkeep` placeholders | `outputs/` (run results) |
| `.gitignore` | Notebook checkpoints, logs, `.DS_Store`, editor folders |

`.gitignore` already implements this table. Clear notebook outputs before committing
(*Edit → Clear All Outputs*, or `jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`).

# Notes

* **Day numbering.** Days are counted from the start of the study window (day 0).
  `FluStudy.create_study` drops sequences dated exactly on the start date and on the last
  date in the FASTA, because its daily bins are right-open. Its stop-date filter keeps one
  day past `stop_date`, and the `date` column of its metadata table is one day early. The
  fixation prediction's own date window removes the extra day.

# License

This repository is dual licensed as [GPL-3.0](../../LICENSE-GPL) (source code) and
[CC0 1.0](../../LICENSE-CC0) (figures, documentation, and our presentation of the data).
