# Data

No sequence data are tracked in this repository. The influenza sequences used in the analyses come from GISAID and may not be redistributed; place your own copies in the
folders below. Only this file, the empty-folder placeholders and the small synthetic files in `examples/` are tracked.

```
data/
├── raw/                 # inputs you provide (not tracked)
├── derived/
│   ├── study/           # FluStudy cache written by the EMD prediction (not tracked)
│   └── legacy/          # precomputed data matrix for the old fixation runs (optional, not tracked)
└── examples/            # tiny synthetic inputs (tracked)
```

## `raw/` — inputs

| File | Used by | Description |
|---|---|---|
| `<subtype>_<protein>.fasta` | all three tasks | Aligned amino-acid sequences: `H3N2_HA.fasta`, `H3N2_NA.fasta`, `H1N1_HA.fasta`, `H1N1_NA.fasta` |

The scripts look for `data/raw/<subtype>_<protein>.fasta` by default; any other file can be
passed with `--fasta PATH`.

**FASTA.** One record per strain. The header has seven `|`-separated fields

```
>strain|isolate_id|date|segment|hash|region|clade
>A/South_Australia/21/2015|EPI_ISL_196611|2015-03-13|HA|EPI646067|oceania|3C.2
```

* `strain` — strain name, used to remove duplicate records (the last record is kept);
* `date` — collection date `YYYY-MM-DD` (`YYYY-MM-XX` is read as the first of the month;
  other partial dates are dropped);
* `region` — Nextstrain region label (`north_america`, `europe`, `china`, `oceania`, ...);
* `clade` — clade label. Clade assignments are read from this field; a record without it
  gets `unassigned`. (`FluStudy.create_study` can still override them from an
  `isolate,clade` CSV through its `clade_csv` keyword, but the scripts do not use it.)

Sequences are aligned to the reference numbering, one residue per position: length 566
for HA and 469 for NA, for both H3N2 and H1N1. Records with a different length, more than
5% gaps or ambiguous residues, or no usable date are skipped by `FluStudy.create_study`.

## `derived/study/` — FluStudy cache (written by the EMD prediction)

`src/emd_prediction/run_emd_prediction.jl` builds the study from the FASTA on first use
(`load_or_create_study`) and reuses it afterwards. With `<tag> = <subtype>_<protein>_<region>`:

| File | Content |
|---|---|
| `data_<tag>.txt` | Data matrix (see below) |
| `frequency_<tag>.txt` | Smoothed allele-frequency trajectories, `q·L` rows × time points |
| `metadata_<tag>.csv` | One row per data-matrix row: `row,day,date,count,strain_names,n_strains,clade,clade_agreement,region,region_agreement` |
| `name_index_<tag>.csv` | One row per strain: `strain_name,data_row,day,date,clade,region` |
| `sequence_to_clade_<tag>.csv` | `day,seq,clade` for every distinct sequence |
| `study_manifest_<tag>.toml` | Settings that produced the cache; a mismatch with the current configuration stops the run |

Delete the folder (or pass `force_rebuild = true`) to rebuild after changing the FASTA or the
study settings.

## Data matrix format

All three tasks work on the same matrix, built by `FluStudy.create_study` (or read from a
`.txt` file): whitespace-delimited, one row per distinct sequence observed on a given day,

```
day  count  s_1  s_2  ...  s_L
```

* `day` — days since the study start date (2009-01-01 for H3N2, 2009-05-01 for H1N1);
  sequences dated exactly on the start date are excluded by `FluStudy`, so day 1 is the
  day after it;
* `count` — number of strains with that sequence on that day;
* `s_i` — amino-acid state at position `i`, encoded 1 = gap/ambiguous, 2..21 =
  `A C D E F G H I K L M N P Q R S T V W Y`.

Rows are sorted by day.

## `derived/legacy/` — optional

The fixation prediction can also read a precomputed data matrix in the format above
(`run_fixation_prediction.jl --source txt`), by default `data/derived/legacy/data_<subtype>_<protein>.txt`
or any file given with `--txt PATH`. The default input is the FASTA.

## `examples/` — synthetic inputs (tracked)

Made with `simulate_population(T = 30, n_sample = 10)` from `src/shared/synthetic_data.jl`,
which runs the negative-binomial population dynamics of `src/shared/population_simulation.jl`
with random selection coefficients. Small enough to track:

* `synthetic_data_matrix.txt` — data matrix, 20 sites, 30 days, 10 sampled sequences per day
  (states 2–5, i.e. A, C, D, E);
* `synthetic_H3N2_HA.fasta` — the same samples as FASTA records (20-residue sequences;
  read with `FluStudy.create_study(...; protein_length = 20)`).

The notebooks generate the full synthetic input on the fly with the same generator
(1000 days, 20 sequences per day).
