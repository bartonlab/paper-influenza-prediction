# Canonical plotting data

This directory is the maintained data interface for figure generation. It is
built from the read-only snapshots at
`archive/pre_plot_reorg_2026-08-26/original/`,
`archive/emd_cv_2026-09-24/original/`, and
`archive/emd_cv_2026-09-30/original/`, plus the first complete 6- and 12-month
release at `archive/emd_cv_2026-10-10/original/` and the authoritative complete
release at `archive/emd_cv_2026-10-13/original/`. The September 24 EMD snapshot
is retained as superseded provenance. The September 30 and October 10 snapshots
are validation-only predecessors; all current 6-, 9-, and 12-month values come
from the October 13 snapshot.

## Layout

- `inputs/` contains plotting inputs organized by subject: background fitness,
  clades, EMD comparisons, fixation predictions, allele frequencies, selection,
  structures, surveillance, and bundled examples.
- `derived/` contains deterministic, plot-ready tables for fixation metrics,
  overview trajectories, and structure/selection comparisons.
- `migration_map.csv` maps each non-EMD source to its canonical destination and
  records the transformation, row counts, and source/destination SHA-256 hashes.
- `schema_summary.json` indexes the non-EMD schema families and their canonical
  paths.
- `inputs/emd/sources.csv` is the equivalent source inventory for EMD data.
  `inputs/emd/values.csv` additionally records the exact source file, row, and
  column behind every canonical EMD value.

The current `emd_v6` materialization contains 2,591 evaluations, 16,578 values,
10 methods, and 76 source records. Its final hashes and byte sizes are recorded
in the migration log and figure manifest. An independent clean rebuild matched
all four canonical files byte-for-byte.

The two `inputs/examples/fix_loss_example_*.zip` files are deterministic bundles
of the many small files used by those example plots. The `.txt` and `.dat`
matrices retained under `inputs/frequencies/`, `inputs/selection/`, and
`inputs/examples/` remain headerless or legacy-shaped where changing their layout
would break existing plotting code.

## Naming and columns

Canonical filenames and CSV headers use lower-case `snake_case`. Shared identity
values use lower-case controlled names (`h1n1`, `h3n2`, `ha`, and `na`), dates use
ISO `YYYY-MM-DD`, identifiers end in `_id`, and time coordinates name their unit
or origin where needed (for example, `horizon_months`, `horizon_days`, and
`observation_day_2009`). `source_variant` separates datasets that share a schema
but are not interchangeable. Mutation tables use `mutation_count` in place of the
legacy `MutationFrequency` and add explicit `subtype` and `segment` columns.

The CSV header is the authoritative column order for each materialized table.
`schema_summary.json` provides the stable schema-family identifier; provenance
tables should be consulted before merging similarly named sources.

## Background-fitness data model

`inputs/background/full.csv` and `inputs/background/population_fitness.csv`
use `background_fitness_v2`. They remain separate because their event
memberships differ, but the shared scientific quantities now have the same
names:

- `initial_background_fitness`, `final_background_fitness`, and
  `background_fitness_change` describe the genetic background;
- `initial_population_fitness` is the population-average fitness at emergence;
- `selection_coefficient` is the focal mutation's inferred coefficient; and
- `duration_days` is `final_time - initial_time`.

For the background figures, the unit of analysis is one biological mutation,
keyed by `(subtype, segment, position, state)`. The
`influenza_figures.background.extract_background_mutations` helper first
applies the figure's inclusive duration cutoff to the observation rows and
then selects the record with the earliest `start_time` among the eligible rows
for each identity. Source `initial_time`, `final_time`, and `outcome` values
remain authoritative; the plotting layer does not reclassify outcomes from one
sampled frequency. The population-fitness table contains 418 mutation
identities without a duration cutoff and 315 at the paper's `>= 90`-day cutoff
(54 fixed and 261 lost).

The archived simulation code uses 0.05 and 0.95 as loss/fixation thresholds and
uses two consecutive observations at or beyond a terminal threshold in its
classification loop. Because the canonical CSVs already contain the upstream
event times and labels, threshold logic is not rerun during plotting.

## EMD data model

The EMD release is normalized into four related tables:

- `evaluations.csv`: one row and stable `evaluation_id` per forecast evaluation.
- `values.csv`: one EMD value per `evaluation_id`, `method_id`, and
  `result_version`, with cell-level provenance and the selected hyperparameter
  label where cross-validation applies.
- `methods.csv`: the authoritative model names and filtering/linkage semantics.
- `sources.csv`: all 76 archived source CSVs used by the canonical release,
  including the four superseded September 24 summaries, the 12 authoritative
  October 13 summaries, and the legacy validation and historical files. The
  September 30 and October 10 predecessors are checked directly but do not
  supply source records or current values.

There are two deliberately separate evaluation protocols:

- `full_grid` contains 1,427 evaluations across 6-, 9-, and 12-month horizons.
  It preserves the detailed FullModel families; SLM and the legacy diagnostic
  methods are present only where their source rows match this grid exactly.
- `results_grid` contains 1,164 evaluations: 388 at each of the 6-, 9-, and
  12-month horizons from the complete cross-validation releases. It retains
  explicit region and model start date.

Compare methods by exact `evaluation_id` within one protocol. Do not approximate
date matches or silently combine `full_grid` and `results_grid`; the
`influenza_figures.emd.compare_emd` helper enforces this by default and reports
unmatched evaluations.

The canonical `method_id` values are:

| `method_id` | Meaning |
| --- | --- |
| `lower_bound` | EMD lower bound |
| `null` | neutral/null comparator |
| `full_model_raw` | full model with rare mutations retained |
| `full_model_unlinked` | unlinked full-model diagnostic |
| `full_model_rare_filtered` | baseline/pre-cross-validation full model with rare mutations removed |
| `full_model_cross_validated_rare_filtered` | full model with rare mutations removed and hyperparameters selected by cross-validation |
| `single_locus_model_rare_filtered` | single-locus/single-site model (SLM), rare mutations removed |
| `multinomial_logistic_regression_rare_filtered` | multinomial logistic regression (MLR), rare mutations removed |
| `full_model_single_site_simulation` | FMSS: full model / single-site simulation; legacy diagnostic |
| `single_locus_full_sites_simulation` | SLFS: single-locus model / full-sites simulation; legacy diagnostic |

For the pre-cross-validation `Results_EMD_*.csv` files, `GenModel` maps to
`full_model_rare_filtered`; `GenModel`, `SLM`, and `MLR` all have rare mutations
filtered. In every 6-, 9-, and 12-month cross-validation release,
`GenModel (baseline)` maps to that same baseline parameterization, but it is a
fresh stochastic rerun rather than a byte-identical copy of the historical
`GenModel`. `best GenModel` maps to
`full_model_cross_validated_rare_filtered`. The exact `CV parameter` string is
stored as `selected_hyperparameters` on the corresponding best-model value.

The September 30 correction has the same schema, evaluation rows, dates, lower
bound, SLM, MLR, baseline, best model, and selected hyperparameters as the
September 24 release. Only `Null` changed, in all 388 rows. The October 13
release is now authoritative for every horizon. Relative to the September 30
and October 10 predecessors, only `best GenModel` and its paired `CV parameter`
change: all 97 H3N2/HA 12-month rows, 13 H1N1/NA 6-month rows, and 10 H1N1/NA
12-month rows. The other nine input files are byte-identical. All six canonical
values on a row come from its one October 13 source file, so results from
separate reruns are never combined. The September 30 and October 10 files are
validation-only; the September 24 files remain provenance-only. The older
6-month Results family and rounded 12-month Results projections likewise supply
no current `results_grid` values. FMSS and SLFS remain historical diagnostics.

The short-lived `emd_v1` identifier
`multilinear_regression_rare_filtered` expanded MLR incorrectly. Canonical
tables use the corrected identifier above as of `emd_v2`; release `emd_v3`
adds the explicit cross-validated full-model method, and `emd_v4` replaces the
incorrect 9-month null EMDs with the September 30 correction. Release `emd_v5`
adds definitive 6- and 12-month cross-validation results and makes the selected
full model the target at every maintained horizon. Release `emd_v6`, produced
by `emd_migration_v6`, adopts the October 13 correction to the lower-bound
horizon association used during hyperparameter selection. The comparison API
accepts the old spelling temporarily as a deprecated input alias.

### EMD warmup and figure recipes

EMD values are compared only on or after an inclusive calendar cutoff 1,095
days after the subtype's initial inference date:

| Subtype | Initial inference date | First comparison date |
| --- | --- | --- |
| H3N2 | `2009-01-02` | `2012-01-02` |
| H1N1 | `2009-05-02` | `2012-05-01` |

The dates come from the explicit `model_start_date` values in the Results
family. `full_grid` rows do not contain that field, so the same subtype dates
are recorded repository defaults; callers can override them when deliberately
reproducing a different legacy convention. Filtering always uses the ISO
`observation_date`, which is safe when a source file was already hand-trimmed.
Every comparison returns a printable report containing the relative day,
initial date, exact cutoff date, inclusivity, and date source.

Canonical paper recipes use `results_grid` and the cross-validation-selected
full model at all three horizons. Null, SLM, and MLR are separate explicit
comparator IDs; display labels never select a method. The old mislabeled
single-site/MLR test is not a maintained result. Registered outputs bind their
comparator and horizon, so a result cannot be saved under another recipe's name.
Paper rendering also refuses unmatched evaluations and prints the relative and
concrete subtype cutoffs by default. The older multi-output slide functions
remain candidates until they receive the same canonical adapter.

The `emd_v6` correction changes the selected full-model result only in the
H3N2/HA 12-month, H1N1/NA 6-month, and H1N1/NA 12-month panels. Consequently,
the maintained 6- and 12-month MLR, null, SLM, and best-versus-baseline figures
must be regenerated. The four 9-month figures are numerically unchanged because
all four October 13 9-month inputs are byte-identical to their September 30
predecessors.

The cross-validation diagnostic calls
`figures.plot_emd_best_vs_baseline(horizon_months)` and writes
`figures/diagnostic/fig-emd-best-vs-baseline-{horizon}m.pdf` for 6, 9, or
12 months. The 9-month compatibility wrapper remains available. Each plot compares
`full_model_cross_validated_rare_filtered` with
`full_model_rare_filtered` from the same definitive rerun. Its displayed
difference is `baseline EMD - best-model EMD`: positive values favor the
cross-validation-selected best model, while negative values favor the
baseline. This diagnostic does not replace or alter any paper recipe.

The October 13 release has no lower-bound inversions: `Lower Bound` is less
than or equal to every reported model EMD in all 1,164 cross-validation rows.
The producer enforces this invariant for the authoritative release.

## Clade-frequency preparation

The maintained clade helper chooses clades from the specified observed
reference window using the legacy strict rule `sum(frequency) > 1.0`, filters
all series to those clades, and only then normalizes within each
`(subtype, segment, series, date)` group. This fixes the old loop-local filter
that left excluded clades in the plotting data. On the current H3N2/HA table
and 2011-08-01 to 2015-06-01 open reference interval, the selected clades are
`3C`, `3C.2`, `3C.2a`, `3C.3`, `3C.3a`, and `3C.3b`. The selection rule is an
unweighted sum over sampled dates, so its reference window and sampling grid
must remain explicit.

## Fixation and AUC variants

`inputs/fixation/predictions.csv` and
`derived/fixation_metrics/auc.csv` preserve two distinct SLM variants:

- `source_variant=paper_date_filtered` is the date-filtered SLM family used for
  the paper figures. Prediction rows carry the explicit `date` value.
- `source_variant=extended_all_time` is the broader later SLM family. It is kept
  because it differs from, and contains information beyond, the paper variant.

Use `model_id=full_model, source_variant=all_time` for the corresponding full
model. Always select both `model_id` and `source_variant`; the SLM variants should
not be pooled. The separate `*_2015.csv` metric tables are retained for the 2015
presentation figures.

## Regeneration and checks

The builders refuse to replace an existing destination by default. From the
repository root, a non-destructive reconstruction in a fresh temporary directory
is:

```bash
python3 scripts/migrate_plotting_data.py \
  --source-root archive/pre_plot_reorg_2026-08-26/original \
  --output-root /tmp/influenza-plotting-rebuild

python3 scripts/prepare_emd_data.py \
  --repo-root archive/pre_plot_reorg_2026-08-26/original \
  --cv-9m-root archive/emd_cv_2026-09-24/original/tmp \
  --corrected-cv-9m-root archive/emd_cv_2026-10-13/original/tmp \
  --corrected-cv-6-12-root archive/emd_cv_2026-10-13/original/tmp \
  --previous-corrected-cv-9m-root archive/emd_cv_2026-09-30/original/tmp \
  --previous-corrected-cv-6-12-root archive/emd_cv_2026-10-10/original/tmp \
  --output-dir /tmp/influenza-plotting-rebuild/inputs/emd
```

The migration checks compare the rebuilt output and its recorded source hashes
against the archive:

```bash
python3 -m unittest tests.test_plotting_data_migration tests.test_emd_migration
```

Archive-wide file hashes for the original reorganization snapshot are recorded in
`archive/pre_plot_reorg_2026-08-26/SHA256SUMS`; its `file_manifest.csv`,
`path_mapping.csv`, and `schema_inventory.csv` provide the broader snapshot
inventory. The September 24 and October 13 cross-validation source files are
additionally pinned by SHA-256 in `inputs/emd/sources.csv`; the validation-only
September 30 and October 10 snapshots carry their own `SHA256SUMS` files.
