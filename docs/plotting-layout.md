# Plotting code and data layout

**Status:** Archive-backed reorganization implemented; final Main Figure 1–5 and Supplementary Figure S5–S19 mapping is explicit, with runnable canonical recipes for every data-driven figure
**Last updated:** 2026-10-15

This document explains the intended organization of the plotting workflow. The
goals are to make figure generation reproducible, keep data provenance visible,
support paper/slide/poster rendering without hidden global state, and preserve
every legacy input in an auditable archive.

See [Plotting migration log](plotting-migration-log.md) for the old-to-new file
map, validation evidence, unresolved decisions, and chronological change log.

## Design principles

- The notebook is a readable driver, not the source of hidden data preparation.
- A figure manifest is the authoritative catalog of active figures.
- Immutable upstream analysis results and locally regenerated intermediates are
  stored separately.
- Plotting functions do not write or overwrite data.
- Every figure explicitly declares its inputs, transformations, canvas, style,
  output, and verification state.
- Scientific methods are identified by controlled names plus provenance, not by
  ambiguous filename suffixes.
- Time cutoffs are applied to normalized calendar dates relative to an explicit
  inference origin, not to source-specific integer offsets.
- For background figures with a duration cutoff, eligible observation rows are
  selected first; repeated observations are then reduced to the earliest
  eligible `start_time` per explicit mutation identity.
- Existing appearance is validated against reference PDFs before style code is
  simplified.
- Archive contents preserve original relative paths and checksums.

## Implemented repository map

```text
archive/
  pre_plot_reorg_2026-08-26/
    original/
    file_manifest.csv
    path_mapping.csv
    SHA256SUMS
  emd_cv_2026-09-24/original/
  emd_cv_2026-09-30/original/
  emd_cv_2026-10-10/original/
  emd_cv_2026-10-13/original/

data/
  plotting/
    inputs/
      frequencies/
      selection/
      emd/
      fixation/
      background/
      clades/
      structure/
      examples/
      surveillance/
    derived/
      fixation_metrics/
      overview/
      structure/

src/
  influenza_figures/
    __init__.py
    background.py
    background_plot.py
    background_plot_data.py
    clades.py
    emd.py
    emd_plot.py
    fixation_plot.py
    manuscript_stats.py
    outputs.py
    paths.py
    selection_plot.py
    styles.py
    registry.py
    render.py
    prepare/
      emd.py
    legacy/
      figures.py
      mplot.py

scripts/
  create_figures_notebook.py
  create_plotting_archive.py
  migrate_plotting_data.py
  migrate_legacy_plotting_code.py
  prepare_emd_data.py
  render_parity_samples.py
  retire_legacy_plotting_tree.py
  validate_figure_manifest.py

tests/
  test_plotting_data_migration.py
  test_background_events.py
  test_background_plot.py
  test_background_plot_data.py
  test_clades.py
  test_emd_migration.py
  test_emd_comparisons.py
  test_emd_plot.py
  test_figure_registry.py
  test_figure_manifest.py
  test_figures_notebook.py
  test_fixation_plot.py
  test_legacy_compatibility.py
  test_plot_styles.py
  test_plot_paths_outputs.py
  test_retire_legacy_plotting_tree.py
  test_selection_plot.py

figure_manifest.yaml
figures.py
figures.ipynb
figures/
  paper/
  supplement/
  slides/
  poster/
  grant/
  diagnostic/
```

## Final manuscript figure map

The final figure PDFs supplied for this migration resolve the earlier ambiguity
between exploratory notebook plots and manuscript figures. The table below is
the canonical Main Figure 1–5 and Supplementary Figure S5–S19 map. Each Python
entry point has a directly runnable cell in `figures.ipynb`; default arguments
select the listed canonical data and output. Figure 1 is intentionally different
because it is composed artwork rather than the result of a data analysis.

| Manuscript label | Scientist-facing action | Canonical plotting source | Canonical output |
|---|---|---|---|
| Figure 1 | Manually export the single slide from `data/plotting/inputs/artwork/fig-model-overview.pptx` | Editable PowerPoint artwork | `figures/paper/fig-model-overview.pdf` |
| Figure 2 | `figures.plot_selection_structure_space()` | selection, SASA, and rendered-structure inputs | `figures/paper/fig-selection-structure-space.pdf` |
| Figure 3 | `figures.plot_pfix_dauc()` | canonical fixation-probability and full-model AUROC tables | `figures/paper/fig-pfix-dauc.pdf` |
| Figure 4 | `figures.plot_emd_comparison("null", 9)` | `emd_v6` | `figures/paper/fig-emd-vs-null-9m.pdf` |
| Figure 5 | `figures.plot_background_clade()` | canonical population-fitness mutation records and clade frequencies | `figures/paper/fig-dw-clade.pdf` |
| Figure S5 | `figures.plot_frequency_selection()` | four-protein frequencies and selection with mutation counts | `figures/supplement/fig-frequency-selection.pdf` |
| Figure S6 | `figures.plot_selection_sasa()` | canonical SASA/selection derivative | `figures/supplement/fig-selection-sasa.pdf` |
| Figure S7 | `figures.plot_fixation_auc()` | full-model AUROC, all-time variant, day 1,095 cutoff | `figures/supplement/fig-fixation-auc.pdf` |
| Figure S8 | `figures.plot_emd_comparison("null", 6)` | `emd_v6` | `figures/supplement/fig-emd-vs-null-6m.pdf` |
| Figure S9 | `figures.plot_emd_comparison("null", 12)` | `emd_v6` | `figures/supplement/fig-emd-vs-null-12m.pdf` |
| Figure S10 | `figures.plot_emd_comparison("MLR", 6)` | `emd_v6` | `figures/supplement/fig-emd-vs-mlr-6m.pdf` |
| Figure S11 | `figures.plot_emd_comparison("MLR", 9)` | `emd_v6` | `figures/supplement/fig-emd-vs-mlr-9m.pdf` |
| Figure S12 | `figures.plot_emd_comparison("MLR", 12)` | `emd_v6` | `figures/supplement/fig-emd-vs-mlr-12m.pdf` |
| Figure S13 | `figures.plot_background_selection()` | canonical population-fitness mutation records, no duration cutoff | `figures/supplement/fig-ds.pdf` |
| Figure S14 | `figures.plot_background_initial_advantage()` | canonical population-fitness mutation records, at least 90 days | `figures/supplement/fig-dw-bg.pdf` |
| Figure S15 | `figures.plot_background_growth()` | canonical population-fitness mutation records, at least 90 days | `figures/supplement/fig-dw-rate.pdf` |
| Figure S16 | `figures.plot_single_site_auc()` | paper date-filtered SLM AUROC, day 1,095 cutoff | `figures/supplement/fig-auc-ss.pdf` |
| Figure S17 | `figures.plot_emd_comparison("SLM", 6)` | `emd_v6` | `figures/supplement/fig-emd-vs-slm-6m.pdf` |
| Figure S18 | `figures.plot_emd_comparison("SLM", 9)` | `emd_v6` | `figures/supplement/fig-emd-vs-slm-9m.pdf` |
| Figure S19 | `figures.plot_emd_comparison("SLM", 12)` | `emd_v6` | `figures/supplement/fig-emd-vs-slm-12m.pdf` |

This map supersedes the earlier provisional classification of the 6- and
9-month SLM comparisons as diagnostics and the unrelated provisional S18/S19
placeholders. Best-GenModel-versus-baseline comparisons remain diagnostics and
write only below `figures/diagnostic/`.

## Data zones

### `data/plotting/inputs/`

These are immutable results supplied by upstream analyses or external sources.
They may already be processed scientifically; “inputs” means inputs to this
repository's plotting workflow, not necessarily raw experimental data.

Rules:

- Never modify them during a figure build.
- Record source path, archive path, hash, schema, producing analysis, and license
  or access restrictions.
- Normalize them by creating a new canonical file, not by rewriting the archived
  source.

### `data/plotting/derived/`

These are deterministic products that can be regenerated from plotting inputs,
including AUC summaries, fixation-probability summaries, overview simulations,
SASA derivatives, and visualization-specific caches.

Every derived file has exactly one registered producer. Figures are consumers,
not producers.

### `figures/`

This directory contains rendered artifacts only. Intermediary CSVs, PDBs, and
attribute files do not belong here. Purpose-specific subdirectories prevent
paper, grant, slide, and poster variants from overwriting one another.

### `archive/`

The archive is a byte-preserving historical record, not another active data
source. It includes hashes and a path map. Private GISAID material remains
ignored and must not become publicly tracked merely because its path changes.
After canonical materialization, 4,935 verified legacy files were retired from
their old working paths; every one remains recoverable below the dated
`original/` tree.

## Naming and schema conventions

- Use lowercase `snake_case` for Python modules, directories, filenames, and
  columns.
- Use `h1n1` and `h3n2` for subtype, and lowercase `ha` and `na` for segment.
- Do not mix `H1`, `H3`, `N1`, `N2`, and full subtype/segment identities in
  canonical paths.
- Store dates as ISO `YYYY-MM-DD` values.
- Put units in names: `observation_time_days`, `horizon_days`,
  `forecast_horizon_months`.
- Replace generic prefixes such as `df_` and suffixes such as `full`, `filt`,
  `dates`, and `plus` with explicit columns or method identifiers.
- Use nullable booleans for scientific annotations; missing/not-applicable is
  distinct from `false`.
- Preserve identifiers as text even when they look numeric.
- Record the original column name and source row for any normalized scientific
  value.
- In loaders, explicitly preserve lowercase segment value `na` and do not let it
  be interpreted as a missing-value token.

## EMD organization

### Why normalized storage is required

The project compares the full model with several alternatives:

- the neutral or null model;
- the single-site/single-locus model, usually called SLM;
- multinomial logistic regression (MLR);
- and, for diagnostics, other pairs of modeling approaches.

A long method/value table supports all these comparisons without privileging one
comparator or creating another sparse file every time a method is added. Plotting
loaders may pivot the selected methods into a temporary wide DataFrame.

### Canonical files

```text
data/plotting/inputs/emd/
  evaluations.csv
  values.csv
  methods.csv
  sources.csv
```

`evaluations.csv` contains one row per prediction evaluation:

```text
evaluation_id
evaluation_protocol
protocol_version
horizon_months
horizon_days
subtype
segment
region
model_start_date
observation_date
prediction_date
observation_day_2009
prediction_day_2009
```

`values.csv` contains one row per method/evaluation result:

```text
evaluation_id
method_id
emd
result_version
source_id
source_row
source_column
transform_version
selected_hyperparameters
value_note
```

The unique key is `(evaluation_id, method_id, result_version)`. Only one result
version is active for a given canonical release.

`methods.csv` defines method semantics:

```text
method_id
model_family
display_label
linkage_policy
rare_mutation_policy
definition_status
notes
```

`sources.csv` records full provenance:

```text
source_id
source_family
original_path
archived_path
sha256
bytes
rows
schema_id
upstream_version
canonical_release
transform_description
```

### Controlled EMD method identifiers

| Method ID | Source interpretation |
|---|---|
| `lower_bound` | `LB`, `LB_Dist`, or `Lower Bound`. |
| `null` | `Null` or `Null_Dist`. |
| `full_model_raw` | `FM` in current `*_full.csv`; legacy `Medoid_Dist`. |
| `full_model_unlinked` | `FM_unlinked`. |
| `full_model_rare_filtered` | `FM_remove_rare_mut`; `FM` in current AllModels; `GenModel` in pre-CV Results; and `GenModel (baseline)` in the 6-/9-/12-month cross-validation releases. |
| `full_model_cross_validated_rare_filtered` | `best GenModel` in the 6-/9-/12-month cross-validation releases; its exact `CV parameter` is retained in `values.selected_hyperparameters`. |
| `single_locus_model_rare_filtered` | `SLM` in AllModels and Results. |
| `multinomial_logistic_regression_rare_filtered` | Multinomial logistic regression (MLR): `MLR_remove_rare_mut` or `MLR` in Results. |
| `full_model_single_site_simulation` | `FMSS`: legacy diagnostic using the full model with a single-site simulation. |
| `single_locus_full_sites_simulation` | `SLFS`: legacy diagnostic using the single-locus model with a full-sites simulation. |

The original label and source family are always retained. In particular, `FM`
must not be globally renamed: it means the raw full model in one family and the
rare-filtered full model in another.

The scientific mapping is authoritative: both GenModel columns are full-model
results with rare mutations filtered. `GenModel (baseline)` uses the pre-CV
parameterization, but is a new stochastic rerun rather than a byte-identical
copy of the historical `GenModel`; `best GenModel` uses parameters selected by
cross-validation. SLM and MLR are also rare-filtered. The 12 authoritative
October 13 files each own all six EMD columns on their rows; they are never
combined with comparator values from another rerun. The September 24 files
remain available as superseded provenance and supply no current values. The
September 30 and October 10 snapshots are immutable validation-only
predecessors. The pre-CV 6-month Results files and rounded 12-month Results
projections remain provenance/validation sources and likewise supply no current
`results_grid` values. FMSS and SLFS remain diagnostics, not primary forecast
comparators.

The corrected MLR vocabulary is part of `emd_normalized_v2`; the explicit
cross-validated/baseline split is `emd_normalized_v3`; and the corrected
9-month null values define `emd_normalized_v4`. Definitive 6- and 12-month
cross-validation rows define `emd_normalized_v5`. The October 13 correction is
canonical release `emd_v6` under transform `emd_migration_v6`: it fixes the
lower-bound horizon association used during hyperparameter selection without
changing the reported `Lower Bound` column. The short-lived
`emd_normalized_v1` spelling
`multilinear_regression_rare_filtered` is accepted only as a deprecated input
alias by `compare_emd`; it is not written to canonical data or figure recipes.

### Evaluation protocols

Only values sharing an exact `evaluation_id`, evaluation protocol, and result
version may be compared.

| Protocol | Meaning |
|---|---|
| `full_grid`, version `1` | Strain-specific FullModel evaluations and matching after-2012 metrics. |
| `results_grid`, version `1` | 6-/9-/12-month Results evaluations with explicit dates and supplied nominal 180-/270-/360-day horizons; every horizon uses a complete definitive cross-validation release. |

The separate root H3N2 distance-to-future table is not consumed by an
active/candidate recipe and remains archive-only; `external_h3n2` is therefore
not a protocol in the maintained canonical release.

The Results grid is not a date-key subset of the full grid. No nearest-date
matching between those protocols is allowed. Historical 12-month
AllModels/Results values remain on the separate full-grid lineage.

The Results files' supplied `horizon_days` value is retained. The corresponding
date span may differ by up to two days because of interval-boundary conventions;
both dates remain authoritative identity fields.

### Time normalization

Legacy `Obs_Time` does not have a universal origin: the notebook applies a
120-day H1N1 offset. Canonical plot time is therefore computed from dates:

```text
observation_day_2009 = observation_date - 2009-01-01
prediction_day_2009  = prediction_date - 2009-01-01
```

Original time fields remain available through source provenance.

The prediction-comparison warmup is likewise date-based. A row is eligible
when its ISO `observation_date` is on or after the subtype's initial inference
date plus 1,095 days:

| Subtype | Initial inference date | Inclusive comparison cutoff |
|---|---|---|
| H3N2 | `2009-01-02` | `2012-01-02` |
| H1N1 | `2009-05-02` | `2012-05-01` |

Results-grid evaluations carry these origins in `model_start_date`. Full-grid
evaluations leave that field blank, so the same values are explicit repository
defaults rather than inferred from filenames or integer times. The comparison
object records the relative day, both dates, inclusivity, and whether the
origin came from the evaluation, repository default, or caller override. This
also makes filtering idempotent for files that upstream code already trimmed.

### Pairwise comparison contract

Pairwise differences are derived, not stored:

```text
delta_emd = emd_comparator - emd_target
```

For full-model comparisons, a positive value means that the full model has
lower EMD than the comparator.

Every comparison must:

1. inner-join on exact `evaluation_id`;
2. require identical protocol and result version;
3. report source counts, paired count, and unmatched identifiers;
4. refuse silent row dropping or nearest-date pairing;
5. record target and comparator method IDs in the figure manifest.
6. apply any warmup to `observation_date` after date normalization and report
   its concrete subtype cutoff.

This supports full versus null, full versus SLM, full versus MLR, explicit
cross-validated-versus-baseline comparison, and diagnostics such as raw versus
rare-filtered or linked versus unlinked.

The maintained figure recipes use `results_grid` and
`full_model_cross_validated_rare_filtered` for all 6-, 9-, and 12-month
horizons. Comparator selection accepts only canonical method IDs. The legacy
function that selected MLR while labeling it single-site/SLM has no maintained
output or compatibility recipe.

Each registered paper EMD output binds its horizon and comparator as immutable
recipe parameters. The renderer refuses a mismatched call and refuses to draw
if either method has an unmatched evaluation. It prints both subtype cutoff
descriptions by default, in addition to retaining them in the audit object.
Legacy multi-output slide functions do not yet satisfy this contract and remain
candidate-only until a canonical wrapper and visual parity check are complete.

The September 30 source correction changed `Null` in all 388 nine-month rows
and no other field. The October 13 release supersedes that current-state role:
its four 9-month files are byte-identical to the September 30 files, while its
6- and 12-month files supersede the October 10 predecessors. Only `best
GenModel` and the paired `CV parameter` change: all 97 H3N2/HA 12-month rows,
13 H1N1/NA 6-month rows, and 10 H1N1/NA 12-month rows. The other nine files are
byte-identical. Therefore the four maintained 6-month and four maintained
12-month EMD figures require regeneration; all four 9-month figures retain
their numerical content.

The registered 6-, 9-, and 12-month best-versus-baseline diagnostics follow the
same exact pairing and warmup contract but are deliberately kept outside the
paper figure set. The scientist-facing call is
`figures.plot_emd_best_vs_baseline(horizon_months)`; the 9-month compatibility
wrapper remains available. Outputs are
`figures/diagnostic/fig-emd-best-vs-baseline-{horizon}m.pdf`. For this comparison
the general definition above becomes `baseline EMD - best-model EMD`; a
positive plotted value therefore means that the cross-validation-selected
model has lower EMD.

Every authoritative October 13 row satisfies the lower-bound invariant:
`Lower Bound` is less than or equal to every reported model EMD in all 1,164
cross-validation evaluations. There are no identity-specific exceptions and
the producer does not clamp, replace, or silently omit values.

### Deterministic EMD producer

`scripts/prepare_emd_data.py` validates all 60 archived legacy EMD CSVs, the
four superseded September 24 summaries, the 12 authoritative October 13
summaries, and the 12 validation-only September 30/October 10 predecessors
before writing the four canonical tables. The paired September releases must
have identical evaluations, metadata, non-null results, and selected
hyperparameters, with at least one null value differing. Every old pre-CV 6-
or 9-month evaluation must be present in the authoritative release. The 12
current files must share the same 97-row panel identities, and comparison with
their predecessors permits changes only when `best GenModel` and `CV parameter`
change together; every other column must match exactly. All authoritative rows
require `Lower Bound` not to exceed any model EMD. The builder refuses to
overwrite an existing canonical table unless explicitly requested. The current
materialization uses:

```text
python3 scripts/prepare_emd_data.py \
  --repo-root archive/pre_plot_reorg_2026-08-26/original \
  --cv-9m-root archive/emd_cv_2026-09-24/original/tmp \
  --corrected-cv-9m-root archive/emd_cv_2026-10-13/original/tmp \
  --corrected-cv-6-12-root archive/emd_cv_2026-10-13/original/tmp \
  --previous-corrected-cv-9m-root archive/emd_cv_2026-09-30/original/tmp \
  --previous-corrected-cv-6-12-root archive/emd_cv_2026-10-10/original/tmp \
  --output-dir data/plotting/inputs/emd
```

The `emd_v6` materialization, produced by `emd_migration_v6`, contains 2,591
evaluations: 1,427 `full_grid` and 1,164 `results_grid` rows. It contains 16,578
values, 10 methods, and 76 source
records: 60 legacy EMD files, four September 24 provenance files, and 12
authoritative October 13 files. The validation-only September 30 and October 10
snapshots are checked but are not duplicated in `sources.csv`. The older
6-/12-month Results families remain auditable but supply no current
`results_grid` values. Canonical hashes live in the manifest and migration log;
focused tests regenerate the tables in temporary directories to verify
determinism without touching the canonical release.

## Other canonical data families

| Family | Canonical organization |
|---|---|
| Frequencies | Four byte-preserved headerless matrices under standardized subtype/segment filenames; their legacy row/time interpretation is documented rather than guessed during migration. |
| Selection | Common mutation table plus explicit subtype/segment fields and normalized functional annotations. `mutation_count` replaces ambiguous `MutationFrequency`. |
| Fixation | Immutable prediction records separated from metric tables; model, source variant, subtype, segment, dates/times, and minimum-time cutoff are explicit columns. The paper date-filtered and extended all-time SLM families remain distinct. |
| Background | Two `background_fitness_v2` observation tables retain their distinct row memberships. Shared fields use scientific names; plot preparation applies the figure-specific duration cutoff to source rows, then retains the earliest eligible record for each `(subtype, segment, position, state)` mutation identity. |
| Clades | Long table with `date`, `clade`, `frequency`, and `series`. Clades are selected on an explicit reference window, filtered before normalization, then normalized within subtype/segment/series/date. |
| Structure | Immutable CIF/image assets are separate from the normalized derived SASA/selection table; generated PDB and visualization scratch files are archive-only. |
| Examples | One compressed, documented trajectory bundle per example rather than thousands of implicit filenames. |
| Surveillance | Canonical cumulative-count table with upstream GISAID provenance but no dependency on the private raw files during plotting. |

### Background-mutation sampling contract

The two canonical background CSVs share the ordered `background_fitness_v2`
schema:

```text
subtype, segment, source_variant, position, state, start_time, frequency,
initial_time, final_time, duration_days, selection_coefficient,
initial_background_fitness, initial_population_fitness,
final_background_fitness, background_fitness_change, outcome, prediction
```

Legacy `H0/H1/dH` and `w0_bg/w1_bg/dw_bg` columns both map to the three
background-fitness fields. Legacy `fitness` is the mutation's
`selection_coefficient`; `w0_pop` is `initial_population_fitness`. These
mappings avoid the earlier misleading `initial_fitness` and
`population_fitness` names.

The plotting sample is one record per biological mutation identity:

```text
(subtype, segment, position, state)
```

The canonical tables contain multiple sliding observation rows for many such
identities. These rows are not independent mutation emergences. When a figure
has a duration cutoff, preparation first retains source rows that meet the
inclusive cutoff and then selects the eligible row with the earliest
`start_time` for each mutation identity. The selected row's source
`initial_time`, `final_time`, outcome, and fitness quantities remain
authoritative. With no duration cutoff, this reduces to selecting the earliest
source row overall. Both cases are deterministic and independent of CSV row
order.

Without a duration cutoff, the population-fitness family yields 418 mutation
records (45 fixed, 373 lost), and the full family yields 414 (43 fixed, 371
lost). Filtering source rows first at the inclusive 90-day cutoff yields 315
population-fitness records (54 fixed, 261 lost) and 314 full-family records (54
fixed, 260 lost). Figure S13 uses the unfiltered population-fitness sample;
Main Figure 5 and Figures S14–S15 use the 90-day eligible sample.

The archive code confirms 0.05/0.95 terminal thresholds, but its classification
loop uses two consecutive samples at or beyond the threshold. Canonical plots
do not attempt to reinterpret those labels from the sparse plotting table.

### Clade-selection contract

For the current H3N2/HA recipe, clade totals are computed from
`series=observed_full` inside the open interval 2011-08-01 to 2015-06-01. The
legacy strict rule `sum(frequency) > 1.0` selects six clades: `3C`, `3C.2`,
`3C.2a`, `3C.3`, `3C.3a`, and `3C.3b`. Every series is filtered to that set
before normalization within `(subtype, segment, series, date)`. Because the
selection statistic is an unweighted sum over sampled dates rather than a time
integral, its window and cadence are part of the recipe.

## Code responsibilities

### `paths.py` and `outputs.py`

`paths.py` resolves repository-relative paths from an installed package or an
explicit project root and rejects traversal outside the configured roots.
`outputs.py` assigns each registered destination to one owner, refuses implicit
overwrites, locks concurrent renders, and commits completed files atomically.

### `prepare/emd.py`, `emd.py`, `emd_plot.py`, and data migration scripts

`prepare/emd.py` owns deterministic EMD normalization from the archived source
families. `emd.py` performs exact, provenance-aware method comparisons and
records concrete subtype-relative date cutoffs. `emd_plot.py` selects the
protocol by horizon and constructs separate null, SLM, or MLR recipes from an
explicit method ID. The non-EMD migration script owns the other normalized
inputs, derived tables, and their source-to-destination ledger. Plotting code is
a read-only consumer.

The maintained paper renderer deliberately preserves the archived EMD visual
language instead of applying a generic plotting layout. It uses the historical
two-grid geometry, line and guide hierarchy, panel/protein annotations, date
format, and comparator-specific labels. The MLR `+/-6.25` range is retained for
cross-horizon visual parity and leaves unclipped margin around the current
nine-month maximum absolute delta (`5.016999906`). True SLM histograms retain a
`0.40` frequency ceiling. Exact p-values and the concrete scales remain in the
render audit. The visible annotation follows one explicit policy: paired-test
results with `p > 0.05` have no p-value text; values with
`0.01 < p <= 0.05` are shown as an exact value rounded to two decimal places;
and values with `p <= 0.01` use the compact historical order-of-magnitude
notation. Because plotted `delta_emd` is comparator EMD minus target EMD, a
significant negative mean indicates that the full model is worse. Those labels
receive `*`, and any affected page includes the key `* full model worse`.

### `selection_plot.py`, `fixation_plot.py`, and `background_plot.py`

These focused modules own the final manuscript layouts that formerly depended
on broad legacy functions. `selection_plot.py` renders Main Figure 2 and S5–S6
from canonical frequency, selection, SASA, and structure inputs.
Its sequence plots keep labeled ticks and display limits separate. HA selection
uses limits 0-0.42 with the outer tick and visible spine ending at 0.4; NA
selection uses limits 0-0.26 with the outer tick and spine ending at 0.2. This
retains the manuscript tick labels while displaying the complete NA range
(maximum 0.24785). Figure S6 also uses SASA limits 0-100 with the outer tick
and right spine ending at 80, which contains its H1 maximum of 93.69. In the
HA sequence panels, the HA1/HA2 span lines sit at selection value 0.39 and the
domain labels at 0.42, above the protein labels and plotted curves.
`fixation_plot.py` renders Main Figure 3, full-model S7, and paper-date-filtered
SLM S16 from the canonical fixation metric tables. `background_plot.py` renders
Main Figure 5 and S13–S15 from one record per mutation identity: the earliest
eligible record for cutoff-based figures and the earliest record overall for
S13. It combines the corrected clade preparation only where the final
composition requires it.
They return audit objects as well as Matplotlib figures so numerical checks do
not have to be recovered from image text.

### `background.py`, `background_plot_data.py`, and `clades.py`

`background.py` first applies the requested inclusive duration cutoff to source
rows, then groups eligible rows by subtype, segment, position, and state and
deterministically selects the record with the earliest `start_time`. With no
cutoff it selects the earliest record overall. The selected row's source times,
outcome, and scientific values remain authoritative.
`background_plot_data.py` partitions the retained mutation records into
H3/N2/H1/N1 fixed/lost series and applies the exact legacy scientific formulas
without random jitter.

For Main Figure 5, the three background-summary rows preserve a small amount
of vertical plotting margin while drawing each visible left spine only between
its outer labeled major ticks. Panel B (background fitness advantage) uses a
data-focused range of -1.6% to +1.6%, with labeled ticks at -1.5%, 0%, and
+1.5%. This contains the complete retained point range (-1.5167% to +1.2290%)
without inheriting the much broader scale of the legacy, differently defined
quantity. These Main-Figure display settings do not alter the Supplementary
Figure S14 histogram scale.

`clades.py` replaces the ineffective loop-local notebook filtering pattern. It
selects clades by the declared observed reference window and strict cumulative
frequency rule, applies the selection to every series, and only then performs
per-date normalization. It rejects combined subtype/segment tables unless the
caller partitions them first.

### `styles.py`, `registry.py`, and `render.py`

`styles.py` owns immutable, versioned presentation profiles. `registry.py` owns
physical canvases and purpose-qualified output paths. `render.py` applies a
profile for one render. Its `legacy_style_context` also updates and restores the
historical module globals and mutable default dictionaries; this is required for
faithful paper/slide/poster switching in the compatibility functions.

### Root `figures.py` and `legacy/`

`legacy/figures.py` contains only the influenza portion of the historical
plotting module, and `legacy/mplot.py` preserves its plotting primitives. The
unrelated CH505/HIV/Wright-Fisher tail is archive-only. Root `figures.py` is the
readable scientist-facing facade: it contains the explicit canonical functions
called by `figures.ipynb`, their data defaults, scientific parameters,
presentation version, and output handling. Unadapted historical functions are
available as `figures.legacy`, and ordinary legacy function lookup remains for
older notebooks. Raw global changes such as `FIG_DIR = ...` must be made
explicitly through `figures.legacy`; maintained wrappers use `output_file` and
a named paper or slide version instead. Root `mplot.py` remains a module alias.

Focused selection, fixation, background/clade, and EMD renderers now cover the
final Main Figure 2–5 and Supplementary Figure S5–S19 set. Any remaining
diagnostic, slide, or older manuscript renderer can still be migrated
incrementally behind the same manifest, style, canvas, path, and output
contracts without changing the canonical data layout.

## Figure manifest

The figure manifest is the authoritative catalog of active and candidate figures.
Purpose, run health, and migration state are separate fields so an old slide is
not mistaken for an authoritative paper figure merely because it was executed
recently. It currently records 44 recipes: 20 active, 14 candidate, 6 legacy,
and 4 blocked. All active/candidate recipes consume only canonical input sets or
canonical derived products; archive-only paths remain attached solely as
provenance for legacy recipes.

Each record declares its notebook provenance, implementation, input and derived
product IDs, transformations, comparison semantics where applicable, style
profile, registered canvas, owned outputs, and blockers. Parameterized families
also declare their dimensions and expected output expansion. The validator
checks storage zones, on-disk paths, purpose-qualified outputs, output ownership,
style/canvas names, collision declarations, and the paper SLM/AUC lineage.

## Style and canvas contract

### Separate style from canvas

A style profile owns typography, stroke and marker scales, semantic colors,
Matplotlib settings, and save defaults. A figure specification owns canvas
dimensions, normalized panel layout, output name, profile choice, and rare local
overrides.

No notebook cell or previously imported module may implicitly choose a style.
Rendering uses `matplotlib.rc_context` and immutable profile records.

### Column and presentation widths

The existing code defines:

| Canvas helper | Width |
|---|---:|
| `single_column` | 8.8 cm / 3.4646 in |
| `one_and_half_column` | 12.1 cm / 4.7638 in |
| `double_column` | 18.4 cm / 7.2441 in |
| `slide_full` | 10.5 in |

These are helpers only. Each manifest entry stores exact width and height because
many current figures use custom dimensions.

### Versioned style profiles

The initial profiles preserve the intended constants near the start of
`figures.py` while explicitly recording the historically different `mplot.py`
defaults.

| Field | `paper_v1` | `slides_v1` | `poster_v1` |
|---|---:|---:|---:|
| Font family | Helvetica | Avenir | Avenir |
| Panel label | 10 pt | 18 pt | 24 pt |
| Axis/annotation text | 8 pt | 18 pt | 24 pt |
| Tick text | 8 pt | 18 pt | 24 pt |
| Figure data line | 0.6 pt | 1.5 pt | 1.5 pt |
| Manual detail stroke | 0.4 pt | 0.4 pt | 0.4 pt |
| Figure scatter area | 6 pt² | 42 pt² | 42 pt² |
| Legacy `mplot` default line | 0.8 pt | 3.0 pt | 3.0 pt |
| Legacy `mplot` spine | 0.4 pt | 1.5 pt | 1.5 pt |
| Legacy `mplot` tick width | 0.2 pt | 0.75 pt | 0.75 pt |
| Legacy `mplot` small marker | 6 pt² | 24 pt² | 24 pt² |
| Legacy `mplot` default scatter | 8 pt² | 8 pt² | 8 pt² |

`legacy_current_v1` preserves the mixed poster settings that happened to be
active at audit time. It is retained for archive-only recipes, but is never the
implicit default for a paper or slide render. The scoped compatibility adapter
has been visually checked against representative paper and slide PDFs; each
remaining recipe can be moved to focused primitives incrementally.

An optional `grant_v1` profile uses Arial, 12-pt panel labels, 8-pt text/ticks,
0.6-pt figure lines, and 7.8-pt² figure markers.

Shared palette values are versioned semantically:

| Meaning | Color |
|---|---|
| Ink | `#252525` |
| Muted / neutral | `#969696` |
| Beneficial | `#EB4025` |
| Deleterious | `#3E8DCF` |
| Model | `#FFB511` |
| H3N2 HA | `#C35999` |
| H3N2 NA | `#DDA0C3` |
| H1N1 HA | `#81AB5B` |
| H1N1 NA | `#ABCE7D` |

### Initial canvas registry

The following dimensions are extracted from the current plotting functions and
serve as parity targets.

#### Paper and general figures

| Renderer | Width × height, inches |
|---|---:|
| Overview true / simulated component | 1.600 × 0.700 |
| Selection structure / SASA | 7.2441 × 3.6220 |
| Frequency-selection-structure | 7.2441 × 4.8294 |
| Frequency-selection | 7.2441 × 4.0245 |
| Selection over sequence/time | 3.4646 × 2.3097 |
| EMD full-vs-MLR or full-vs-SLM | 7.2441 × 3.8127 |
| EMD full-vs-null | 7.2441 × 3.2928 |
| EMD plus clade | 7.2441 × 4.5276 |
| Clade | 3.4646 × 4.1575 |
| Fixation probability plus AUC | 7.2441 × 2.5872 |
| Fixation probability | 3.4646 × 3.4646 |
| AUC | 3.4646 × 5.1969 |
| Background difference | 3.4646 × 3.4646 |
| Background plus clade | 4.7638 × 3.4027 |
| Background rate / distribution / selection difference | 4.7638 × 4.7638 |
| HA circle | 3.4646 × 3.4646 |

#### Slide figures

| Renderer | Width × height, inches |
|---|---:|
| Surveillance, fix/loss example, simulation | 10.500 × 4.200 |
| Overview true component | 3.000 × 2.000 |
| Overview simulated component | 4.200 × 2.000 |
| Combined overview | 10.500 × 3.000 |
| Frequency | 8.500 × 1.600 |
| Selection | 3.400 × 1.600 |
| Selection subset | 5.250 × 3.2447 |
| dH or EMD distribution | 5.000 × 3.000 |
| Example frequency | 10.500 × 3.500 |
| AUC component | 2.625 × 5.250 |
| Fixation-probability component | 5.250 × 5.250 |
| EMD | 10.500 × 5.8333 |
| Joint EMD | 10.500 × 5.250 |

Reference-PDF checks record page dimensions, Matplotlib versions, and rasterized
comparison metrics. Existing PDFs contain a mixture of Avenir, Helvetica, and
fallback DejaVu fonts, so style is validated from the actual reference artifact
rather than inferred from a filename such as `-slides`.

## Reproduction workflow

The working workflow is:

1. Install `environment-plotting.yml` and install the local package.
2. Validate `figure_manifest.yaml` and the canonical release.
3. Rebuild data into a fresh temporary directory when checking determinism; the
   builders refuse to replace the maintained release by default.
4. Open `figures.ipynb`, run its setup cell once, and then run the plainly
   labeled cell for the desired figure family. Each runnable cell shows its
   canonical inputs, scientific selectors, version, canvas, and output path.
5. Begin at the named wrapper in root `figures.py`. It prints the exact source
   file containing the marks, axes, labels, and layout when those remain in a
   focused or historical renderer. Manifest/registry validation stays
   underneath this direct interface.
6. Figures marked **pending adapter** stay visible in the notebook but are not
   given a misleading run cell. Connect their canonical table to a transparent
   wrapper or corrected renderer before promoting them.

The manifest and output manager refuse missing declared inputs, path escapes,
undeclared styles/canvases, collisions, and implicit overwrites. A complete
single-command batch renderer is intentionally deferred until the remaining
legacy functions are converted to canonical loaders; the manifest makes that
work incremental rather than notebook-order dependent.

## Validation requirements

Before an original path is retired from the working tree:

- its archive hash must match;
- canonical row keys and required columns must validate;
- retained values must match exactly, except documented validation of rounded
  legacy copies;
- pairwise EMD comparisons must report paired and unmatched counts;
- random processes and visual jitter must use declared seeds;
- its active/candidate manifest consumers must point only to canonical records;
- representative paper and presentation renders must pass page-size, font,
  label, limit, and raster comparison checks before global style switches are
  removed;
- any intentional scientific or visual change must be logged separately.

## Adding or changing a figure

1. Register or reuse a canonical dataset and schema.
2. Add a preparation function only if the required result is reproducibly
   derived.
3. Write a renderer that accepts prepared data and returns the figure and axes.
4. Add a semantic figure ID with explicit comparison parameters, canvas, style,
   and output.
5. Add schema and smoke tests.
6. Render against the intended paper, slide, or poster reference.
7. Record the change and verification evidence in the migration log.
