# Plotting migration log

**Status:** Archive-backed reorganization executed; authoritative EMD release and final Main Figure 1–5 / Supplementary Figure S5–S19 plotting map integrated
**Last updated:** 2026-10-15

This is the durable audit trail for reorganizing the influenza plotting code and
data. It records what each legacy file becomes, why a consolidation or rename is
safe, how it was validated, and where the original is archived.

The companion document, [Plotting code and data layout](plotting-layout.md),
describes the intended end state. This document records the path taken to reach
that state.

## Migration rules

1. **Archive before changing.** Every in-scope file is copied or moved into a
   dated archive with its original relative path before the working copy is
   renamed, normalized, or consolidated.
2. **Never use Git as the only backup.** Much of the current plotting work is
   untracked or modified, and private data are intentionally excluded from Git.
3. **Hash every original.** The archive manifest records SHA-256, byte count,
   row count where applicable, and schema identifier.
4. **Use family-aware scientific mappings.** Ambiguous labels retain their
   source-family provenance. Confirmed mappings are encoded explicitly rather
   than inferred from similar names.
5. **Separate organizational parity from scientific corrections.** The first
   pass preserves intended calculations and appearance. Corrections to filters,
   statistical logic, or row selection are logged and reviewed separately.
6. **Keep immutable inputs separate from regenerated data.** A plotting step
   must not overwrite an input table.
7. **One owner per output.** The build fails if two figure recipes or two data
   preparation steps claim the same destination.
8. **Append to this log.** Completed migrations are never rewritten without a
   dated correction explaining the change.

## Status vocabulary

| Status | Meaning |
|---|---|
| `planned` | Proposed mapping; no files moved or rewritten. |
| `needs_decision` | Scientific meaning or manuscript relevance must be confirmed. |
| `in_progress` | Archive exists and conversion is underway. |
| `converted` | Canonical file exists but full validation is incomplete. |
| `verified` | Values, schema, provenance, and consuming figures passed validation. |
| `archived` | Original is preserved and no longer used by the active build. |
| `legacy_keep` | Historical version is intentionally preserved but not promoted. |

## Baseline inventory

The audit used the working tree as it existed on 2026-08-26. No files were
changed during the audit.

| Item | Baseline observation |
|---|---|
| `data/` | Approximately 2.0 GB and 4,612 files. |
| `data/GISAID/` | Approximately 1.8 GB, 120 files, and no direct use by `figures.ipynb`. |
| `data/emd/` | 60 CSV files, 9 schemas, approximately 0.75 MB. |
| `figures.py` | 9,473 lines and 65 top-level functions. |
| `mplot.py` | 1,319 lines of general plotting helpers. |
| `figures.ipynb` | 73 cells: 63 code and 10 markdown; 33 code cells have saved execution counts. |
| `figures/` | 301 PDFs, 7 PPTX files, 3 PNGs, 3 generated CSVs, and `.DS_Store`. |
| Git state | Plotting files include untracked, staged, and modified material; Git history is not a complete snapshot. |

## Decisions recorded before migration

| Decision | Record |
|---|---|
| `DEC-001` | A figure manifest, not notebook execution count, defines the active build. |
| `DEC-002` | EMD data must support full-model comparisons with null, SLM, and MLR, plus diagnostic comparisons between methods. No single comparator is privileged in storage. |
| `DEC-003` | EMD values use a normalized method table as the source of truth; plotting loaders may generate wide views. |
| `DEC-004` | Paper, slide, poster, and optional grant appearance are versioned style profiles. Canvas size is a separate per-figure specification. |
| `DEC-005` | Existing mixed `figures.py`/`mplot.py` styling is preserved temporarily as `legacy_current_v1` until reference-PDF comparisons pass. |
| `DEC-006` | Immutable upstream analysis outputs live under `data/plotting/inputs/`; locally reproducible intermediates live under `data/plotting/derived/`. |
| `DEC-007` | Private GISAID material is removed from active plotting data but remains in a private ignored archive with provenance. |
| `DEC-008` | Organizational changes and scientific bug fixes are separate reviewable changes. |
| `DEC-009` | In every Results EMD release, GenModel columns are full-model results; GenModel, SLM, and MLR variants all have rare mutations filtered. |
| `DEC-010` | FMSS means full model/single-site simulation and SLFS means single-locus model/full-sites simulation; both remain legacy diagnostic methods. |
| `DEC-011` | EMD warmup filtering is inclusive at initial inference date + 1,095 days: H3N2 `2012-01-02`, H1N1 `2012-05-01`. The comparison reports both relative and calendar cutoffs. |
| `DEC-012` | Maintained EMD recipes select null, SLM, or MLR by explicit canonical method ID. The mislabeled single-site/MLR test has no maintained result. |
| `DEC-013` | Background plots use one record per `(subtype, segment, position, state)` mutation identity. The initial convention selected the earliest `start_time` before applying the duration cutoff; this ordering is superseded by `DEC-023`. |
| `DEC-014` | Clades are filtered to the selected set before per-series/date normalization. The reference-window sum threshold remains strict and cadence-dependent. |
| `DEC-015` | The working notebook and root `figures.py` are the scientist-facing interface. They favor visible, directly runnable recipes over a generic manifest dispatcher; the manifest remains validation and provenance infrastructure. |
| `DEC-016` | The 9-month `best GenModel` is a distinct cross-validation-selected full-model method and is the primary 9-month figure target. `GenModel (baseline)` remains a separately comparable method. Six- and 12-month recipes remain pre-CV until their matching releases arrive. |
| `DEC-017` | The 2026-09-30 9-month correction supersedes only the erroneous null EMDs in the 2026-09-24 cross-validation release. Both releases remain archived; all current 9-month row values use the corrected source for coherent same-run provenance. Only the 9-month null comparison requires numerical regeneration. |
| `DEC-018` | The 2026-10-10 6-/12-month cross-validation release is definitive. All maintained 6-, 9-, and 12-month paper comparisons use `results_grid` and `best GenModel`; same-run baselines remain diagnostic comparators. One source-reported H3N2/HA 12-month row has best-model EMD 0.050554 below `Lower Bound`; the value is preserved with an explicit identity-specific exception rather than altered or dropped. |
| `DEC-019` | The complete 2026-10-13 12-file release supersedes the September 30 and October 10 current-state roles and is the sole authoritative cross-validation source for `emd_v6`. The earlier snapshots remain immutable validation-only inputs. The correction changes only paired `best GenModel`/`CV parameter` cells in 97 H3N2/HA 12-month rows, 13 H1N1/NA 6-month rows, and 10 H1N1/NA 12-month rows; nine files are byte-identical, and all 1,164 authoritative rows satisfy the lower-bound invariant. |
| `DEC-020` | The supplied final PDFs define the active manuscript map for this phase: Main Figures 1–5 and Supplementary Figures S5–S19. In particular, S17, S18, and S19 are the 6-, 9-, and 12-month full-model-versus-SLM comparisons. The earlier diagnostic classification of the 6-/9-month SLM plots and the unrelated provisional S18/S19 placeholders are superseded. |
| `DEC-021` | EMD p-value text is omitted for `p > 0.05`; `0.01 < p <= 0.05` is displayed to two decimal places; and `p <= 0.01` uses order-of-magnitude notation. With `delta_emd = comparator EMD - target EMD`, a significant negative mean receives `*`; manuscript comparisons also include the key `* full model worse`. Exact statistics remain available in the render audit. |
| `DEC-022` | Main Figure 1 is an explicit editable-artwork exception. Its canonical source is the single-slide PowerPoint file and its canonical PDF is produced by a visually verified manual export. It is not represented as a data-driven Python plot. |
| `DEC-023` | For background plots with a duration cutoff, source rows are filtered first using the inclusive threshold, then the earliest eligible `start_time` is selected per mutation identity. This retains mutations whose first observed interval is short but which later persist long enough. With no cutoff, the earliest source row remains the representative. Source times and outcomes remain authoritative. |

## Migration ledger

The entries below describe family-level actions. During execution, each family
entry will be expanded into per-path records containing source and destination
hashes, row counts, schema checks, and figure verification.

### Notebook and Python code

| ID | Current source | Canonical target | Action and evidence | Status |
|---|---|---|---|---|
| `CODE-001` | Influenza portion of `figures.py:214-6331` | `src/influenza_figures/legacy/figures.py` plus explicit infrastructure modules | Isolated 49 influenza plotting functions behind tested paths, registry, styles, output ownership, EMD comparison, and preparation interfaces. Topic-by-topic renderer conversion can proceed incrementally. | `converted` |
| `CODE-002` | `figures.py:6341-9473` | Dated archive | Unrelated CH505/HIV/Wright-Fisher code is excluded from active imports and remains byte-preserved in the archive. | `archived` |
| `CODE-003` | `mplot.py` | `src/influenza_figures/legacy/mplot.py` plus root module alias | Retained intact for compatibility after representative golden-image validation; reduction to focused primitives is deferred. | `legacy_keep` |
| `CODE-004` | Import-time mutable style constants in `figures.py` and `mplot.py` | `styles.py`, `registry.py`, and `render.legacy_style_context` | Five immutable profiles, 31 canvases, purpose-qualified output ownership, and a scoped adapter now update/restore both historical layers. | `verified` |
| `CODE-005` | Root `figures.py` module alias | Readable canonical facade plus explicit `figures.legacy` namespace | Restored discoverable plotting functions with explicit inputs, scientific parameters, paper/slide version, canvas reporting, and output paths. Ordinary unadapted legacy function lookup remains available. | `verified` |
| `CODE-006` | Final Main Figures 2–5 and Supplementary Figures S5–S19 | Focused selection, fixation, background/clade, and EMD renderers behind readable root functions | Added one direct scientist-facing call per data-driven final figure, using canonical plotting inputs and registered manuscript canvases. Main Figure 1 remains the documented editable-artwork exception. | `converted` |
| `NB-001` | `figures.ipynb` | Thin ordered driver plus archived original | The read-only 12-cell catalog was a verified migration intermediate, but it removed the direct figure-making workflow and is superseded by `NB-003`. | `superseded` |
| `NB-002` | Cells marked `OLD`, invalid cells 67-69, and known superseded cells | Archive notebook | Preserved byte-for-byte only in the dated original notebook and excluded from the active driver. | `archived` |
| `NB-003` | Read-only manifest driver | Editable, executable `figures.ipynb` recipe book | Restored direct cells that visibly name canonical inputs, thresholds/cutoffs, style, canvas, output, and a function in root `figures.py`. Pending adapters are documented without fake or archive-backed run cells. | `verified` |
| `NB-004` | Provisional manuscript labels and diagnostic placement | Final Main Figure 1–5, Supplementary Figure S5–S19, then diagnostics | Reconciled the recipe book to the supplied final PDFs. The three SLM horizons are now S17–S19; best-versus-baseline remains diagnostic; each data-driven final figure has a single explicit runnable call. | `converted` |

### EMD data

The canonical EMD design is detailed in the layout guide. It keeps evaluation
metadata separate from method values so every valid pairwise comparison is
derived from the same exact evaluation identifiers.

| ID | Current source | Canonical target | Action and evidence | Status |
|---|---|---|---|---|
| `EMD-001` | 12 `EMD_*_FullModel_*_full.csv` files | `data/plotting/inputs/emd/evaluations.csv` and `values.csv` | Imported evaluation keys plus lower bound, null, raw full model, unlinked full model, and rare-filtered full model from the verified archive. | `verified` |
| `EMD-002` | 12 `EMD_*_FullModel_*_full_plus_MLR.csv` files | Same canonical EMD tables | Imported rare-filtered MLR values. All seven shared columns and all rows match paired `full` files exactly. | `verified` |
| `EMD-003` | Four 9-month `*_full_dates_plus_MLR.csv` files | Validation input, then archive | Validated as the exact 10-column union of paired 9-month `full` and `full_plus_MLR` files; no independent canonical rows. | `verified` |
| `EMD-004` | Four `EMD_12m_AllModels_*_filt.csv` files | Same canonical EMD tables | Imported rare-filtered SLM plus legacy FMSS and SLFS diagnostics on matching full-grid evaluations; this family's `FM` maps only to rare-filtered full model. | `verified` |
| `EMD-005` | Four `Results_EMD_12m_*.csv` files | Validation input, then archive | Validated as rounded and reformatted copies of AllModels values; canonical values use the full-precision source. | `verified` |
| `EMD-006` | Eight pre-CV `Results_EMD_{6m,9m}_*.csv` files | Retained provenance under `results_grid`, protocol version `1` | The four 6-month files are superseded by `EMD-013`, and the four 9-month files are superseded by `EMD-011`; none supplies current values. Every old evaluation key is verified present in its definitive release. | `verified` |
| `EMD-007` | 12 `data/emd/old/EMD_*_FullModel_*.csv` files | Dated archive | Validated as exact renamed projections of current full files: `LB_Dist=LB`, `Null_Dist=Null`, and `Medoid_Dist=FM`; they supply no canonical values. | `verified` |
| `EMD-008` | Four `data/emd/old/*_since_2012.csv` files | Historical archive with version metadata | Values differ from newer estimates; preserve as `legacy_since_2012` and never mix with current results. | `legacy_keep` |
| `EMD-009` | Root `h3n2_distances_to_the_future.csv` | Dated archive | Not consumed by an active/candidate manifest recipe; its additional nine-column information is preserved in the archive rather than promoted into the maintained EMD release. | `archived` |
| `EMD-010` | Mixed legacy `Obs_Time` filters and the mislabeled `plot_emd_vs_ss` test | `emd.py` and `emd_plot.py` | Replaced source-specific offsets with inclusive ISO-date cutoffs and explicit comparator IDs. All 6-/9-/12-month recipes use `results_grid`. | `verified` |
| `EMD-011` | Four `*_9m_hyperparam_selection_summary.csv` files | Canonical EMD release `emd_v3` plus `archive/emd_cv_2026-09-24/original/` | Imported the complete 388-evaluation rerun. `best GenModel` maps to the cross-validation-selected full-model method; `GenModel (baseline)` is a fresh stochastic rerun using the pre-CV parameterization, not a byte-identical historical value. Lower bound, null, SLM, and MLR come from the same run. Exact CV parameter labels and cell-level provenance are retained. | `verified` |
| `EMD-012` | Four `*_9m_hyperparam_selection_summary_corrected.csv` files | Canonical EMD release `emd_v4` plus `archive/emd_cv_2026-09-30/original/` | Replaced the erroneous null EMDs in all 388 evaluations. Evaluation metadata, lower bound, SLM, MLR, baseline, best model, and selected hyperparameters are exact matches to `EMD-011`. The earlier release remains provenance-only; all six current values on each row come from the corrected file. | `verified` |
| `EMD-013` | Eight `*_{6m,12m}_hyperparam_selection_summary_corrected.csv` files | Canonical EMD release `emd_v5` plus `archive/emd_cv_2026-10-10/original/` | Imported 388 evaluations at each horizon as complete same-run rows. The 6-month release contains every prior 6-month key and adds 61 evaluations; all 388 12-month rows are new `results_grid` identities. The source-reported H3N2/HA 12-month lower-bound inversion is retained and explicitly annotated. Final canonical and rendered-artifact hashes await the post-materialization verification entry. | `converted` |
| `EMD-014` | Twelve `*_{6m,9m,12m}_hyperparam_selection_summary_corrected.csv` files | Canonical EMD release `emd_v6` plus `archive/emd_cv_2026-10-13/original/` | Imported one authoritative same-run source for every panel and horizon. The lower-bound horizon association used during hyperparameter selection is corrected without changing the reported `Lower Bound` values. Relative to `EMD-012`/`EMD-013`, only paired `best GenModel` and `CV parameter` cells differ in the three documented panels; nine files are byte-identical and no lower-bound inversions remain. | `verified` |

For `EMD-001` through `EMD-007`, `EMD-011`, and `EMD-012`, `verified` refers to the
archive-to-canonical data release: schemas, values, scientific method mappings,
provenance, and deterministic reproduction have passed. `EMD-012` records the
integrated source correction; its final materialization and output checks are
recorded in the dated activity entry below. Notebook migration and
representative rendered-figure parity are separately recorded under the
notebook, code, output, and activity entries below.
`EMD-013` records the superseded October 10 source integration and remains
historically unchanged. `EMD-014` is the current release: the September 30 and
October 10 snapshots are validation-only inputs and supply no `emd_v6` source
records or current values.

### Other plotting data

| ID | Current source | Canonical target | Action and evidence | Status |
|---|---|---|---|---|
| `SEL-001` | Four root `selection_coeff_*_add_counts.csv` files | `data/plotting/inputs/selection/` | Row counts preserved, identity fields added, and `MutationFrequency` renamed to `mutation_count`; hashes and schemas are in `migration_map.csv`. | `verified` |
| `SEL-002` | Duplicate nested selection result filenames | Dated archive | Retired after archive verification; canonical `SEL-001` retains the additional mutation-count information. | `archived` |
| `SEL-003` | Short older `selection_coeff_*_since_*.csv` files | Dated archive | Scientifically different historical coefficients are preserved for legacy diagnosis but are not active plotting inputs. | `legacy_keep` |
| `FREQ-001` | Four headerless allele-frequency matrices | `data/plotting/inputs/frequencies/` | Byte-preserving standardized filenames retain the legacy matrix layout; subtype/segment identity is explicit in the path and manifest. | `verified` |
| `FIX-001` | `data/2025-02-fix-loss/` | Dated archive | Superseded projections retired from the active tree after hash verification. | `archived` |
| `FIX-002` | Full-model, paper SLM, and extended SLM prediction families | `data/plotting/inputs/fixation/predictions.csv` | Consolidated with explicit model, source variant, subtype, segment, date, and time fields; distinct SLM families remain separate. | `verified` |
| `FIX-003` | AUC and pfix files encoded by model/cutoff in filenames | `data/plotting/derived/fixation_metrics/` | Canonical metric tables retain model, source variant, cutoff, and presentation lineage; plots are read-only consumers. | `verified` |
| `BG-001` | Four `_with_w_pop.csv` files | Dated archive | Exact projections retired after validation against the richer population-fitness sources. | `archived` |
| `BG-002` | `_full.csv` and `_with_w_pop_fitness.csv` families | `inputs/background/full.csv` and `population_fitness.csv` | Both scientifically distinct row memberships remain in separate canonical variants rather than being silently merged. | `verified` |
| `BG-003` | Misleading sparse background names and implicit first-row mutation selection | `background_fitness_v2`, `background.py`, and `background_plot_data.py` | Unified shared scientific columns, added duration, and made the revised sampling rule explicit: apply the inclusive duration cutoff to source rows, then select the earliest eligible `start_time` per mutation identity and construct deterministic fixed/lost series. | `verified` |
| `CLADE-001` | Separate forecast files and tall forecast/observed files | `data/plotting/inputs/clades/h3n2_ha_frequencies.csv` | Consolidated to subtype, segment, series, ISO date, clade, and frequency columns. Forecast date encodes the horizon relationship; source provenance remains in `migration_map.csv`. | `verified` |
| `CLADE-002` | Notebook loop-local clade filtering | `clades.py` | Selects six clades on the declared reference window, filters every series, then normalizes within subtype/segment/series/date. | `verified` |
| `STRUCT-001` | CIF/PDB, SASA tables, cropped PNGs, and root scratch outputs | `inputs/structure/` and `derived/structure/` | Immutable structures/images are separated from a normalized SASA/selection table; duplicate scratch outputs are archive-only. | `verified` |
| `EXAMPLE-001` | Two fix/loss example directories containing 4,014 files | Two deterministic ZIP bundles | Each 2,007-member bundle passed ZIP integrity tests; member sources and hashes remain in the migration ledger and archive. | `verified` |
| `SURV-001` | `CumulativeNoOfSequencesByYear.csv` | `inputs/surveillance/sequences_by_year.csv` | BOM removed, columns normalized, and private-source provenance retained without a raw-data plotting dependency. | `verified` |
| `PRIVATE-001` | `data/GISAID/` | Private ignored dated archive | No active/candidate recipe reads it. The 1.8 GB working copy was retired only after archive hash verification. | `archived` |

### Generated artifacts and collisions

| ID | Current artifact or collision | Resolution | Status |
|---|---|---|---|
| `OUT-001` | `figures/fig-selection-structure-space.pdf` from cells 11 and 12 | Cell 12 is the candidate canonical producer; archived cell 11 remains explicitly legacy. | `converted` |
| `OUT-002` | `figures/fig-clade.pdf` from cells 20 and 21 | The tidy-data producer is the canonical candidate; the one-file-per-horizon producer remains archive-only. Corrected filter-before-normalize preparation is verified, while reference-PDF parity remains pending. | `converted` |
| `OUT-003` | `figures/fig-auc.pdf` from paper cell 28 and grant cell 70 | Purpose-qualified paper and grant paths have distinct owners. | `verified` |
| `OUT-004` | Selection slide PDFs from cells 53 and 54 | Cell 54 is retained as candidate provenance; the incompatible predecessor is archive-only. | `archived` |
| `OUT-005` | EMD slide outputs from cells 64 and 65 | Cell 65 is retained as candidate provenance; horizon-omitting legacy names are archive-only. | `archived` |
| `OUT-006` | FullModel AUC CSVs produced by several notebook cells | One canonical derived AUC table owns the values and all recipe dependencies. | `verified` |
| `OUT-007` | FullModel pfix CSVs produced by several notebook cells | One canonical fixation-probability table owns the values and all recipe dependencies. | `verified` |
| `OUT-008` | Overview CSVs stored in `figures/` | Consolidated as `data/plotting/derived/overview/trajectories.csv`; active `figures/` contains rendered artifacts only. | `verified` |
| `OUT-009` | Root HA-circle PDFs and `h3-h1-selection.pdf` | Approved destinations are purpose-qualified; old root artifacts are archive-only. | `converted` |
| `OUT-010` | `Caton` versus `Canton` slide filename | One spelling-only `canton` migration is recorded; it is not treated as a second analysis. | `verified` |
| `OUT-011` | Supplied final Main Figure 1–5 and Supplementary Figure S5–S16 PDFs, plus specified S17–S19 | One manuscript label now owns one canonical paper/supplement path. The final references resolve the former exploratory/diagnostic ambiguity; output fingerprints are recorded only after canonical regeneration and visual inspection. | `converted` |

## Scientific issues isolated from the layout migration

These findings were reviewed separately from the path/schema migration:

| Finding | Resolution |
|---|---|
| `plot_emd_vs_ss` selected MLR while labeling the comparator SLM. | Maintained recipes now require one explicit null/SLM/MLR method ID. The wrong legacy result is not maintained. |
| Background `full` and population-fitness families differ in row membership. | Both remain distinct canonical variants; shared measurements now use one schema. |
| Background plots implicitly selected the first row for each position/state within a subtype/segment file. | Replaced by an explicit, row-order-independent rule: apply the figure's inclusive duration cutoff to source rows, then select the earliest eligible `start_time` per `(subtype, segment, position, state)`. Sliding observations of the same mutation are not independent samples. |
| Clade filters rebound a loop variable without changing plotted data. | Replaced by filter-before-normalize preparation with an auditable selected-clade set. |
| An `OLD` selection-space cell supplied the H3 fitness-change file twice. | The cell remains archive-only and no maintained caller repeats it. The H1 file is canonical, but the dormant time panels also have unresolved matrix/date length mismatches, so the archived call was not superficially edited. |
| Minimum EMD time was interpreted differently across schemas. | Replaced by an inclusive ISO-date cutoff relative to explicit subtype inference origins. |

The corrected data and preparation paths have numerical tests. New focused
renderers remain parity candidates until their PDFs have been compared with the
archived references; scientific correctness and visual parity are recorded
separately.

## Per-path execution record template

When a planned family is migrated, append a record in this form:

```text
### YYYY-MM-DD — CHANGE-ID

- Source path(s):
- Archive path(s):
- Canonical destination(s):
- Relationship: exact duplicate | projection | superset | normalized | regenerated
- Source SHA-256:
- Destination SHA-256:
- Rows and schema before:
- Rows and schema after:
- Transformation command/version:
- Numerical validation:
- Consuming figure IDs tested:
- Render comparison:
- Reviewer and date:
- Notes or deviations:
```

## Activity log

### 2026-08-26 — `AUDIT-001`

- Completed a read-only inventory of the notebook, plotting modules, generated
  artifacts, and data families.
- Traced notebook cells to functions, inputs, outputs, and hidden side effects.
- Compared EMD, selection, fixation, background, clade, structure, and example
  data families for duplicates, projections, and schema differences.
- No repository files were moved, renamed, or rewritten.

### 2026-08-26 — `DOC-001`

- Added this migration ledger and the companion layout guide.
- Recorded the requirement to retain null, SLM, and MLR EMD comparisons and
  arbitrary diagnostic comparisons.
- Recorded explicit paper, slide, and poster style/canvas requirements.
- No data or plotting code was migrated.

### 2026-08-26 — `EMD-TOOL-001`

- Added `scripts/prepare_emd_data.py` and the deterministic
  `influenza_figures.prepare.emd` converter.
- Encoded the confirmed Results semantics: `GenModel` is the rare-filtered full
  model, while SLM and MLR are their rare-filtered alternatives.
- Expanded FMSS and SLFS to their confirmed legacy diagnostic method IDs.
- Validated all 60 source CSVs before producing temporary outputs: 2,087
  evaluations, 12,894 values, 9 methods, and 60 SHA-256-pinned sources.
- Temporary-output SHA-256 values were
  `bedec23233580a7161bde74f4772068ed28a9836a6a7732241780e6594c5d695`
  (`evaluations.csv`),
  `28c7e1755f96862a07a0ebccf1b8ec5a52f6c7d1456e466a9e75e69d6ade2cff`
  (`values.csv`),
  `6d0b9c8c1f987b670801e5c3ff945185e31e55cd62d66f83fbfaa1f656bc78fc`
  (`methods.csv`), and
  `2ff15a652f39e152634364eb2b7f59573ffd3b0f7e337973e427b81ba7712302`
  (`sources.csv`).
- Eleven focused EMD tests and all 30 discovered repository tests passed. The
  tests verify byte determinism, unique canonical keys, exact source mappings,
  overwrite refusal, and unchanged legacy hashes.
- No canonical output was written under `data/plotting/`; conversion was tested
  only in temporary directories pending the verified archive snapshot.

### 2026-08-26 — `EMD-MATERIALIZE-001`

- Used all 60 EMD CSVs under
  `archive/pre_plot_reorg_2026-08-26/original/data/emd/` as immutable source
  inputs and wrote the four canonical tables under
  `data/plotting/inputs/emd/`.
- Ran `python3 scripts/prepare_emd_data.py --repo-root
  archive/pre_plot_reorg_2026-08-26/original --output-dir
  data/plotting/inputs/emd` after the archive manifest and checksums existed.
- Materialized 2,087 evaluation rows, 12,894 value rows, 9 method rows, and 60
  source rows. These counts exclude CSV headers.
- Verified the canonical SHA-256 values as
  `bedec23233580a7161bde74f4772068ed28a9836a6a7732241780e6594c5d695`
  (`evaluations.csv`),
  `28c7e1755f96862a07a0ebccf1b8ec5a52f6c7d1456e466a9e75e69d6ade2cff`
  (`values.csv`),
  `6d0b9c8c1f987b670801e5c3ff945185e31e55cd62d66f83fbfaa1f656bc78fc`
  (`methods.csv`), and
  `2ff15a652f39e152634364eb2b7f59573ffd3b0f7e337973e427b81ba7712302`
  (`sources.csv`). All four files are byte-identical to the independently
  generated temporary reference.
- Re-ran 12 focused EMD migration tests and 6 EMD comparison-contract tests;
  all 55 repository tests discovered at materialization time passed. Coverage
  includes exact family relationships, confirmed Results/FMSS/SLFS semantics,
  source hashes, unique canonical keys, overwrite refusal, archive-to-release
  byte equality, byte determinism, unchanged legacy inputs, exact pair joins,
  and comparison-sign conventions.
- `EMD-009`, the separate root H3N2 distance table, is not part of these four
  core EMD tables; it was later classified as archive-only because no
  active/candidate manifest recipe consumes it.

### 2026-08-26 — `ARCHIVE-001`

- Created the local read-only snapshot at
  `archive/pre_plot_reorg_2026-08-26/original/` before rewriting or retiring any
  plotting path.
- Preserved 4,978 files totaling 2,111,191,458 bytes, including untracked,
  modified, staged, generated, and private material.
- Verified every source/archive pair by SHA-256 and recorded `SHA256SUMS`, a file
  manifest, schema inventory, path map, archive summary, and the contemporaneous
  Git state. All detailed snapshot records remain ignored because they can name
  nonpublic data.

### 2026-08-26 — `DATA-MATERIALIZE-001`

- Ran `scripts/migrate_plotting_data.py` exclusively against the verified
  archive and materialized the non-EMD canonical tree at `data/plotting/`.
- `migration_map.csv` contains 4,130 source records mapped to 36 canonical
  destinations across 17 schema families. Of these, 4,014 individual trajectory
  files became two deterministic 2,007-member ZIP bundles; both passed ZIP
  integrity checks.
- Standardized filenames, snake-case CSV columns, ISO dates, explicit
  subtype/segment/model/source-variant fields, and `mutation_count`. Legacy
  headerless numeric matrices remain byte-identical where reshaping them would
  break the compatibility renderer.
- Preserved the paper date-filtered SLM prediction/AUC lineage separately from
  the distinct extended all-time SLM family. Neither is silently pooled.
- The maintained canonical tree contains the non-EMD release, the four verified
  EMD tables, schema summary, provenance ledgers, and its own README.

### 2026-08-26 — `CODE-ORGANIZE-001`

- Moved the 49 influenza plotting functions into
  `src/influenza_figures/legacy/figures.py` and the historical primitives into
  `legacy/mplot.py`. The unrelated 3,135-line project tail remains archive-only.
- Replaced root `figures.py` and `mplot.py` with true module aliases so legacy
  global assignments still reach function globals.
- Added explicit repository paths, exact EMD joins, five immutable style
  profiles, 31 physical canvases, 43 output-owner specifications, atomic output
  commits, and collision checks.
- Added `legacy_style_context`, which changes and restores both historical
  modules' scalar globals and mutable default-property dictionaries for exactly
  one render.

### 2026-08-26 — `MANIFEST-NOTEBOOK-001`

- Canonicalized `figure_manifest.yaml`: 42 recipes (18 active, 13 candidate, 6
  legacy, 5 blocked), 26 input sets, 11 derived-product records, and 8 declared
  historical collisions.
- Active/candidate recipes now reference only canonical inputs/products and
  purpose-qualified output paths. Archive-only inputs remain attached solely to
  legacy provenance.
- Replaced the 73-cell notebook only after its SHA-256 matched the archived
  original (`14dfd3646e24fd1ab7ec188337d65cbe4fceb17288bafc9633b3935507da5c44`).
  The new deterministic notebook has 12 cells (7 code), no outputs or execution
  counts, and SHA-256
  `045b089a0013ba915308fc860a53ca62cf0cd7d038a13a0003cf826d5b858738`.
- All seven notebook code cells executed successfully in memory. They validate
  and summarize the manifest, expose styles/canvases, load canonical EMD data,
  compare the rare-filtered full model exactly against null/SLM/MLR on one
  protocol, and list blockers; they perform no writes or preprocessing.

### 2026-08-26 — `STYLE-PARITY-001`

- Rendered representative paper frequency/selection, paper EMD-vs-null, and
  slide simulation PDFs from archived inputs after relocating the legacy code.
- An initial render reproduced the audit-time poster globals and showed severe
  text over-scaling, confirming that Matplotlib `rcParams` alone was
  insufficient. Applying the new two-layer scoped adapter restored the intended
  paper and slide typography, line weights, axes, labels, and canvas layouts.
- Visual inspection found no clipping or semantic layout drift in the corrected
  samples. At 150 dpi, old/new raster comparisons were: frequency/selection
  MAE 4.1567 (5.8943% of pixels differed by more than 16 levels), EMD-vs-null
  MAE 4.3016 (4.2518%), and simulation MAE 0.3777 (0.3078%).
- The slide MediaBox matched exactly at 756 × 302.4 points. The two paper
  MediaBoxes differed by at most 0.324 points because the references were made
  with Matplotlib 3.10.9 and the verified environment uses 3.11.1; their aspect,
  panels, limits, and typography remained visually aligned. PDF byte hashes are
  therefore intentionally not treated as parity hashes.

### 2026-08-26 — `RETIRE-001`

- `scripts/retire_legacy_plotting_tree.py` performed two complete archive audits
  and an immediate per-file identity check before mutation. It never follows
  symlinks, never recursively sweeps a directory, and defaults to dry-run.
- Retired 4,935 files and 77 directories (2,104,739,891 logical bytes) from 364
  exact working-tree roots. This included the unused 1.8 GB `data/GISAID/`
  working copy, redundant legacy data families, root scratch inputs/outputs, and
  315 flat generated figure artifacts.
- Every retired file remains recoverable at the same relative path below
  `archive/pre_plot_reorg_2026-08-26/original/`; the archive was not modified.
- The active `data/` tree now contains only `data/plotting/` (apart from ignored
  operating-system metadata). `figures/` contains a README and the empty owned
  directories `paper/`, `supplement/`, `slides/`, `poster/`, `grant/`, and
  `diagnostic/`.

### 2026-08-26 — `VALIDATE-001`

- Path-aware manifest validation passes on the retired active tree: 42 figure
  recipes, 11 product records, and 8 declared collision records.
- All 4,978 archived files pass a fresh `SHA256SUMS` verification after active
  path retirement.
- The dependency-light system-Python suite passes 83 tests with 4 expected skips
  for optional scientific plotting imports. The pinned `gen` environment passes
  all 83 tests, including legacy alias/style checks.
- Sequential in-memory execution of all seven notebook code cells succeeds after
  retirement. The example `results_grid`, 6-month, H3N2/HA comparisons contain
  74 exact pairs each for full model vs null, SLM, and MLR.
- Re-running the notebook generator is an idempotent no-op, and a post-retirement
  dry run of the retirement tool reports zero remaining removal roots.

### 2026-08-27 — `BACKGROUND-CORRECT-001`

- Corrected the canonical background schema from `background_fitness_v1` to
  `background_fitness_v2`. `H0/H1/dH` and `w0_bg/w1_bg/dw_bg` now share the
  scientific fields `initial_background_fitness`,
  `final_background_fitness`, and `background_fitness_change`.
- Renamed legacy `fitness` to `selection_coefficient`, retained `w0_pop` as
  `initial_population_fitness`, and added explicit `duration_days`.
- Regenerated both active CSVs from the immutable archive in a temporary
  directory, compared the result before replacing the maintained materialized
  copies, and visually inspected their headers and sample rows. The outputs are
  15,234 full rows (`e28715ace3ed4ceb1ef0b47e24c0c5994cf05b749a5d55277470a6a1844a7d21`)
  and 15,277 population-fitness rows
  (`ca81508c402b5fa1f65129c958b9223924729738410b3e7e91d80f0c441aac16`).
- Verified that 15,012 observations shared by the two variants have identical
  background-fitness quantities after normalization.
- An intermediate migration treated changing source time tuples as distinct
  samples. A later audit showed that these were sliding observations of the
  same mutations and that this interpretation over-counted the plotting
  sample; `BACKGROUND-MUTATION-SAMPLING-001` below supersedes it.
- Added deterministic H3/N2/H1/N1 plot-series preparation without random
  jitter. Its sampling stage now follows the superseding mutation-level rule.

### 2026-08-27 — `EMD-CORRECT-001`

- Defined H3N2 `2009-01-02` and H1N1 `2009-05-02` as the explicit initial
  inference dates recorded by the newer Results files.
- Replaced source-specific integer-time filtering with an inclusive ISO-date
  comparison at day 1,095: H3N2 `2012-01-02` and H1N1 `2012-05-01`.
  Comparison results report relative day, initial date, cutoff date,
  inclusivity, and date provenance.
- Added comparator-driven recipes. Six- and nine-month comparisons use
  `results_grid`; 12-month comparisons use `full_grid`; null, SLM, and MLR are
  separate method IDs. The wrong legacy SLM-labeled/MLR result is not promoted
  or copied into a maintained output.
- Bound the horizon and comparator to each registered paper output. Rendering
  now refuses a mismatched specification or any unmatched evaluation IDs, and
  prints the relative/calendar cutoff report by default.
- The six-month post-warmup panel counts are H3 73, N2 82, H1 81, and N1 81 for
  each of the three Results-grid comparators. Already-trimmed input fixtures
  produce the same date-based selection without an additional manual offset.
- Updated the deterministic notebook to apply and print the cutoff; its new
  SHA-256 is
  `c767c081e196f28a505838418f03f7e1cbbc31b5bd7483c94e5868f9604e10fd`.
- Rendered temporary paper-profile null, SLM, and MLR candidates at the
  registered canvas for visual inspection. No generated candidate was placed
  in `figures/`; reference-PDF parity remains a distinct verification step.
- Kept the legacy multi-output MLR slide recipe as a candidate rather than an
  active corrected recipe: its historical renderers still use wide full-grid
  files and strict `Obs_Time > t_min` filtering. The canonical data preparation
  and slide typography/canvases are ready, but a focused wrapper and visual
  parity check remain necessary.

### 2026-08-27 — `EMD-TERMINOLOGY-001`

- Corrected MLR to **multinomial logistic regression**. Canonical release
  `emd_v2` uses method ID
  `multinomial_logistic_regression_rare_filtered`, model family
  `multinomial_logistic_regression`, and transform version
  `emd_migration_v2`. The short-lived `emd_v1` spelling incorrectly expanded
  the acronym as “multilinear regression.”
- Regenerated all four EMD tables from the immutable 60-file archive. The
  evaluation table remains byte-identical and all 12,894 numerical values and
  row-level source references are unchanged. Exactly 2,087 MLR rows use the
  corrected controlled-vocabulary identifier.
- Recorded the release hash transition:
  - `evaluations.csv` remains
    `bedec23233580a7161bde74f4772068ed28a9836a6a7732241780e6594c5d695`;
  - `values.csv` changes from
    `28c7e1755f96862a07a0ebccf1b8ec5a52f6c7d1456e466a9e75e69d6ade2cff`
    to `e7f43b5feada2268481aaba70a99755da604897a3ecbbb08243d58ff1269c08c`;
  - `methods.csv` changes from
    `6d0b9c8c1f987b670801e5c3ff945185e31e55cd62d66f83fbfaa1f656bc78fc`
    to `5715ec3ab6039f02e580e216ff9411bd7b514b03dc6dd55570c9598985751cdf`;
  - `sources.csv` changes from
    `2ff15a652f39e152634364eb2b7f59573ffd3b0f7e337973e427b81ba7712302`
    to `f19a41b00935fef67117f642b4a8bae5824d3a6d3f18bc79c3c738a1d3a0a770`.
- Preserved `MLR`, `MLR_remove_rare_mut`, archived filenames, source-column
  provenance, plotting function names, registry keys, and output filenames.
  `compare_emd` temporarily accepts the old method ID as an input-only
  deprecated alias and returns the corrected canonical ID.

### 2026-08-27 — `CLADE-CORRECT-001`

- Replaced the ineffective loop-local filter with a preparation function that
  selects clades on the declared observed reference window, filters every
  series, and then normalizes within subtype/segment/series/date.
- The strict `sum(frequency) > 1.0` rule on the current H3N2/HA reference
  interval selects `3C`, `3C.2`, `3C.2a`, `3C.3`, `3C.3a`, and `3C.3b`; every
  normalized group sums to one.
- Confirmed that the duplicated H3 selection-space argument occurs only in an
  archived `OLD` cell. No maintained caller executes it. The correct H1 data
  are canonical, but the dormant legacy time panels have incompatible matrix
  and date-vector dimensions and were not reactivated.

### 2026-08-27 — `RETIRE-FOLLOWUP-001`

- A final active-tree audit found seven archive-verified legacy roots still
  present: `data/2025-02-fix-loss`, `data/Fig_5`, `data/Fig_6`, `data/Fig_S1`,
  `data/Fig_S2`, `data/Fix-loss_example_1`, and
  `data/Fix-loss_example_2`.
- The retirement tool rechecked every file against the immutable archive before
  removing 4,074 redundant files in 15 directories (49,179,596 logical bytes).
  All remain recoverable below the archive at their original relative paths.
- The initial post-action dry run passed and reported zero remaining roots.
  Dropbox File Provider later rehydrated six of those cloud placeholders when a
  further verifier touched them. The retirement tool again checked their hashes
  and removed the 2,065 restored files in 10 directories (26,550,915 logical
  bytes). A third placeholder probe was deliberately avoided; the completed
  guarded action, rather than cloud-placeholder enumeration, is the final local
  state check.

### 2026-08-27 — `NOTEBOOK-USABILITY-001`

- Superseded the read-only manifest-inspection notebook with an editable,
  executable recipe book. The archived 73-cell original remains byte-preserved;
  the template installer now refuses to overwrite subsequent notebook edits
  unless `--force` is explicit.
- Replaced the root `figures.py` alias with a readable scientist-facing facade.
  Canonical calls accept ordinary human names and explicit input/output paths;
  unadapted historical functions remain available as `figures.legacy`.
- Added direct run cells for the nine paper EMD outputs (full model versus null,
  SLM, or multinomial logistic regression at 6, 9, and 12 months), the H3N2 HA
  clade figure, and four allele-frequency slide panels. Each cell exposes its
  scientific selectors, paper/slide version, physical canvas, and destination.
- Kept incomplete families visible as markdown-only pending sections. At that
  stage, the background sampling rule was still under review, and paper EMD
  canvases could not be relabeled as slide layouts. The final background
  wrappers now implement the explicitly restored mutation-level selection.
- Replaced tests that enforced a maximum of 12 non-rendering cells and exact-byte
  regeneration with structural recipe checks, safe-edit preservation checks,
  direct-facade compatibility checks, and render smoke tests.
- Path-aware validation passes for all 42 manifest records. The complete suite
  passes 142 tests, and temporary smoke renders succeeded for all nine EMD
  combinations, the clade figure, and all four frequency-slide panels.

### 2026-08-27 — `EMD-STYLE-PARITY-001`

- Compared the canonical EMD renderer with the archived neutral and MLR paper
  PDFs and restored their visual grammar: separate trajectory/distribution
  grids, 1.5-point protein trajectories, the darker zero guide plus faint
  guides at `-4`, `-2`, `2`, and `4`, open bounded axes, explicit minor ticks,
  fully opaque histogram bars, standalone bold `A`/`B`, and the original
  protein/p-value annotation placement.
- Restored the paper wording and typography: eight `%b %Y` date labels,
  `Time prediction made`, comparator-specific two-line improvement labels,
  `Frequency`, and compact plotted inequalities of the form `p < 10^k`. Exact
  p-values remain available in the immutable render audit.
- Preserved corrected scientific behavior. The renderer still uses exact
  evaluation-ID pairing, subtype-specific inclusive 1,095-day calendar
  cutoffs, rare-filtered full/SLM/MLR values, and the corrected SLM and MLR
  method identities. The archived `fig-emd-vs-ss-*` outputs remain unsuitable
  as scientific SLM references because their notebook call selected MLR data.
- Retained two explicit no-clipping exceptions to literal legacy scales. MLR
  remained at `+/-6.25` because the then-current pre-CV nine-month data reached
  `6.078`.
  SLM histograms remain at frequency `0.40` because a canonical bin reaches
  approximately `0.346`; neutral and MLR histograms use the archived `0.25`
  ceiling. Neutral restores its horizon-specific trajectory/distribution
  ranges (`+/-5`/`+/-5.25` at 6 and 12 months, `+/-4.25`/`+/-4.5` at 9 months).
- Added artist-level regression tests for geometry, strokes, guide levels,
  bars, annotations, axes, dates, typography, and non-clipping. These tests
  intentionally validate the visual contract rather than pixel-identical data
  traces, because the canonical pairing and cutoff corrections change plotted
  observations.

### 2026-09-24 — `EMD-CV-9M-001`

- Preserved the four delivered 9-month cross-validation summaries byte-for-byte
  under `archive/emd_cv_2026-09-24/original/tmp/`, added an archive README and
  `SHA256SUMS`, verified every archived copy, and retired only the duplicate
  delivery copies from the working `tmp/` directory.
- Recorded source SHA-256 values of
  `85a34820d721d2b5ff0dd3416efce5987e58b94d51e5ee10ceee08ebd9183ea6`
  (H1),
  `5d37be7328d9891189e3cd0e7e32341b20b0c97cae917e67449b2472d9606582`
  (H3),
  `2e1ee4e7f5833541213e07b3cbc76c0db894f952a2ee0d9ea2d9f3d56a9fdad1`
  (N1), and
  `7238f36aaa1bdcc688b65cb8285954c951c9ce0a3d866602a4a95c627a7578c8`
  (N2).
- Promoted `best GenModel` to the explicit
  `full_model_cross_validated_rare_filtered` method and made it the primary
  9-month figure target. Preserved `GenModel (baseline)` as
  `full_model_rare_filtered`; it is a fresh stochastic rerun of the pre-CV
  parameterization, not an assumed copy of an older result. Exact `CV parameter`
  strings are stored in `values.selected_hyperparameters`.
- Replaced the complete 9-month Results run as one internally consistent unit:
  lower bound, null, SLM, MLR, baseline GenModel, and best GenModel all come from
  the same new source row. The 333 old evaluation keys are a subset of the new
  388-key release; 55 evaluations were added, while superseded 9-month sources
  remain provenance-only and supply no canonical values.
- Materialized `emd_v3` with 2,142 evaluations (1,427 `full_grid`, 715
  `results_grid`), 13,557 values, 10 methods, and 64 sources. Canonical SHA-256
  values are
  `6c9621c66324364289c89f883aed221ed98b1bba0a4f2623a6f7664d0aa1db4a`
  (`evaluations.csv`),
  `fef5347f1683cf3490324ed0d8668600a5bb03de42e2cea635940d1621a0ebd1`
  (`values.csv`),
  `6f346c48249e2aabd1da766aa8ae229656f35591e0fbe4fc7c03a8fa1141250a`
  (`methods.csv`), and
  `391f79ab7e9cd76a6d4c5dacbf3d261fa3e241b2ee5ef658973abc072f6471a9`
  (`sources.csv`).
- Regenerated the null, MLR, and SLM 9-month PDFs with the restored paper
  styling. After the inclusive 1,095-day warmup, all three contain 84 H3, 84
  N2, 81 H1, and 81 N1 exact pairs. The MLR `+/-6.25` scale is retained for
  cross-horizon visual parity and leaves margin around the new maximum absolute
  delta (`5.016999906`). Output SHA-256 values are
  `915003ea72afcaf349e8c2e53130ecec22fd9e46cd0f21d1d213ccc22d2edfdf`
  (null),
  `da6155316edc6badc039966bd7558ba0279fb80f4d7092a9461c33d62d7de1c5`
  (MLR), and
  `6b9cf292b3ccc6ff7114c6207e63ffd0625801b75b40da9cb90404deefae7782`
  (SLM); all three one-page PDFs passed visual inspection.
- Path-aware manifest validation and byte-deterministic rebuild checks passed.
  The complete base-environment suite passed 148 tests with 15 documented
  optional-dependency skips; a plotting-enabled environment then passed all 21
  EMD renderer/facade tests without skips. Python compilation also passed.

### 2026-09-24 — `EMD-CV-DIAGNOSTIC-001`

- Added a dedicated 9-month diagnostic comparing the cross-validation-selected
  `best GenModel` with `GenModel (baseline)` from the same definitive rerun.
  Exact `evaluation_id` pairing and the inclusive 1,095-day warmup are retained.
- Defined the diagnostic difference as `baseline EMD - best-model EMD`.
  Positive values favor the cross-validation-selected model; negative values
  favor the baseline.
- Added the direct scientist-facing call
  `figures.plot_emd_best_vs_baseline_9m` to a clearly separated runnable cell in
  `figures.ipynb`. Its output belongs to
  `figures/diagnostic/fig-emd-best-vs-baseline-9m.pdf`; the existing paper and
  supplement recipes remain unchanged.
- Rendered the one-page diagnostic on the `paper_v1` `double_over_1_9` canvas.
  It contains 84 H3, 84 N2, 81 H1, and 81 N1 exact post-warmup pairs, uses a
  non-clipping `+/-4` trajectory/distribution scale and `0.40` frequency scale,
  and passed full-page visual inspection. Its SHA-256 value is
  `ee9c3ca402f156b517108bf388471958a75e89291461adec4c8a973766e356ad`.
- Manifest/path validation, notebook-template verification, Python compilation,
  and whitespace checks passed. The complete base-environment suite passed 155
  tests with 17 documented optional-dependency skips; the plotting-enabled
  EMD/facade/registry/manifest/notebook suite passed all 68 tests without skips.

### 2026-09-30 — `EMD-CV-9M-NULL-CORRECTION-001`

- Preserved the four corrected cross-validation summaries byte-for-byte under
  `archive/emd_cv_2026-09-30/original/tmp/`. The original September 24 snapshot
  remains immutable under `archive/emd_cv_2026-09-24/original/tmp/` as
  superseded provenance.
- Compared every field in all 388 paired rows. The schema, evaluation identity,
  dates, lower bound, SLM, MLR, `GenModel (baseline)`, `best GenModel`, and
  `CV parameter` are unchanged. Only `Null` differs, in all 388 rows.
- Defined `emd_v4` so the corrected four files supply all six current values on
  each 9-month row. The four earlier cross-validation files remain in the
  68-source inventory but supply no current values. This preserves complete
  same-run row provenance while documenting the narrow scientific correction.
- Materialized 2,142 evaluations (1,427 `full_grid`, 715 `results_grid`),
  13,557 values, 10 methods, and 68 sources. A fresh rebuild matched all four
  maintained files byte-for-byte. Canonical SHA-256 values are
  `6c9621c66324364289c89f883aed221ed98b1bba0a4f2623a6f7664d0aa1db4a`
  (`evaluations.csv`),
  `1b74ef7c0b48b4f0d21fa4cc79d943699c78f5fd16cb767cd15f75a920a15b97`
  (`values.csv`),
  `6f346c48249e2aabd1da766aa8ae229656f35591e0fbe4fc7c03a8fa1141250a`
  (`methods.csv`), and
  `42fd82df159a549c82f2db34644cb742d8b88994070dd202902c638be88525f9`
  (`sources.csv`). Both cross-validation archives passed their recorded
  `SHA256SUMS` checks.
- Limited numerical figure impact to
  `figures/paper/fig-emd-vs-null-9m.pdf`. The 9-month MLR, SLM, and
  best-versus-baseline plots do not consume the null method and therefore do
  not require regeneration.
- Retained the registered null plot scales. After the inclusive 1,095-day
  warmup, the corrected maximum absolute null-minus-best delta is
  `3.9700313123`, below the `+/-4.25` trajectory limit, and the largest
  normalized histogram bin is `19/81` (`0.234568`), below the `0.25`
  frequency ceiling.
- Regenerated the corrected null PDF as a 29,107-byte, one-page artifact with
  a 521.575 x 237.079 point page. Its SHA-256 is
  `fa0c733ab817ae6b05ff9d5390a093241519568f9bd4f5f2decc4ba63571e22a`.
  Full-page visual inspection found a clean render with no clipping and the
  established paper style intact.
- Focused validation passed 16 EMD migration tests and 37 plotting tests. Full
  discovery passed all 160 tests under system Python with 17 documented
  optional-dependency skips, and all 160 tests in the plotting environment
  without skips. Python compilation and whitespace checks also passed; the
  only emitted warning was the pre-existing legacy SciPy KDE deprecation.

### 2026-10-10 — `EMD-CV-6M-12M-001`

- Designated the eight delivered 6-/12-month cross-validation summaries as the
  definitive release and assigned their immutable archive root
  `archive/emd_cv_2026-10-10/original/tmp/`. Each file has the same 14-column
  schema and 97-row panel structure as the corrected 9-month release.
- Audited all 776 rows before integration. Natural keys are unique, dates are
  strictly increasing, nominal 180-/360-day horizons agree with calendar spans
  within two days, CV parameter strings are valid, and there are no missing,
  nonfinite, or negative numeric values. Observation identities match the
  definitive 9-month panel grids exactly.
- Verified that every one of the 327 prior 6-month `results_grid` evaluations
  occurs in the new release; the new files add 61 evaluations. Across overlaps,
  null values agree within the older files' decimal rounding, while lower bound,
  SLM, MLR, and baseline values are fresh same-run results. The 388 12-month
  rows are new `results_grid` identities rather than approximate matches to the
  irregular historical full grid.
- Recorded one source-reported exception instead of modifying it: H3N2/HA,
  observation `2013-07-01`, prediction `2014-06-26`, has
  `Lower Bound = 2.8464822104905676` and
  `best GenModel = 2.795927719950489`. The selected-model value is lower by
  0.0505544905400786. The remaining 775 rows satisfy the lower-bound check.
  A possible mismatch in the conventions used for the lower-bound and
  GenModel EMD calculations is under investigation; this is a hypothesis, not
  a canonical correction.
- Defined `emd_v5` with 2,591 evaluations (1,427 `full_grid`, 1,164
  `results_grid`), 16,578 values, 10 methods, and 76 source records. All six
  values on each definitive 6-/12-month row retain same-run source provenance;
  the superseded pre-CV Results families remain auditable but supply no current
  `results_grid` values.
- Switched all maintained 6-/9-/12-month paper recipes to `results_grid` and
  `full_model_cross_validated_rare_filtered`. Added registered 6- and 12-month
  best-versus-baseline diagnostics alongside the existing 9-month diagnostic.
  The inclusive warmup yields 84 H3/N2 and 81 H1/N1 pairs at every horizon.
- Source SHA-256 values are recorded below. Final v5 canonical-table and PDF
  hashes are deliberately omitted until the post-materialization rebuild,
  render, and visual-verification pass; that evidence should be appended rather
  than replacing this source audit.
- The archived exploratory notebook is intentionally still used for familiar
  reruns. Its current byte hash therefore differs from the immutable hash
  recorded on 2026-08-26 because outputs, execution counts, and notebook
  metadata changed. The working-notebook generator now pins the ordered cell
  types and source text (`09d970289744066bd95478a0ca3ef988e9a584ba1d1ba514edfae9e6422cc6da`)
  while ignoring rerun state; it does not modify the archived notebook.

| Source | SHA-256 |
|---|---|
| `h1_6m_hyperparam_selection_summary_corrected.csv` | `613c3371b881b99e168046f9f9966260af3e88f582c93c3472dc235ddd267483` |
| `h3_6m_hyperparam_selection_summary_corrected.csv` | `4b6a227e272905c6749ff001581b868095fbbe5aef33a35e98c18f294081c5ae` |
| `n1_6m_hyperparam_selection_summary_corrected.csv` | `b2ac8a2668471bb55a63007e7dd42415b5762a8d4ffec89d7e824ac865be7fb7` |
| `n2_6m_hyperparam_selection_summary_corrected.csv` | `f1a3c6508965f5e7811b0c32d7ed2b69b1d267e044d44296edf7ead8a9770b7f` |
| `h1_12m_hyperparam_selection_summary_corrected.csv` | `5878bc2f8b865e7172e14efa9ee615d241542f1881fa315825b5c78874c14e27` |
| `h3_12m_hyperparam_selection_summary_corrected.csv` | `ccd5d9c0ec1e956a740939383848cf40f2b21789b90ee0aa0a4dbdfdedbd7d9f` |
| `n1_12m_hyperparam_selection_summary_corrected.csv` | `167ee04d51e77d7a044ac3ca28623d79ccdfd8dba43c20bc29ae513e79ca06f5` |
| `n2_12m_hyperparam_selection_summary_corrected.csv` | `8ca16deb9a87dedc92daadc267554e89a4034fd42fbd413a0f9677c2bf5cefdb` |

#### Materialization and figure verification

- Materialized `emd_v5` and reproduced all four canonical files
  byte-for-byte in an independent temporary rebuild. SHA-256 values are
  `339c0111b221ae558abb4085af4d166d45ae1a4c86e2c58d79a404717487d14b`
  (`evaluations.csv`),
  `ab6c80411e71b55b9c38eea5273da656c41fa1bc588debeba01c3609382b20bd`
  (`values.csv`),
  `27bdcb2e64fd3bfb7663a42aec33be76e05eca7126adb3098c39efc2ba5dc7de`
  (`methods.csv`), and
  `fde07fbd1059b937c73eb3f70211dc2d2f2fd50226e8ea5463e6dd5d16ac46cb`
  (`sources.csv`). All eight archived source checksums passed again before
  the redundant delivery copies were retired from `tmp/`.
- Regenerated the six maintained 6-/12-month comparator figures and added the
  two corresponding best-versus-baseline diagnostics. The 6-month null
  histogram ceiling is `0.30`; the 12-month null trajectory/histogram limits
  are `+/-6.0` and `+/-6.25`; and the 12-month MLR limit is `+/-6.5`.
  These are the smallest registered expansions that avoid clipping while
  retaining the archived paper artist contract.
- Rendered and inspected every regenerated one-page PDF at full-page scale.
  All text, guide lines, trajectories, histograms, and significance labels are
  clean and unclipped. The null figures retain the 521.575 x 237.079 point
  canvas; MLR, SLM, and diagnostic figures retain the 521.575 x 274.513 point
  canvas.

| Figure | SHA-256 | Bytes |
|---|---|---:|
| `figures/paper/fig-emd-vs-mlr-6m.pdf` | `877d00a4f427228f48c3c9b36f2e6334d0a4ed5d05b50c2a977f9befedf5b0de` | 30,964 |
| `figures/paper/fig-emd-vs-mlr-12m.pdf` | `beda0599a0edf05a981272e67ca273e3dcd3ce7f1c3f009b10da5e811ff31520` | 31,081 |
| `figures/paper/fig-emd-vs-null-6m.pdf` | `3603c66e6c97f43773d1fa8bf34e497acf19e78f636840380e5e99aec6af526f` | 29,665 |
| `figures/paper/fig-emd-vs-null-12m.pdf` | `10958c4d14be6c75436f64b3168355a56d6c278c45bb1ca80433870f3e9534b8` | 30,138 |
| `figures/supplement/fig-emd-vs-slm-6m.pdf` | `22ab0720d44882397b32a0a356a9b4f6dfebad7d5846dbf0c1b9c1897ec0627f` | 30,615 |
| `figures/supplement/fig-emd-vs-slm-12m.pdf` | `04e78ea01dd3d88ba95eaf7d8fecc4776799075933187b7c947691d8b0a45f3a` | 30,670 |
| `figures/diagnostic/fig-emd-best-vs-baseline-6m.pdf` | `36af3418008fbd5b6b7afa6e059eddbe709fc53d8475c5a535017b89a6bf951d` | 28,885 |
| `figures/diagnostic/fig-emd-best-vs-baseline-12m.pdf` | `e28182144b04358ffd329d04cb723b239b8b5c449d54ed5740bce3f64f60a470` | 29,006 |

- The four 9-month PDF hashes remained byte-identical: MLR
  `da6155316edc6badc039966bd7558ba0279fb80f4d7092a9461c33d62d7de1c5`,
  null
  `fa0c733ab817ae6b05ff9d5390a093241519568f9bd4f5f2decc4ba63571e22a`,
  SLM
  `6b9cf292b3ccc6ff7114c6207e63ffd0625801b75b40da9cb90404deefae7782`,
  and best-versus-baseline
  `ee9c3ca402f156b517108bf388471958a75e89291461adec4c8a973766e356ad`.
- Full discovery passed all 162 tests under system Python with 17 documented
  optional-dependency skips, and all 162 tests in the scientific plotting
  environment without skips. The only emitted warning was the pre-existing
  legacy SciPy KDE deprecation.

### 2026-10-13 — `EMD-CV-LOWER-BOUND-HORIZON-CORRECTION-001`

- Preserved the completed 12-file delivery byte-for-byte under
  `archive/emd_cv_2026-10-13/original/tmp/` and verified every file against the
  archive's `SHA256SUMS`. This one snapshot now supplies all authoritative 6-,
  9-, and 12-month cross-validation rows.
- Compared the release cell-by-cell with the September 30 9-month and October
  10 6-/12-month predecessors. Evaluation identities, dates, `Lower Bound`,
  `Null`, `SLM`, `MLR`, and `GenModel (baseline)` are exact matches. Only `best
  GenModel` and `CV parameter` change, and they change together: all 97 H3N2/HA
  12-month rows, 13 H1N1/NA 6-month rows, and 10 H1N1/NA 12-month rows. The
  other nine source files are byte-identical.
- The correction fixes which lower-bound horizon was used during upstream
  hyperparameter selection; it does not rewrite the reported `Lower Bound`
  column. Across all 1,164 authoritative rows, `Lower Bound` is less than or
  equal to every model EMD. The former H3N2/HA 12-month exception at observation
  date `2013-07-01` now has `best GenModel = 4.1554496544696775` with
  `CV parameter = gamma=250, f=1e-3`, while its reported lower bound remains
  `2.8464822104905676`.
- Defined canonical release `emd_v6` and transform `emd_migration_v6`. Its 76
  source records comprise 60 legacy EMD files, four September 24 provenance
  files, and the 12 October 13 authoritative files. The September 30 and
  October 10 snapshots are immutable validation-only predecessors and are not
  duplicated in `sources.csv`.
- The changed best-model values affect the maintained MLR, null, SLM, and
  best-versus-baseline figures at 6 and 12 months. All four 9-month source files
  and therefore all four 9-month PDFs retain their existing numerical content.

#### Materialization and figure verification

- Materialized `emd_v6` with 2,591 evaluations, 16,578 values, 10 methods, and
  76 source records. An independent clean rebuild reproduced all four canonical
  files byte-for-byte. Focused EMD migration and plotting tests passed.

| Canonical file | SHA-256 | Bytes |
|---|---|---:|
| `data/plotting/inputs/emd/evaluations.csv` | `339c0111b221ae558abb4085af4d166d45ae1a4c86e2c58d79a404717487d14b` | 252,124 |
| `data/plotting/inputs/emd/values.csv` | `69d2c3e99f4daeea014fa0a9ee2a6123ebb3b7df50fc2d23c25d084e0a09c1b3` | 2,470,291 |
| `data/plotting/inputs/emd/methods.csv` | `27bdcb2e64fd3bfb7663a42aec33be76e05eca7126adb3098c39efc2ba5dc7de` | 1,835 |
| `data/plotting/inputs/emd/sources.csv` | `8ef452f8b31e3ef119934dfd125dcebd3727a62796ad291bb7c85b0135d27243` | 29,901 |

- Regenerated the four maintained 6-month and four maintained 12-month PDFs.
  Each is a one-page artifact on its registered paper canvas. Full-page visual
  inspection found the text, guide lines, trajectories, histograms, and
  significance annotations clean and unclipped.

| Figure | SHA-256 | Bytes |
|---|---|---:|
| `figures/paper/fig-emd-vs-mlr-6m.pdf` | `eb2a38113cd9d97b10324a5fbb5c56ec31889557cbd5476a1d39ff67366a04dd` | 27,945 |
| `figures/paper/fig-emd-vs-mlr-12m.pdf` | `1fe36ae62fe685a2c39dd50c766573026f5398a8c59340912a0703d3b8eef0bf` | 27,930 |
| `figures/paper/fig-emd-vs-null-6m.pdf` | `3a357022484a41f2e4959d958eaea0d0a82d33e351409298a70298b36ebb75fe` | 26,654 |
| `figures/paper/fig-emd-vs-null-12m.pdf` | `756a57e68dcb90a36a341aec7289881f50a37576ac6401a14a782c5844ebd6cf` | 26,618 |
| `figures/supplement/fig-emd-vs-slm-6m.pdf` | `c3f1bf8cb3eecb561424a1095bf3cce7db6dae5862f2079c56b7602242fa2e99` | 27,572 |
| `figures/supplement/fig-emd-vs-slm-12m.pdf` | `40b9ec979e8d7562b68666872a15d16064c7dbf8f43c59346c3765684befae51` | 27,638 |
| `figures/diagnostic/fig-emd-best-vs-baseline-6m.pdf` | `03e4a3aceacc4013c630a7c80e16abfc3baf4868e449d5199b92a95c0168b4a7` | 25,884 |
| `figures/diagnostic/fig-emd-best-vs-baseline-12m.pdf` | `f9f4249c0644a41e97fe7e1caae06265694af560307deff5242248016cac6777` | 25,995 |

### 2026-10-13 — `EMD-DATE-AXIS-PADDING-001`

- Added a symmetric 50-day display margin before the first January tick and
  after the final January tick in every maintained EMD trajectory panel. The
  eight tick dates, plotted observations, warmup cutoffs, statistics, vertical
  scales, typography, and physical canvas sizes are unchanged.
- Regenerated all 12 EMD comparison PDFs so paper, supplement, and diagnostic
  artifacts use the same renderer revision. Full-page raster inspection found
  all trajectories, guide lines, labels, histograms, and annotations clean and
  unclipped.
- Recorded one current `emd_v6` verification record per horizon and comparison
  in `figure_manifest.yaml`; the renderer-only revision is
  `symmetric_date_axis_padding_50d`.
- Manifest path and artifact-hash checks passed, as did all 169 repository
  tests in the scientific plotting environment.

| Figure | SHA-256 | Bytes |
|---|---|---:|
| `figures/supplement/fig-emd-vs-mlr-6m.pdf` | `2b28c29a710888ea9c3f06ff6e6b1b3f6741403e8b105739780f4c09c0c9085d` | 30,963 |
| `figures/supplement/fig-emd-vs-mlr-9m.pdf` | `7dbd6ccfe00809cb601e9e3c884e7a0aeb61dc7fbf49b458bb9cb94ba433e68f` | 30,936 |
| `figures/supplement/fig-emd-vs-mlr-12m.pdf` | `62a73e4617449d005e28cfab5b20f7521717f97879680541e9ad2fa35559b24d` | 30,942 |
| `figures/supplement/fig-emd-vs-null-6m.pdf` | `fc425a89b5e7beb15ab44362f6c4010f68bcd810dea8b3fc4f6b6b26e22c1175` | 29,666 |
| `figures/paper/fig-emd-vs-null-9m.pdf` | `f678cdcbcb027f201d6b0d0480ebba586a404c23d38a8636907928795307d27a` | 29,101 |
| `figures/supplement/fig-emd-vs-null-12m.pdf` | `a1a0030d6340bcc5a036ff8d868f71aafa84d7dd60972c7d3e4e98b1645d67c2` | 29,660 |
| `figures/diagnostic/fig-emd-vs-slm-6m.pdf` | `a21ba32d45c5f857a44fe35e1fa98f644ade1d2cdc91b1d84b511a68ef60b138` | 30,595 |
| `figures/diagnostic/fig-emd-vs-slm-9m.pdf` | `067ebe1dd531bea9f221e12862962e63e154bfe97033f95c5736823c790c2a9c` | 30,629 |
| `figures/supplement/fig-emd-vs-slm-12m.pdf` | `6e4b0364ff65ab9f1e417e581cae868e7baa32a9f096b481d5006fbc43151b07` | 30,630 |
| `figures/diagnostic/fig-emd-best-vs-baseline-6m.pdf` | `3b1c54de952954f695214df99dce199ceb93d112098d378861344deab417c9c0` | 28,895 |
| `figures/diagnostic/fig-emd-best-vs-baseline-9m.pdf` | `74b9d0e26999aab59a9c96373bae322ef5d1f8b7cde00e8b4b857ddde32f7c89` | 28,945 |
| `figures/diagnostic/fig-emd-best-vs-baseline-12m.pdf` | `0393ced776cc8d3adf4084e85b6ae1c34d30a6bdd55ef7dffbe09645b95e0ed6` | 29,014 |

### 2026-10-13 — `MANUSCRIPT-FIGURE-MIGRATION-001`

- Reconciled the final PDFs supplied in `tmp/` with the canonical registry,
  manifest, focused renderers, and scientist-facing notebook. The supplied set
  covers Main Figures 1–5 and Supplementary Figures S5–S16; the user-specified
  continuation assigns S17, S18, and S19 to the 6-, 9-, and 12-month
  full-model-versus-SLM comparisons.
- Promoted the canonical selection, fixation, background/clade, and EMD
  renderers for the 19 data-driven outputs. Each final label has a direct root
  `figures.py` call, declared inputs, a registered canvas, and a unique
  paper/supplement output path. No new scientific analysis was introduced.
- Recorded Main Figure 1 as the sole editable-artwork exception. Its source is
  `data/plotting/inputs/artwork/fig-model-overview.pptx`; the PDF is a manual
  single-slide export because generic office conversion did not preserve all
  trajectory artwork.
- Superseded the provisional diagnostic classification of the 6- and 9-month
  SLM comparisons and the unrelated provisional S18/S19 placeholders. The
  best-GenModel-versus-baseline family remains diagnostic.
- Updated the EMD annotation contract. Nonsignificant paired-test p-values are
  not drawn; values in `(0.01, 0.05]` are rounded to two decimals; values at or
  below 0.01 use the historical order notation. A significant negative mean
  `comparator EMD - target EMD` receives `*`, with the manuscript key
  `* full model worse`.
- Regenerated all 20 mapped outputs and marked the adapted families
  `verified_clean`. The manifest now records the current PDF hashes, byte
  counts, one-page canvas sizes, renderer revisions, and visual-QA result. All
  pages were rendered to images and inspected after generation, and the full
  202-test suite passed. Older verification hashes remain identified by their
  earlier renderer revisions.

### 2026-10-14 — `BACKGROUND-MUTATION-SAMPLING-001`

- This record documents the initial restored convention; its filter ordering is
  superseded by `BACKGROUND-MUTATION-SAMPLING-002` below.
- Compared the reorganized Main Figure 5 and Figures S13–S15 with the archived
  plotting code and references. The reorganized preparation had interpreted
  changing source time tuples as independent samples, so sliding
  observations of one mutation were counted repeatedly.
- Restored one plotting contribution per `(subtype, segment, position, state)`
  mutation identity. Preparation now selects the source record with the
  earliest `start_time` and retains that record's times, outcome, and fitness
  quantities. This reproduces the intended legacy first-record behavior without
  depending on CSV row order.
- The duration cutoff is applied only after mutation-level selection and remains
  inclusive. Figure S13 has no duration cutoff and contains 418 records (45
  fixed, 373 lost). Main Figure 5 and Figures S14–S15 require at least 90 days
  and contain 264 records (45 fixed, 219 lost).
- Upstream classifications remain authoritative: plotting does not reconstruct
  emergence, fixation, or loss from the sparse observation rows. Documentation,
  manifest transformations, and the scientist-facing notebook recipe text now
  state the sampling order explicitly.

### 2026-10-15 — `BACKGROUND-MUTATION-SAMPLING-002`

- Revised the order of the 90-day selection rule after reviewing mutations
  whose first source interval was short but whose later interval met the
  persistence criterion. The inclusive duration filter now acts on source rows
  first; preparation then chooses the earliest eligible `start_time` for each
  `(subtype, segment, position, state)` identity.
- Figure S13 remains unchanged because it has no duration cutoff and therefore
  still selects the earliest source row overall. Main Figure 5 and Figures
  S14–S15 use the revised 90-day rule and contain 315 records (54 fixed, 261
  lost), 51 more identities than the superseded filter-after-selection rule.
- Source times, outcomes, and fitness quantities from the selected eligible row
  remain authoritative. The plotting layer still does not merge observation
  intervals or reconstruct terminal outcomes.
- Regenerated Main Figure 5 and Figures S14–S15 as render revision
  `duration_first_sampling_v3`; Figure S13 was not rewritten. The notebook was
  re-executed only for those three affected recipes, and the manifest records
  their 2026-10-15 hashes, byte sizes, canvases, and visual-QA results.

### 2026-10-15 — `FIGURE-5-AXIS-COSMETICS-001`

- Refined Main Figure 5 without changing its scientific sample, summary
  statistics, clade preparation, or physical canvas. The visible left spine in
  each background-summary row now terminates at that row's outer labeled major
  ticks; the existing axis limits continue to provide point and annotation
  padding.
- Tightened Panel B's background-fitness-advantage limits from the inherited
  legacy range of -1% to +3.4% to -1.6% to +1.6%, with ticks at -1.5%, 0%, and
  +1.5%. The retained mutation points span -1.5167% to +1.2290%, so the revised
  scale displays every point and restores the previously clipped low N1 lost
  observation. The older asymmetric range belonged to a differently defined,
  historically mislabeled quantity and is not appropriate for this metric.
- Kept the Supplementary Figure S14 histogram limits unchanged. Added renderer
  tests for the spine bounds, unchanged padded data limits, complete Panel B
  point inclusion, and isolation from the supplementary histogram renderer.
- Regenerated only Main Figure 5 as render revision
  `figure5_axis_cosmetics_v4`; its final artifact metadata and visual-QA result
  are recorded in the manifest.

### 2026-10-15 — `SELECTION-AXIS-PADDING-001`

- Restored the distinction between the outer labeled tick and the padded
  display limit in Main Figure 2's sequence panels and Supplementary Figure
  S6. The canonical renderer had collapsed these values together during the
  migration. The archived helper instead nudged the data envelope above the
  ticks; for Figure 2D its data-driven limit was approximately 0.26024.
- Kept the established selection ticks and visible spine bounds at 0.4 for HA
  and 0.2 for NA while setting the corresponding limits to 0.42 and 0.26.
  Before correction, Figure 2D and S6 clipped five N2 positions and ten N1
  positions above 0.2; the largest retained NA value is 0.24785. The extra
  room also separates Figure 2's panel-D label from the 0.2 tick label.
- Extended Figure S6's SASA limit to 100 while retaining 80 as its outer tick
  and right-spine endpoint. This also restores seven H1 SASA positions that
  exceeded the former limit of 80; the maximum is 93.69.
- Added renderer tests requiring every selection and SASA curve to remain
  inside its display limits, requiring each limit to extend above its outer
  tick, and preserving tick-bounded visible spines. No data transformation,
  statistic, physical canvas, or non-axis styling changed.
- Regenerated Main Figure 2 and Supplementary Figure S6 as render revision
  `selection_axis_padding_v2`; final hashes, byte sizes, canvases, and visual
  QA are recorded in the manifest.

### 2026-10-15 — `HA-DOMAIN-ANNOTATION-001`

- Removed the overlap between the HA1 span indicator and the H3/H1 labels in
  Main Figure 2 and Supplementary Figure S6. The shared HA1/HA2 span lines move
  from selection value 0.37 to 0.39, and their domain labels move from 0.40 to
  0.42, preserving the line-to-label spacing and leaving the protein labels and
  data curves unchanged.
- Added renderer checks for the exact shared coordinates and for visible
  separation between the span lines, protein labels, and HA1/HA2 text. No data,
  statistics, axis limits, ticks, spines, canvas dimensions, or other styling
  changed.
- Regenerated Main Figure 2 and Supplementary Figure S6 as render revision
  `ha_domain_annotation_clearance_v3`; final hashes, byte sizes, canvases, and
  visual-QA results are recorded in the manifest.

## Archive checklist for the implementation phase

- [x] Create `archive/pre_plot_reorg_2026-08-26/original/`.
- [x] Extend ignore rules before relocating private GISAID material.
- [x] Preserve the original directory structure.
- [x] Save `SHA256SUMS`, file inventory, schema inventory, and path map.
- [x] Save current Git commit, status, staged diff, unstaged diff, and untracked-file list.
- [x] Copy the original notebook, `figures.py`, `mplot.py`, root plotting inputs,
      and generated artifacts.
- [x] Verify archive hashes before changing a working path.
- [x] Keep archived inputs immutable after verification. The archived
      exploratory notebook is an acknowledged exception for user reruns; its
      ordered cell types and source text are fingerprinted instead.
