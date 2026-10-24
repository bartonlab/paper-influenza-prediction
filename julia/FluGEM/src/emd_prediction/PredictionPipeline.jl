"""
    PredictionPipeline

Extended prediction pipeline built on top of the existing
`mutation_simulation.jl` machinery. This file does NOT modify the
original `run_prediction_pipeline`; it adds:

  * `run_prediction_pipeline_ext`     -- full version with SLM support
                                         and explicit h_corr / h_nocorr
                                         outputs.
  * `run_prediction_pipeline_short_ext` -- thin wrapper returning the
                                          distance/Hamming summary plus
                                          new SLM fields.
  * `distance_summary_table_short_ext` -- summary DataFrame including
                                          the SLM row and the selection
                                          coefficients.

Step A in the staged rollout: SLM model + selection coefficients with /
without correlations. Subsequent steps will add clade tracking,
CSV export, and a configuration layer.

Usage (drop-in once you've already loaded mutation_simulation.jl):

    include("mutation_simulation.jl")
    include("PredictionPipeline.jl")
    using .PredictionPipeline

    res = run_prediction_pipeline_short_ext(
        data, date_, Δ, "H3N2";
        slm = true,
        K_slm = 100,
        correlation = true,
    )
    distance_summary_table_short_ext(res)
"""
module PredictionPipeline

using Dates
using DataFrames
using DelimitedFiles
using CSV
using TOML
using Distributions: Multinomial    # for the SLM forward-simulation step

# We rely on functions defined in mutation_simulation.jl and its
# transitively-included files (MAPestimate, MLRmodel, MLR helpers,
# sequence_distribution, select_distribution_subset, EMD_HiGHS,
# consensus_sequence, unique_sequences, etc.) being available in `Main`
# when this module is loaded. We forward to them via the qualified
# `Main.` prefix to make that dependency explicit without re-including
# the source files (which would duplicate definitions).
#
# `single_locus_sequences_WF_Evolution` lives in SLM_mainfunctions.jl,
# which we do NOT include from the driver (it would re-define dozens of
# functions and trigger redefinition warnings against mutation_simulation.jl).
# Instead we inline the SLM-specific functions below and call
# `Main.unique_sequences` at the end.

export run_prediction_pipeline_ext,
       run_prediction_pipeline_short_ext,
       run_prediction_pipeline_ext_denoised,
       run_prediction_pipeline_short_ext_denoised,
       distance_summary_table_short_ext,
       save_pipeline_run,
       compute_clade_trajectories,
       SLMResult,
       PredictionConfig,
       load_or_create_study,
       save_config,
       load_config


# ===========================================================================
# Single-locus species + WF helpers (extracted from SLM_mainfunctions.jl)
# ===========================================================================

mutable struct slocSpecies
    n::Float64    # count / frequency
    s::Float64    # state, 1..q
    f::Float64    # mutation fitness
    i::Int64      # position
end

function init_slocSpecies(position::Int64,
                          config::Float64,
                          effectiveCount::Float64,
                          field::Array{Float64,1};
                          q::Int = 21)
    if config == 0.0
        rand_state = rand(1.0:q)
        return slocSpecies(effectiveCount, rand_state,
                           1 + field[(position - 1) * q + Int(rand_state)],
                           position)
    else
        return slocSpecies(effectiveCount, config,
                           1 + field[(position - 1) * q + Int(config)],
                           position)
    end
end

function initial_population_unique(sVec_i::Matrix{Float64},
                                   effectiveCounts::Array{Float64,1},
                                   q::Int64,
                                   h::Array{Float64,1},
                                   position::Int64)
    pop_dict = Dict{Tuple{Float64,Float64}, Float64}()
    (M_, _) = size(sVec_i)
    for i in 1:M_
        species = init_slocSpecies(position, sVec_i[i, position],
                                   effectiveCounts[i], h; q = q)
        key = (species.s, species.f)
        pop_dict[key] = get(pop_dict, key, 0.0) + species.n
    end
    pop = [slocSpecies(n, s, f, position) for ((s, f), n) in pop_dict]
    return pop
end

"""
    single_locus_sequences_WF_Evolution(u_sequences, u_counts, h_0;
                                        K=1000, forecast_time=360, q=21)

Independent-sites Wright-Fisher simulation: evolve every position
separately under the per-site fitnesses in `h_0`, then reconstruct
sequence frequencies as products of per-site state probabilities.
Returns the unique sequences and their normalised expected counts
averaged over `K` independent runs.

This is a verbatim port of the SLM_mainfunctions.jl version (kept
self-contained so PredictionPipeline doesn't have to include that file).
"""
function single_locus_sequences_WF_Evolution(u_sequences::Matrix{Float64},
                                              u_counts::Vector{Float64},
                                              h_0::Vector{Float64};
                                              K::Int = 1000,
                                              forecast_time::Int = 360,
                                              q::Int = 21)
    MM, L = size(u_sequences)
    N = 1000

    # Haplotype weights are a product of L per-position allele frequencies.
    # Accumulated directly, those L=566 sub-unit factors collapse past
    # Float64's floor long before the biology says the haplotype is gone,
    # driving every weight to exactly 0 -- which `unique_sequences` then
    # normalises by sum=0, giving Inf*0 = NaN. That was the "degenerate
    # sample" `_slm_one_sample` had to replace with the initial population.
    #
    # So accumulate Σ log(fr) instead of ∏ fr, via a per-haplotype streaming
    # log-sum-exp: `max_logw[m]` is the largest log-weight seen for
    # haplotype m so far, `acc[m]` the sum of exp(logw - max_logw[m]) over
    # replicates. This keeps the replicate-to-replicate weighting exactly as
    # the old product accumulation had it (replicates are NOT individually
    # renormalised, so a replicate with more mass on observed haplotypes
    # still counts for more) while never materialising the product itself.
    max_logw = fill(-Inf, MM)
    acc      = zeros(MM)

    for _ in 1:K
        list_species = [Vector{slocSpecies}() for _ in 1:L]

        for pos in 1:L
            i_pop = initial_population_unique(u_sequences, u_counts, q, h_0, pos)
            t = 1
            while t < forecast_time + 1
                r = [s.n * s.f for s in i_pop]
                r = abs.(r)
                total_r = sum(r)
                p = total_r > 0 ? r / total_r : fill(1 / length(r), length(r))
                if any(isnan, p) || sum(p) == 0
                    p .= 1.0 / length(p)
                end
                n_new = rand(Multinomial(N, p)) ./ N
                for i in 1:length(i_pop)
                    i_pop[i].n = n_new[i]
                end
                t += 1
            end
            list_species[pos] = i_pop
        end

        for m in 1:MM
            s_1 = @view u_sequences[m, :]
            logw = 0.0
            for i in 1:L
                @inbounds st = s_1[i]
                indices = findall(x -> x.s == st, list_species[i])
                fr = !isempty(indices) ? list_species[i][indices[1]].n : 0.0
                if fr <= 0.0
                    # Allele genuinely lost at this position (or absent from
                    # the species list): the haplotype is impossible in this
                    # replicate. That is a real zero, not an underflow, so
                    # stop early -- log(0) would be -Inf anyway.
                    logw = -Inf
                    break
                end
                logw += log(fr)
            end

            # -Inf contributes exp(-Inf) = 0 to the replicate sum: skip it.
            isfinite(logw) || continue
            if logw > max_logw[m]
                # New largest term: rescale the running sum onto the new max.
                acc[m] = acc[m] * exp(max_logw[m] - logw) + 1.0
                max_logw[m] = logw
            else
                acc[m] += exp(logw - max_logw[m])
            end
        end
    end

    # Back out of log space with ONE shift shared by every haplotype, so the
    # ratios between haplotypes stay exact while the arbitrary overall scale
    # -- which `unique_sequences` normalises away regardless -- cannot
    # underflow. With no underflow present this reproduces the old
    # normalised weights exactly.
    logtotal = [acc[m] > 0 ? max_logw[m] + log(acc[m]) : -Inf for m in 1:MM]
    shift    = maximum(logtotal)
    prop_counts_av = isfinite(shift) ?
        [isfinite(lt) ? exp(lt - shift) : 0.0 for lt in logtotal] :
        # Every haplotype lost an allele in every replicate: a genuine
        # degeneracy, not an artefact. Return zeros and let the caller's
        # fallback in `_slm_one_sample` handle it.
        zeros(MM)

    return Main.unique_sequences(u_sequences, prop_counts_av)
end


# ===========================================================================
# SLM result container
# ===========================================================================

"""
    SLMResult

Outputs from the single-locus (SLM) prediction path. Mirrors what
`simulate_medoid_distribution` returns for GenModel: a medoid
distribution selected from `M` SLM sample distributions, plus the
samples and their pairwise EMD distances.

* `h`                 -- selection coefficients used (from MAPestimate with
                         `correlation=false`).
* `output`            -- medoid SLM sequence distribution at t2 (after
                         `select_distribution_subset(freq_threshold)`).
* `output_corrected`  -- same, after `correct_weird_mutations`.
                         `nothing` when `full_frequency` was not supplied.
* `cons`              -- consensus sequence of `output`.
* `cons_corrected`    -- consensus of `output_corrected`, or `nothing`.
* `emd`               -- full `EMD_HiGHS` result (NamedTuple with fields
                         `:emd`, `:flow`, `:status`) between
                         `pop_result.d2` and `output`. Access the scalar
                         distance as `slm.emd.emd`. (Same convention as
                         `result.naive`, `result.prediction`, etc.)
* `emd_corrected`     -- same shape, between corrected d2 and
                         `output_corrected`. `nothing` when
                         `full_frequency` was not supplied.
* `ham`               -- Hamming distance between consensus(`pop_result.d2`)
                         and `cons`.
* `ham_corrected`     -- corrected variant, or `nothing`.
* `output_average`    -- average (not medoid) of the same `M` SLM samples,
                         via `average_distribution` -- no extra simulation
                         cost.
* `output_average_corrected` -- same, after `correct_weird_mutations`.
                         `nothing` when `full_frequency` was not supplied.
* `cons_average`      -- consensus sequence of `output_average`.
* `cons_average_corrected` -- consensus of `output_average_corrected`, or
                         `nothing`.
* `emd_average`       -- `EMD_HiGHS` result between `pop_result.d2` and
                         `output_average`.
* `emd_average_corrected` -- same shape, corrected variant, or `nothing`.
* `ham_average`       -- Hamming distance between consensus(`pop_result.d2`)
                         and `cons_average`.
* `ham_average_corrected` -- corrected variant, or `nothing`.
* `medoid_index`      -- index (1..M) of the chosen medoid in `samples`.
* `samples`           -- the `M` SLM sample distributions used for the
                         medoid selection.
* `pairwise_distances`-- `M x M` matrix of EMDs between samples.
* `medoid_scores`     -- per-sample medoid score (sum of EMDs to others).
"""
struct SLMResult
    h::Vector{Float64}
    output::Any
    output_corrected::Any
    cons::Any
    cons_corrected::Any
    emd::Any
    emd_corrected::Any
    ham::Any
    ham_corrected::Any
    output_average::Any
    output_average_corrected::Any
    cons_average::Any
    cons_average_corrected::Any
    emd_average::Any
    emd_average_corrected::Any
    ham_average::Any
    ham_average_corrected::Any
    medoid_index::Any
    samples::Any
    pairwise_distances::Any
    medoid_scores::Any
end


# ===========================================================================
# SLM forward simulation: medoid over M samples
# ===========================================================================

"""
    _sanitize_distribution(d, threshold)

Make `d::Dict{BioSequence,Float64}` safe for downstream
`select_distribution_subset` calls.

  * drops NaN / Inf / negative entries,
  * if at least one entry exceeds `threshold`, returns the
    threshold-filtered & renormalised dict,
  * otherwise (everything below threshold), renormalises the surviving
    finite-positive entries without filtering -- preserving small but
    real mass so the medoid step can still compute EMDs.

The pre-existing `Main.select_distribution_subset` indexes
`new_seqs[end]` after the filter, which crashes if the filter empties
the dict. This wrapper guarantees the returned dict is non-empty,
finite, positive, and sums to 1, so subsequent calls to
`select_distribution_subset(d, t)` with `t <= threshold` are safe.
"""
function _sanitize_distribution(d::AbstractDict, threshold::Real)
    isempty(d) && return d

    # Step 1: drop NaN, Inf, and non-positive values. Build the result
    # with the SAME abstract key type as the input so downstream
    # `select_distribution_subset(::Dict{BioSequence,Float64}, ...)`
    # method dispatch still matches. A naive `Dict(...)` comprehension
    # would narrow to the concrete runtime key type (e.g. LongAA) and
    # break dispatch.
    finite = Dict{keytype(d), valtype(d)}()
    for (k, v) in d
        if isfinite(v) && v > 0
            finite[k] = v
        end
    end
    isempty(finite) && return finite       # truly empty -> can't help further

    # Step 2: count how many entries pass the threshold.
    above = count(v -> v >= threshold, values(finite))

    # Step 3: keep entries above threshold if there are any, otherwise
    # keep everything finite-positive. Renormalise either way.
    # IMPORTANT: use filter! on `finite` (which has the abstract key
    # type) rather than `Dict(k => v for ...)` (which would narrow).
    kept = finite
    if above > 0
        filter!(kv -> kv.second >= threshold, kept)
    end
    s = sum(values(kept))
    s == 0 && return kept                   # paranoid; shouldn't happen
    for k in keys(kept); kept[k] /= s; end
    return kept
end

"""
    _safe_select_subset(d, thr) -> Dict

Apply `Main.select_distribution_subset(d, thr)` but skip the filtering
when it would empty the dict. `select_distribution_subset` has a
pre-existing bug where an all-below-threshold input triggers a
`BoundsError` (it indexes `new_seqs[end]` after the filter removed
everything). For trajectory work this manifests when individual SLM
samples are noisy enough that every species sits below `thr`.

NaN values are treated as missing.

The fallback (return the sanitised dict) preserves enough mass for
the medoid-by-EMD step to function; the final filtering will be
applied to the medoid output once.
"""
function _safe_select_subset(d::AbstractDict, thr::Real)
    return _sanitize_distribution(d, thr)
end

"""
    _slm_one_sample(init_seqs, init_counts, h_nocorr, T_gen;
                    K_slm, q, freq_threshold, mut_pos, sigma_h, perturb_h)

Produce ONE SLM sample distribution: a single call to
`single_locus_sequences_WF_Evolution` with `K = K_slm` -- i.e.
`K_slm` independent single-locus WF trajectories per position
averaged into one sample distribution. This is the SLM analogue of
`generate_sample_distribution` for GenModel.

When `perturb_h` is true, `h_nocorr` is first perturbed by a draw from
the posterior of the selection coefficients, mirroring what
`generate_sample_distribution` does for GenModel. The posterior here is
DIAGONAL (see `simulate_slm_medoid_distribution`), so the draw is just
`sigma_h[k] * randn()` per active position -- no matrix square root.

If the inner SLM run produces a degenerate distribution (e.g. all
species ended up with zero or NaN weight because drift at L=566
positions over a long `T_gen` killed every species' frequency), fall
back to the initial population distribution -- the "no evolution"
baseline -- so the medoid-of-M selection can still function.
"""
function _slm_one_sample(init_seqs::Matrix{Float64},
                          init_counts::Vector{Float64},
                          h_nocorr::Vector{Float64},
                          T_gen::Int;
                          K_slm::Int = 1,
                          q::Int = 21,
                          freq_threshold::Float64 = 0.0,
                          mut_pos::Vector{Int}      = Int[],
                          sigma_h::Vector{Float64}  = Float64[],
                          perturb_h::Bool           = false)
    # Posterior draw for h, the SLM analogue of the `GaussianVector`
    # perturbation in `generate_sample_distribution`. Only copy when we
    # actually perturb -- the no-noise path stays allocation-free.
    h_use = h_nocorr
    if perturb_h && !isempty(mut_pos)
        h_use = copy(h_nocorr)
        @inbounds for (k, p) in enumerate(mut_pos)
            h_use[p] += sigma_h[k] * randn()
        end
    end

    full_seqs, full_counts = single_locus_sequences_WF_Evolution(
        init_seqs, init_counts, h_use;
        K = K_slm, forecast_time = T_gen, q = q,
    )

    # Detect a degenerate sample: all-zero or all-NaN counts. The latter
    # happens when `unique_sequences` normalises by `sum(counts) = 0`,
    # producing `inv(0) = Inf` -> NaN. Falling back to the initial
    # population is a defensible "no evolution" sample that keeps the
    # medoid step well-defined.
    total = sum(c -> isnan(c) ? 0.0 : c, full_counts)
    if total == 0
        @warn "SLM sample degenerate (all-zero / all-NaN counts after " *
              "T_gen=$T_gen, K_slm=$K_slm). Falling back to the initial " *
              "population as this sample. Consider raising K_slm."
        # Use the initial population, properly normalised.
        s = sum(init_counts)
        full_seqs   = init_seqs
        full_counts = s > 0 ? init_counts ./ s : init_counts
    end

    d = Main.sequence_distribution(full_seqs, full_counts)
    return _safe_select_subset(d, freq_threshold)
end

"""
    simulate_slm_medoid_distribution(pop_result, h_nocorr;
                                     T_gen, M, K_slm, q, freq_threshold,
                                     medoid_threshold, verbose,
                                     noise, gamma, Ne, map_result) -> NamedTuple

Run `M` independent SLM forward simulations (each one averaging
`K_slm` independent single-locus WF trajectories per position) and
return the medoid distribution together with pairwise EMD distances.
Mirrors the return shape of `Main.simulate_medoid_distribution`:

    (output, medoid_index, samples, pairwise_distances, medoid_scores)

`T_gen` defaults to `pop_result.t2 - pop_result.t1` (the full forecast
horizon), but can be set to a smaller value when sampling at
intermediate snapshot times during a trajectory build.

Cost: `M * K_slm` single-locus WF trajectories per position, each over
`T_gen` generations. For HA (L=566) this can be slow if both `M` and
`K_slm` are large; `K_slm = 1` is the cheap configuration that matches
how GenModel runs `generate_sample_distribution` with K = 1.

With `noise = true` each of the `M` samples first draws its own `h` from
the posterior, the SLM analogue of GenModel's `sq_isC` perturbation.
That requires `map_result` -- the `correlation=false` MAPestimate that
produced `h_nocorr` -- since the spread is built from its `C_int`. `Ne`
and `gamma` set the width; see the derivation in the function body.
With `noise = false` (default) every sample reuses `h_nocorr` unchanged
and the samples differ only through WF drift, exactly as before.
"""
function simulate_slm_medoid_distribution(pop_result::NamedTuple,
                                           h_nocorr::Vector{Float64};
                                           T_gen::Int = pop_result.t2 - pop_result.t1,
                                           M::Int                    = 50,
                                           K_slm::Int                = 1,
                                           q::Int                    = 21,
                                           freq_threshold::Float64   = 0.0,
                                           medoid_threshold::Float64 = 1.0e-6,
                                           verbose::Bool             = true,
                                           noise::Bool               = false,
                                           gamma::Float64            = 20.0,
                                           Ne::Int                   = 500,
                                           map_result::Union{NamedTuple,Nothing} = nothing)
    init_seqs, init_counts = pop_result.pop1
    verbose && @info "SLM medoid: $M samples (K_slm=$K_slm per sample), " *
                     "T_gen=$T_gen, " *
                     "$(size(init_seqs, 1)) initial unique sequences."

    # --- Posterior spread for h (same scheme as GenModel) ------------------
    # `h_nocorr` comes from `processStandardNoCorr`, which is exactly
    # h_i = Δx_i / (C_int[i,i] + γ) -- the DIAGONAL estimator. So its
    # posterior is the diagonal counterpart of GenModel's
    # Σ = (1/Ne)·(C_int + γI)⁻¹, i.e. per-position variances
    #
    #     σ_i = 1 / sqrt(Ne · (C_int[i,i] + γ)),
    #
    # with the same γ used for the point estimate and the same Ne, so the
    # prior width matches GenModel's: σ_i → 1/sqrt(Ne·γ) = 0.01 at Ne=500,
    # γ=20 in the no-data limit. Using GenModel's FULL correlated Σ here
    # would be inconsistent -- it is not the posterior of this estimator.
    #
    # `Cintegrate` already adds γ to its diagonal (see `_cintegrate_core!`),
    # and its diagonal accumulation is term-for-term identical to the
    # `diag_cov` built inside `processStandardNoCorr`, so `Cint_[p,p]` is
    # already `C_int[p,p] + γ` -- no back-out needed (unlike the GenModel
    # path, which must reassemble the full matrix).
    mut_pos = Int[]
    sigma_h = Float64[]
    if noise
        map_result === nothing && throw(ArgumentError(
            "simulate_slm_medoid_distribution: noise=true needs `map_result` " *
            "(the correlation=false MAPestimate that produced `h_nocorr`) " *
            "to build the posterior spread."))
        verbose && @info "SLM: computing posterior spread for h (Ne=$Ne, gamma=$gamma)."
        L_     = size(map_result.seqs_observed, 2)
        Cint_  = Main.Cintegrate(map_result.seqs_observed,
                                  map_result.effectiveCounts_list,
                                  map_result.times_measure, L_, q, gamma)
        mut_pos = Main.find_active_positions(
            map_result.observed_path_freq, map_result.ref_s, q)
        sigma_h = [1.0 / sqrt(Ne * Cint_[p, p]) for p in mut_pos]
    end
    perturb_h = noise && !isempty(mut_pos)

    # Pre-build the "no evolution" fallback distribution (the initial
    # population, normalised). This is what we substitute whenever an
    # SLM sample comes back degenerate (empty, all-NaN, all-zero). It
    # keeps the medoid step well-defined and is a defensible
    # "evolution failed" sample.
    init_d = let s = sum(init_counts)
        d = Main.sequence_distribution(init_seqs,
                                        s > 0 ? init_counts ./ s : init_counts)
        _sanitize_distribution(d, medoid_threshold)
    end

    samples   = Vector{Any}(undef, M)
    fell_back = Vector{Bool}(undef, M)
    Threads.@threads for m in 1:M
        raw = _slm_one_sample(init_seqs, init_counts, h_nocorr, T_gen;
                              K_slm = K_slm, q = q,
                              freq_threshold = freq_threshold,
                              mut_pos = mut_pos, sigma_h = sigma_h,
                              perturb_h = perturb_h)
        # `medoid_by_emd` internally calls
        # `select_distribution_subset(s, medoid_threshold)` on every
        # sample, which crashes on the buggy `new_seqs[end]` indexing
        # if the filter empties the dict. Pre-sanitise here so that
        # the threshold filter inside medoid_by_emd always leaves at
        # least one entry. If sanitisation can't produce a usable
        # sample (truly empty), substitute the initial population.
        clean = _sanitize_distribution(raw, medoid_threshold)
        if isempty(clean)
            fell_back[m] = true
            samples[m]   = init_d
        else
            fell_back[m] = false
            samples[m]   = clean
        end
    end
    # Tally fallbacks after the threaded loop -- `n_fallback += 1` inside
    # the loop would be a data race across threads (lost updates), so each
    # iteration instead writes its own `fell_back[m]` slot and we count once
    # here.
    n_fallback = count(fell_back)
    if n_fallback > 0
        @warn "SLM medoid: $n_fallback / $M samples were degenerate and " *
              "replaced with the initial population. Consider raising K_slm."
    end

    medoid_index, output, D, scores = Main.medoid_by_emd(
        samples, medoid_threshold; verbose = verbose)

    return (
        output             = output,
        medoid_index       = medoid_index,
        samples            = samples,
        pairwise_distances = D,
        medoid_scores      = scores,
    )
end


# ===========================================================================
# Extended pipeline
# ===========================================================================

"""
    run_prediction_pipeline_ext(data, date_, Δ, subtype; slm=false, K_slm=100,
                                  kwargs...)

Drop-in superset of `run_prediction_pipeline` from `mutation_simulation.jl`.

When `slm = false` (default) the behaviour and outputs match the original
pipeline exactly, with these additional fields injected into the
returned NamedTuple regardless of the slm flag:

  * `h_correlation`    -- `map_result.h` from MAPestimate with correlation = true
  * `h_no_correlation` -- `map_result.h` from MAPestimate with correlation = false
  * `slm_result`       -- an `SLMResult` if `slm = true`, otherwise `nothing`

`K_slm` controls the number of independent SLM runs (smaller than `K`
because `single_locus_sequences_WF_Evolution` is sequential over
positions and generations).
"""
function run_prediction_pipeline_ext(
    data::Matrix{Float64},
    date_::Date,
    Δ::Int,
    subtype::String;
    # SLM-specific kwargs
    slm::Bool                 = false,
    K_slm::Int                = 100,
    # original pipeline kwargs (unchanged defaults)
    gamma::Float64            = 20.0,
    q::Int                    = 21,
    tstep::Float64            = 10.0,
    tsmooth::Float64          = 60.0,
    noise::Bool               = false,
    N::Int                    = 1000,
    # Posterior spread scale for the h-sampling (noise=true). Matches the
    # `Ne` default in `simulate_medoid_distribution`, so GenModel and SLM
    # use the same prior width: σ = 1/sqrt(Ne·γ) = 0.01 at Ne=500, γ=20.
    Ne::Int                   = 500,
    K::Int                    = 1000,
    M::Int                    = 100,
    freq_threshold::Float64   = 1.0e-10,
    medoid_threshold::Float64 = 1.0e-6,
    correlation::Bool         = true,
    verbose::Bool             = true,
    full_frequency::Union{Matrix{Float64},Nothing} = nothing,
    user_start_date::Union{Date,Nothing}           = nothing,
)
    # --- Run the original pipeline -----------------------------------------
    res = Main.run_prediction_pipeline(
        data, date_, Δ, subtype;
        gamma=gamma, q=q, tstep=tstep, tsmooth=tsmooth,
        noise=noise, N=N, Ne=Ne, K=K, M=M,
        freq_threshold=freq_threshold,
        medoid_threshold=medoid_threshold,
        correlation=correlation, verbose=verbose,
        full_frequency=full_frequency,
        user_start_date=user_start_date,
    )

    # --- Compute selection coefficients in BOTH modes for reporting ---------
    # `res.map_result.h` already contains coefficients in the user-requested
    # mode; we always run the OTHER one too so both are reported.
    h_corr   = nothing
    h_nocorr = nothing
    # The MAPestimate result that produced `h_nocorr`. The SLM path needs it
    # (not just the h-vector) to build the posterior spread when noise=true.
    map_nocorr_result = nothing
    if correlation
        h_corr = res.map_result.h
        if slm
            # SLM needs the no-correlation h-vector; only compute it when needed.
            verbose && @info "Computing MAPestimate with correlation=false for SLM."
            map_nocorr = Main.MAPestimate(
                data, date_, subtype;
                tstep=tstep, gamma=gamma, tsmooth=tsmooth, q=q,
                correlation=false, user_start_date=user_start_date,
            )
            h_nocorr = map_nocorr.h
            map_nocorr_result = map_nocorr
        end
    else
        h_nocorr = res.map_result.h
        map_nocorr_result = res.map_result
        # When the user requested no correlation, we don't recompute the
        # correlated version unless asked (it's expensive and not used by
        # the SLM path). We still expose `h_correlation = nothing` so
        # downstream tooling has a stable schema.
    end

    # --- Run SLM forward simulation (medoid over M samples) -----------------
    slm_result = nothing
    if slm
        @assert h_nocorr !== nothing "SLM requires a correlation=false h-vector."

        slm_sim = simulate_slm_medoid_distribution(
            res.pop_result, h_nocorr;
            M = M, K_slm = K_slm, q = q,
            freq_threshold = freq_threshold,
            medoid_threshold = medoid_threshold,
            verbose = verbose,
            noise = noise, gamma = gamma, Ne = Ne,
            map_result = map_nocorr_result,
        )
        slm_output = slm_sim.output

        slm_cons = Main.consensus_sequence(slm_output)
        slm_emd  = Main.EMD_HiGHS(res.pop_result.d2, slm_output)
        slm_ham  = Main.BioTools.hamming(
            Main.consensus_sequence(res.pop_result.d2), slm_cons,
        )

        # Average across the same M samples used for the medoid (no extra
        # simulation cost -- see `average_distribution` in misc_EMD.jl).
        slm_output_average = Main.average_distribution(slm_sim.samples)
        slm_cons_average = Main.consensus_sequence(slm_output_average)
        slm_emd_average  = Main.EMD_HiGHS(res.pop_result.d2, slm_output_average)
        slm_ham_average  = Main.BioTools.hamming(
            Main.consensus_sequence(res.pop_result.d2), slm_cons_average,
        )

        # Corrected (filtered) variants -- only if full_frequency was given.
        slm_output_corrected = nothing
        slm_cons_corrected   = nothing
        slm_emd_corrected    = nothing
        slm_ham_corrected    = nothing
        slm_output_average_corrected = nothing
        slm_cons_average_corrected   = nothing
        slm_emd_average_corrected    = nothing
        slm_ham_average_corrected    = nothing
        if full_frequency !== nothing && res.output_corrected !== nothing
            # Re-derive the weird-mutation mask using the same recipe the
            # original pipeline used.
            L_corr   = size(res.map_result.seqs_observed, 2)
            weird_df = Main.find_weird_mutations(full_frequency, L_corr, q)
            slm_output_corrected, _ =
                Main.correct_weird_mutations(slm_output, weird_df)
            slm_cons_corrected = Main.consensus_sequence(slm_output_corrected)
            slm_emd_corrected  = Main.EMD_HiGHS(
                res.d2_corrected, slm_output_corrected,
            )
            slm_ham_corrected  = Main.BioTools.hamming(
                Main.consensus_sequence(res.d2_corrected), slm_cons_corrected,
            )

            slm_output_average_corrected, _ =
                Main.correct_weird_mutations(slm_output_average, weird_df)
            slm_cons_average_corrected = Main.consensus_sequence(slm_output_average_corrected)
            slm_emd_average_corrected  = Main.EMD_HiGHS(
                res.d2_corrected, slm_output_average_corrected,
            )
            slm_ham_average_corrected  = Main.BioTools.hamming(
                Main.consensus_sequence(res.d2_corrected), slm_cons_average_corrected,
            )
        end

        slm_result = SLMResult(
            h_nocorr,
            slm_output, slm_output_corrected,
            slm_cons, slm_cons_corrected,
            slm_emd, slm_emd_corrected,
            slm_ham, slm_ham_corrected,
            slm_output_average, slm_output_average_corrected,
            slm_cons_average, slm_cons_average_corrected,
            slm_emd_average, slm_emd_average_corrected,
            slm_ham_average, slm_ham_average_corrected,
            slm_sim.medoid_index,
            slm_sim.samples,
            slm_sim.pairwise_distances,
            slm_sim.medoid_scores,
        )
    end

    # --- Merge with original output as a NamedTuple -------------------------
    return merge(res, (
        h_correlation     = h_corr,
        h_no_correlation  = h_nocorr,
        slm_result        = slm_result,
    ))
end


# ===========================================================================
# Short wrapper
# ===========================================================================

"""
    run_prediction_pipeline_short_ext(data, date_, Δ, subtype; slm=false,
                                       K_slm=100, kwargs...)

Slim version that returns only the distances, distributions, Hamming
scores, selection coefficients, and (optional) SLM result needed for
reporting.
"""
function run_prediction_pipeline_short_ext(
    data::Matrix{Float64},
    date_::Date,
    Δ::Int,
    subtype::String;
    slm::Bool                 = false,
    K_slm::Int                = 100,
    gamma::Float64            = 20.0,
    q::Int                    = 21,
    tstep::Float64            = 10.0,
    tsmooth::Float64          = 60.0,
    noise::Bool               = false,
    N::Int                    = 1000,
    # Posterior spread scale for the h-sampling (noise=true). Matches the
    # `Ne` default in `simulate_medoid_distribution`, so GenModel and SLM
    # use the same prior width: σ = 1/sqrt(Ne·γ) = 0.01 at Ne=500, γ=20.
    Ne::Int                   = 500,
    K::Int                    = 1000,
    M::Int                    = 100,
    freq_threshold::Float64   = 1.0e-10,
    medoid_threshold::Float64 = 1.0e-6,
    correlation::Bool         = true,
    verbose::Bool             = true,
    full_frequency::Union{Matrix{Float64},Nothing} = nothing,
    user_start_date::Union{Date,Nothing}           = nothing,
)
    res = run_prediction_pipeline_ext(
        data, date_, Δ, subtype;
        slm=slm, K_slm=K_slm,
        gamma=gamma, q=q, tstep=tstep, tsmooth=tsmooth,
        noise=noise, N=N, Ne=Ne, K=K, M=M,
        freq_threshold=freq_threshold,
        medoid_threshold=medoid_threshold,
        correlation=correlation, verbose=verbose,
        full_frequency=full_frequency,
        user_start_date=user_start_date,
    )

    return (
        # Original short outputs
        naive                  = res.result_naive,
        lower_bound            = res.result_lb,
        prediction             = res.result_prediction,
        prediction_corrected   = res.result_prediction_corrected,
        average                = res.result_average,
        average_corrected      = res.result_average_corrected,
        naive_corrected        = res.result_naive_corrected,
        lower_bound_corrected  = res.result_lb_corrected,
        naive_emp              = res.result_naive_emp,
        naive_emp_corrected    = res.result_naive_emp_corrected,
        output                 = res.output,
        output_corrected       = res.output_corrected,
        output_average           = res.output_average,
        output_average_corrected = res.output_average_corrected,
        start_day_date         = res.start_day_date,
        date1                  = res.pop_result.date1,
        date2                  = res.pop_result.date2,
        ham_naive              = res.ham_naive,
        ham_prediction         = res.ham_prediction,
        ham_average            = res.ham_average,
        ham_lowerbound         = res.ham_lowerbound,
        ham_naive_emp          = res.ham_naive_emp,
        ham_naive_corrected      = res.ham_naive_corrected,
        ham_prediction_corrected = res.ham_prediction_corrected,
        ham_average_corrected    = res.ham_average_corrected,
        ham_lowerbound_corrected = res.ham_lowerbound_corrected,
        ham_naive_emp_corrected  = res.ham_naive_emp_corrected,
        mlr                    = res.result_mlr,
        mlr_corrected          = res.result_mlr_corrected,
        ham_mlr                = res.ham_mlr,
        ham_mlr_corrected      = res.ham_mlr_corrected,
        # NEW: selection coefficients in both modes
        h_correlation          = res.h_correlation,
        h_no_correlation       = res.h_no_correlation,
        # NEW: SLM result (nothing when slm=false)
        slm                    = res.slm_result,
        # NEW: distributions for CSV / plotting (Step B)
        d1                     = res.pop_result.d1,         # smoothed observation @ t1
        d2                     = res.pop_result.d2,         # smoothed truth @ t2
        dlb                    = res.pop_result.dlb,        # lower-bound prediction
        bin_d1                 = res.emp_result.bin_d1,     # empirical observation
        bin_d2                 = res.emp_result.bin_d2,     # empirical truth
        d_mlr                  = res.d_mlr,                 # MLR prediction
        # NEW: initial WF population for trajectory tracking (Step C)
        pop1                   = res.pop_result.pop1,
        # Day-numbers as they appear in `data[:, 1]`. These are the exact
        # values that `distributions_at_t1_t2` resolved to via
        # `closest_utime`, so any further slicing of `data` for trajectory
        # work must use these (not arithmetic on the dates).
        t1                     = res.pop_result.t1,
        t2                     = res.pop_result.t2,
        # Full inner results, exposed so that `compute_clade_trajectories`
        # can re-run the model at intermediate snapshot times:
        #   - mlr_result : evaluated at each snapshot via
        #                  predict_both_probs_counts
        #   - pop_result : used as the WF starting point
        #   - map_result : feeds h, noise, and other args into
        #                  simulate_medoid_distribution
        mlr_result             = res.mlr_result,
        pop_result             = res.pop_result,
        map_result             = res.map_result,
    )
end


# ===========================================================================
# run_prediction_pipeline_ext_denoised (causal / leakage-free denoising)
# ===========================================================================

"""
    run_prediction_pipeline_ext_denoised(data, date_, Δ, subtype; slm=false,
                                          K_slm=100, weird_threshold=0.01,
                                          kwargs...)

Causal / leakage-free a priori counterpart to `run_prediction_pipeline_ext`
-- see `Main.run_prediction_pipeline_denoised` for the full explanation.
Calls `Main.run_prediction_pipeline_denoised` (which builds `d1`/`pop1` from
data denoised using only information up to `t1`, and `d2` independently
using only information up to `t2`), then fits the no-correlation `h` needed
for SLM on the SAME t1-denoised data (`res.data_t1`) already used for the
correlated fit and `pop1` -- not on raw `data`. The SLM simulation is seeded
from the (already denoised) `res.pop_result`, so it's clean automatically.

There is nothing left to patch after the fact, so `_corrected` fields are
`nothing` throughout, same as `run_prediction_pipeline_denoised`.

Requires `FluStudy.jl` to be loaded (for `frequency_trajectory`).

Purely additive: does not modify `run_prediction_pipeline_ext` or change
its default behaviour.
"""
function run_prediction_pipeline_ext_denoised(
    data::Matrix{Float64},
    date_::Date,
    Δ::Int,
    subtype::String;
    slm::Bool                 = false,
    K_slm::Int                = 100,
    gamma::Float64            = 20.0,
    q::Int                    = 21,
    tstep::Float64            = 10.0,
    tsmooth::Float64          = 60.0,
    noise::Bool               = false,
    N::Int                    = 1000,
    # Posterior spread scale for the h-sampling (noise=true). Matches the
    # `Ne` default in `simulate_medoid_distribution`, so GenModel and SLM
    # use the same prior width: σ = 1/sqrt(Ne·γ) = 0.01 at Ne=500, γ=20.
    Ne::Int                   = 500,
    K::Int                    = 1000,
    M::Int                    = 100,
    freq_threshold::Float64   = 1.0e-10,
    medoid_threshold::Float64 = 1.0e-6,
    correlation::Bool         = true,
    verbose::Bool             = true,
    weird_threshold::Float64  = 0.01,
    user_start_date::Union{Date,Nothing}           = nothing,
)
    res = Main.run_prediction_pipeline_denoised(
        data, date_, Δ, subtype;
        gamma=gamma, q=q, tstep=tstep, tsmooth=tsmooth,
        noise=noise, N=N, Ne=Ne, K=K, M=M,
        freq_threshold=freq_threshold,
        medoid_threshold=medoid_threshold,
        correlation=correlation, verbose=verbose,
        weird_threshold=weird_threshold,
        user_start_date=user_start_date,
    )

    h_corr   = nothing
    h_nocorr = nothing
    # The MAPestimate result that produced `h_nocorr`. The SLM path needs it
    # (not just the h-vector) to build the posterior spread when noise=true.
    map_nocorr_result = nothing
    if correlation
        h_corr = res.map_result.h
        if slm
            verbose && @info "Computing MAPestimate with correlation=false for SLM " *
                             "(causal, t1-denoised data)."
            map_nocorr = Main.MAPestimate(
                res.data_t1, date_, subtype;
                tstep=tstep, gamma=gamma, tsmooth=tsmooth, q=q,
                correlation=false, user_start_date=user_start_date,
            )
            h_nocorr = map_nocorr.h
            map_nocorr_result = map_nocorr
        end
    else
        h_nocorr = res.map_result.h
        map_nocorr_result = res.map_result
    end

    slm_result = nothing
    if slm
        @assert h_nocorr !== nothing "SLM requires a correlation=false h-vector."

        slm_sim = simulate_slm_medoid_distribution(
            res.pop_result, h_nocorr;
            M = M, K_slm = K_slm, q = q,
            freq_threshold = freq_threshold,
            medoid_threshold = medoid_threshold,
            verbose = verbose,
            noise = noise, gamma = gamma, Ne = Ne,
            map_result = map_nocorr_result,
        )
        slm_output = slm_sim.output

        slm_cons = Main.consensus_sequence(slm_output)
        slm_emd  = Main.EMD_HiGHS(res.pop_result.d2, slm_output)
        slm_ham  = Main.BioTools.hamming(
            Main.consensus_sequence(res.pop_result.d2), slm_cons,
        )

        slm_output_average = Main.average_distribution(slm_sim.samples)
        slm_cons_average = Main.consensus_sequence(slm_output_average)
        slm_emd_average  = Main.EMD_HiGHS(res.pop_result.d2, slm_output_average)
        slm_ham_average  = Main.BioTools.hamming(
            Main.consensus_sequence(res.pop_result.d2), slm_cons_average,
        )

        slm_result = SLMResult(
            h_nocorr,
            slm_output, nothing,
            slm_cons, nothing,
            slm_emd, nothing,
            slm_ham, nothing,
            slm_output_average, nothing,
            slm_cons_average, nothing,
            slm_emd_average, nothing,
            slm_ham_average, nothing,
            slm_sim.medoid_index,
            slm_sim.samples,
            slm_sim.pairwise_distances,
            slm_sim.medoid_scores,
        )
    end

    return merge(res, (
        h_correlation     = h_corr,
        h_no_correlation  = h_nocorr,
        slm_result        = slm_result,
    ))
end

"""
    run_prediction_pipeline_short_ext_denoised(data, date_, Δ, subtype; slm=false,
                                                K_slm=100, weird_threshold=0.01,
                                                kwargs...)

Causal a priori counterpart to `run_prediction_pipeline_short_ext` -- see
`run_prediction_pipeline_ext_denoised`.

The Null-Emp baseline comes in two versions: `naive_emp`, `ham_naive_emp`,
`bin_d1`, `bin_d2` from the raw observations (as in run_prediction_emp_main.jl),
and `naive_emp_denoised`, `ham_naive_emp_denoised`, `bin_d1_denoised`,
`bin_d2_denoised` from the same t1/t2-denoised data as `d1`/`d2`.
"""
function run_prediction_pipeline_short_ext_denoised(
    data::Matrix{Float64},
    date_::Date,
    Δ::Int,
    subtype::String;
    slm::Bool                 = false,
    K_slm::Int                = 100,
    gamma::Float64            = 20.0,
    q::Int                    = 21,
    tstep::Float64            = 10.0,
    tsmooth::Float64          = 60.0,
    noise::Bool               = false,
    N::Int                    = 1000,
    # Posterior spread scale for the h-sampling (noise=true). Matches the
    # `Ne` default in `simulate_medoid_distribution`, so GenModel and SLM
    # use the same prior width: σ = 1/sqrt(Ne·γ) = 0.01 at Ne=500, γ=20.
    Ne::Int                   = 500,
    K::Int                    = 1000,
    M::Int                    = 100,
    freq_threshold::Float64   = 1.0e-10,
    medoid_threshold::Float64 = 1.0e-6,
    correlation::Bool         = true,
    verbose::Bool             = true,
    weird_threshold::Float64  = 0.01,
    user_start_date::Union{Date,Nothing}           = nothing,
)
    res = run_prediction_pipeline_ext_denoised(
        data, date_, Δ, subtype;
        slm=slm, K_slm=K_slm,
        gamma=gamma, q=q, tstep=tstep, tsmooth=tsmooth,
        noise=noise, N=N, Ne=Ne, K=K, M=M,
        freq_threshold=freq_threshold,
        medoid_threshold=medoid_threshold,
        correlation=correlation, verbose=verbose,
        weird_threshold=weird_threshold,
        user_start_date=user_start_date,
    )

    return (
        naive                  = res.result_naive,
        lower_bound            = res.result_lb,
        prediction             = res.result_prediction,
        prediction_corrected   = res.result_prediction_corrected,
        average                = res.result_average,
        average_corrected      = res.result_average_corrected,
        naive_corrected        = res.result_naive_corrected,
        lower_bound_corrected  = res.result_lb_corrected,
        naive_emp              = res.result_naive_emp,              # raw observations
        naive_emp_corrected    = res.result_naive_emp_corrected,
        naive_emp_denoised     = res.result_naive_emp_denoised,     # t1/t2-denoised observations
        ham_naive_emp_denoised = res.ham_naive_emp_denoised,
        output                 = res.output,
        output_corrected       = res.output_corrected,
        output_average           = res.output_average,
        output_average_corrected = res.output_average_corrected,
        start_day_date         = res.start_day_date,
        date1                  = res.pop_result.date1,
        date2                  = res.pop_result.date2,
        ham_naive              = res.ham_naive,
        ham_prediction         = res.ham_prediction,
        ham_average            = res.ham_average,
        ham_lowerbound         = res.ham_lowerbound,
        ham_naive_emp          = res.ham_naive_emp,
        ham_naive_corrected      = res.ham_naive_corrected,
        ham_prediction_corrected = res.ham_prediction_corrected,
        ham_average_corrected    = res.ham_average_corrected,
        ham_lowerbound_corrected = res.ham_lowerbound_corrected,
        ham_naive_emp_corrected  = res.ham_naive_emp_corrected,
        mlr                    = res.result_mlr,
        mlr_corrected          = res.result_mlr_corrected,
        ham_mlr                = res.ham_mlr,
        ham_mlr_corrected      = res.ham_mlr_corrected,
        h_correlation          = res.h_correlation,
        h_no_correlation       = res.h_no_correlation,
        slm                    = res.slm_result,
        d1                     = res.pop_result.d1,
        d2                     = res.pop_result.d2,
        dlb                    = res.pop_result.dlb,
        bin_d1                 = res.emp_result.bin_d1,
        bin_d2                 = res.emp_result.bin_d2,
        bin_d1_denoised        = res.emp_result_denoised.bin_d1,
        bin_d2_denoised        = res.emp_result_denoised.bin_d2,
        d_mlr                  = res.d_mlr,
        pop1                   = res.pop_result.pop1,
        t1                     = res.pop_result.t1,
        t2                     = res.pop_result.t2,
        mlr_result             = res.mlr_result,
        pop_result             = res.pop_result,
        map_result             = res.map_result,
    )
end


# ===========================================================================
# Summary table (Step A version)
# ===========================================================================

"""
    distance_summary_table_short_ext(result; digits=4)

Like `distance_summary_table_short`, but adds an SLM row when
`result.slm` is not `nothing`. Full selection-coefficient vectors are
not summarised here; they remain available on the result NamedTuple as
`result.h_correlation` and `result.h_no_correlation` for downstream
use (e.g. the CSV export).
"""
function distance_summary_table_short_ext(result; digits::Int = 4)
    rows = ["Null", "GenModel", "GenModel-Average", "Lower Bound", "Null-Emp", "MLR"]
    emd_pop = [result.naive.emd, result.prediction.emd, result.average.emd, result.lower_bound.emd,
               result.naive_emp.emd, result.mlr.emd]
    emd_corr = [
        isnothing(result.naive_corrected)        ? result.naive.emd        : result.naive_corrected.emd,
        isnothing(result.prediction_corrected)   ? result.prediction.emd   : result.prediction_corrected.emd,
        isnothing(result.average_corrected)      ? result.average.emd      : result.average_corrected.emd,
        isnothing(result.lower_bound_corrected)  ? result.lower_bound.emd  : result.lower_bound_corrected.emd,
        isnothing(result.naive_emp_corrected)    ? result.naive_emp.emd    : result.naive_emp_corrected.emd,
        isnothing(result.mlr_corrected)          ? result.mlr.emd          : result.mlr_corrected.emd,
    ]
    ham      = [result.ham_naive, result.ham_prediction, result.ham_average, result.ham_lowerbound,
                result.ham_naive_emp, result.ham_mlr]
    ham_corr = [
        isnothing(result.ham_naive_corrected)      ? result.ham_naive      : result.ham_naive_corrected,
        isnothing(result.ham_prediction_corrected) ? result.ham_prediction : result.ham_prediction_corrected,
        isnothing(result.ham_average_corrected)    ? result.ham_average    : result.ham_average_corrected,
        isnothing(result.ham_lowerbound_corrected) ? result.ham_lowerbound : result.ham_lowerbound_corrected,
        isnothing(result.ham_naive_emp_corrected)  ? result.ham_naive_emp  : result.ham_naive_emp_corrected,
        isnothing(result.ham_mlr_corrected)        ? result.ham_mlr        : result.ham_mlr_corrected,
    ]

    if hasproperty(result, :naive_emp_denoised)   # denoised pipelines only
        push!(rows, "Null-Emp-Denoised")
        push!(emd_pop, result.naive_emp_denoised.emd)
        push!(emd_corr, result.naive_emp_denoised.emd)
        push!(ham, result.ham_naive_emp_denoised)
        push!(ham_corr, result.ham_naive_emp_denoised)
    end

    if result.slm !== nothing
        push!(rows, "SLM")
        push!(emd_pop, result.slm.emd.emd)
        push!(emd_corr, isnothing(result.slm.emd_corrected) ?
                        result.slm.emd.emd : result.slm.emd_corrected.emd)
        push!(ham, result.slm.ham)
        push!(ham_corr, isnothing(result.slm.ham_corrected) ?
                        result.slm.ham : result.slm.ham_corrected)

        push!(rows, "SLM-Average")
        push!(emd_pop, result.slm.emd_average.emd)
        push!(emd_corr, isnothing(result.slm.emd_average_corrected) ?
                        result.slm.emd_average.emd : result.slm.emd_average_corrected.emd)
        push!(ham, result.slm.ham_average)
        push!(ham_corr, isnothing(result.slm.ham_average_corrected) ?
                        result.slm.ham_average : result.slm.ham_average_corrected)
    end

    n = length(rows)

    df = DataFrame(
        "StartDate"             => fill(result.start_day_date, n),
        "ObsDate"               => fill(result.date1, n),
        "PredDate"              => fill(result.date2, n),
        "Method"                => rows,
        "EMD(Pop)"              => round.(emd_pop,  digits=digits),
        "EMD(Pop-filt.)"        => round.(emd_corr, digits=digits),
        "Hamming(ConsSeq)"      => ham,
        "Hamming(ConsSeq-filt.)" => ham_corr,
    )

    sort!(df, "EMD(Pop-filt.)")
    return df
end


# ===========================================================================
# CSV export (Step B)
# ===========================================================================

# Encode a sequence (typically a BioSequence of `AminoAcid`s, but also
# accepts plain integer vectors / matrices rows) as a comma-joined integer
# string so that distribution side-cars are round-trippable.
# Amino-acid sequences are mapped through `Main.aaToint` (defined in
# misc_EMD.jl), which collapses every ambiguous / gap symbol to 1 and
# numbers the 20 canonical residues 2..21.
function _seq_to_string(seq)
    isempty(seq) && return ""
    el = first(seq)
    ints = if el isa Integer
        Int.(seq)
    elseif el isa AbstractFloat
        Int.(seq)
    else
        # Assume AminoAcid (or anything else `aaToint` knows about).
        [Main.aaToint[a] for a in seq]
    end
    return join(ints, ',')
end

# Generate the filename tag shared by all side-cars of one run.
_run_tag(subtype, protein, region, obs_date, horizon) =
    "$(subtype)_$(protein)_$(region)_$(obs_date)_h$(horizon)"

# Sluggify a method name for safe use in a filename.
_method_slug(m::AbstractString) =
    replace(lowercase(m), r"[^a-z0-9]+" => "_") |> s -> strip(s, '_') |> String

"""
    save_distribution_csv(path, dist)

Write a `Dict{BioSequence,Float64}` (or `Dict{Vector,Float64}`) to a CSV
with two columns:
  * `seq`  -- comma-joined integer states of the sequence vector
  * `freq` -- frequency mass.
Rows are sorted by descending frequency.
"""
function save_distribution_csv(path::AbstractString, dist)
    isnothing(dist) && return nothing
    isempty(dist) && return CSV.write(path, DataFrame(seq=String[], freq=Float64[]))

    seqs  = String[]
    freqs = Float64[]
    for (k, v) in dist
        push!(seqs,  _seq_to_string(k))
        push!(freqs, v)
    end
    df = DataFrame(seq = seqs, freq = freqs)
    sort!(df, :freq, rev = true)
    CSV.write(path, df)
    return path
end

"""
    save_pipeline_run(result, outdir;
                      subtype, protein, region, obs_date, horizon,
                      config = Dict{Symbol,Any}(),
                      include_distributions = true,
                      include_h             = true) -> NamedTuple

Persist all relevant outputs from one `run_prediction_pipeline_short_ext`
call to disk. Creates one summary CSV plus side-car files:

```
\$outdir/
    predictions_<tag>.csv              # main summary, one row per method
    h_correlation_<tag>.txt            # length q*L float vector (if available)
    h_no_correlation_<tag>.txt         # length q*L float vector (if available)
    output_<method>_<tag>.csv          # one per method (seq, freq)
    d1_<tag>.csv                       # smoothed observation @ t1
    d2_<tag>.csv                       # smoothed truth      @ t2
    dlb_<tag>.csv                      # lower-bound prediction
    bin_d1_<tag>.csv                   # empirical observation
    bin_d2_<tag>.csv                   # empirical truth
    bin_d1_denoised_<tag>.csv          # empirical observation, denoised (denoised pipelines only)
    bin_d2_denoised_<tag>.csv          # empirical truth, denoised (denoised pipelines only)
```

Results of the denoised pipelines (`run_prediction_pipeline_short_ext_denoised`)
also get a `Null-Emp-Denoised` row and `output_null_emp_denoised_<tag>.csv`;
their `Null-Emp` row is the baseline from the raw observations.

where `<tag>` defaults to `<subtype>_<protein>_<region>_<obs_date>_h<horizon>`; pass
`tag` to choose another file-name stem (`run_emd_prediction.jl` adds the main parameters).

# Arguments
* `result`   -- output of `run_prediction_pipeline_short_ext`.
* `outdir`   -- target directory (created if missing).
* `subtype`, `protein`, `region`, `obs_date`, `horizon`
             -- run identifiers (used both for filenames and as
                columns in the summary CSV).
* `config`   -- a `Dict{Symbol,Any}` of additional run parameters to log
                as extra columns in the summary (e.g. `:tstep`,
                `:tsmooth`, `:gamma`, `:N`, `:K`, `:M`, `:K_slm`,
                `:correlation`, `:noise`, ...). Anything in here becomes
                a constant column on every row of the summary.
* `include_distributions`
             -- write the per-method `output_<method>_<tag>.csv` and
                the d1/d2/dlb/bin_d1/bin_d2 side-cars.
* `include_h`
             -- write `h_correlation_<tag>.txt` and
                `h_no_correlation_<tag>.txt` (only when present in
                `result`).
* `clade_trajectories`
             -- optional `DataFrame` (the output of
                `compute_clade_trajectories`). When supplied, it is
                written to `clade_trajectories_<tag>.csv`.

# Returns
A `NamedTuple` listing the paths actually written.
"""
function save_pipeline_run(result, outdir::AbstractString;
                           subtype::AbstractString,
                           protein::AbstractString,
                           region::AbstractString,
                           obs_date::Date,
                           horizon::Int,
                           config::Dict{Symbol,Any} = Dict{Symbol,Any}(),
                           include_distributions::Bool = true,
                           include_h::Bool             = true,
                           clade_trajectories::Union{DataFrame,Nothing} = nothing,
                           tag::AbstractString = _run_tag(subtype, protein, region, obs_date, horizon))
    isdir(outdir) || mkpath(outdir)
    has_denoised_emp = hasproperty(result, :naive_emp_denoised)   # denoised pipelines only

    # ------------------------------------------------------------------
    # 1. Side-car: h-vectors
    # ------------------------------------------------------------------
    h_corr_path   = ""
    h_nocorr_path = ""
    if include_h
        if result.h_correlation !== nothing
            h_corr_path = joinpath(outdir, "h_correlation_$(tag).txt")
            writedlm(h_corr_path, result.h_correlation)
        end
        if result.h_no_correlation !== nothing
            h_nocorr_path = joinpath(outdir, "h_no_correlation_$(tag).txt")
            writedlm(h_nocorr_path, result.h_no_correlation)
        end
    end

    # ------------------------------------------------------------------
    # 2. Side-cars: distributions
    # ------------------------------------------------------------------
    # Per-method predicted distribution. The "prediction" of a baseline
    # method is the distribution it implicitly returns: Null -> d1,
    # Lower Bound -> dlb, Null-Emp -> bin_d1.
    method_to_dist = Dict(
        "Null"             => result.d1,
        "Lower Bound"      => result.dlb,
        "Null-Emp"         => result.bin_d1,
        "GenModel"         => result.output,
        "GenModel-Average" => result.output_average,
        "MLR"              => result.d_mlr,
    )
    if has_denoised_emp
        method_to_dist["Null-Emp-Denoised"] = result.bin_d1_denoised
    end
    if result.slm !== nothing
        method_to_dist["SLM"]         = result.slm.output
        method_to_dist["SLM-Average"] = result.slm.output_average
    end

    method_to_output_path = Dict{String,String}()
    if include_distributions
        for (method, dist) in method_to_dist
            slug = _method_slug(method)
            path = joinpath(outdir, "output_$(slug)_$(tag).csv")
            save_distribution_csv(path, dist)
            method_to_output_path[method] = path
        end

        # Also save the reference distributions (observation, truth, lower bound)
        references = Tuple{String,Any}[
            ("d1",     result.d1),
            ("d2",     result.d2),
            ("dlb",    result.dlb),
            ("bin_d1", result.bin_d1),
            ("bin_d2", result.bin_d2),
        ]
        if has_denoised_emp
            push!(references, ("bin_d1_denoised", result.bin_d1_denoised),
                              ("bin_d2_denoised", result.bin_d2_denoised))
        end
        for (label, dist) in references
            save_distribution_csv(joinpath(outdir, "$(label)_$(tag).csv"), dist)
        end
    end

    # ------------------------------------------------------------------
    # 3. Main summary CSV (one row per method)
    # ------------------------------------------------------------------
    # Per-method (EMD, EMD-filt, Hamming, Hamming-filt) pulled from the
    # short result.
    rows = ["Null", "GenModel", "GenModel-Average", "Lower Bound", "Null-Emp", "MLR"]
    emd_pop = Float64[
        result.naive.emd,
        result.prediction.emd,
        result.average.emd,
        result.lower_bound.emd,
        result.naive_emp.emd,
        result.mlr.emd,
    ]
    emd_corr = Float64[
        isnothing(result.naive_corrected)       ? result.naive.emd       : result.naive_corrected.emd,
        isnothing(result.prediction_corrected)  ? result.prediction.emd  : result.prediction_corrected.emd,
        isnothing(result.average_corrected)     ? result.average.emd     : result.average_corrected.emd,
        isnothing(result.lower_bound_corrected) ? result.lower_bound.emd : result.lower_bound_corrected.emd,
        isnothing(result.naive_emp_corrected)   ? result.naive_emp.emd   : result.naive_emp_corrected.emd,
        isnothing(result.mlr_corrected)         ? result.mlr.emd         : result.mlr_corrected.emd,
    ]
    ham      = Float64[
        result.ham_naive, result.ham_prediction, result.ham_average, result.ham_lowerbound,
        result.ham_naive_emp, result.ham_mlr,
    ]
    ham_corr = Float64[
        isnothing(result.ham_naive_corrected)      ? result.ham_naive      : result.ham_naive_corrected,
        isnothing(result.ham_prediction_corrected) ? result.ham_prediction : result.ham_prediction_corrected,
        isnothing(result.ham_average_corrected)    ? result.ham_average    : result.ham_average_corrected,
        isnothing(result.ham_lowerbound_corrected) ? result.ham_lowerbound : result.ham_lowerbound_corrected,
        isnothing(result.ham_naive_emp_corrected)  ? result.ham_naive_emp  : result.ham_naive_emp_corrected,
        isnothing(result.ham_mlr_corrected)        ? result.ham_mlr        : result.ham_mlr_corrected,
    ]
    if has_denoised_emp
        push!(rows,     "Null-Emp-Denoised")
        push!(emd_pop,  result.naive_emp_denoised.emd)
        push!(emd_corr, result.naive_emp_denoised.emd)
        push!(ham,      result.ham_naive_emp_denoised)
        push!(ham_corr, result.ham_naive_emp_denoised)
    end
    if result.slm !== nothing
        push!(rows,     "SLM")
        push!(emd_pop,  result.slm.emd.emd)
        push!(emd_corr, isnothing(result.slm.emd_corrected) ?
                        result.slm.emd.emd : result.slm.emd_corrected.emd)
        push!(ham,      result.slm.ham)
        push!(ham_corr, isnothing(result.slm.ham_corrected) ?
                        result.slm.ham : result.slm.ham_corrected)

        push!(rows,     "SLM-Average")
        push!(emd_pop,  result.slm.emd_average.emd)
        push!(emd_corr, isnothing(result.slm.emd_average_corrected) ?
                        result.slm.emd_average.emd : result.slm.emd_average_corrected.emd)
        push!(ham,      result.slm.ham_average)
        push!(ham_corr, isnothing(result.slm.ham_average_corrected) ?
                        result.slm.ham_average : result.slm.ham_average_corrected)
    end

    n = length(rows)

    # Identify which h-vector each method used (if any).
    method_to_h = Dict("GenModel"         => h_corr_path,
                       "GenModel-Average" => h_corr_path,
                       "SLM"              => h_nocorr_path,
                       "SLM-Average"      => h_nocorr_path)

    df = DataFrame(
        subtype       = fill(subtype, n),
        protein       = fill(protein, n),
        region        = fill(region, n),
        start_date    = fill(result.start_day_date, n),
        obs_date      = fill(result.date1, n),
        pred_date     = fill(result.date2, n),
        horizon_days  = fill(horizon, n),
        method        = rows,
        emd_pop       = emd_pop,
        emd_pop_filt  = emd_corr,
        ham_cons      = ham,
        ham_cons_filt = ham_corr,
        output_file   = [get(method_to_output_path, m, "") for m in rows],
        h_file        = [get(method_to_h, m, "") for m in rows],
    )

    # Inject any extra config knobs the caller wants logged as constant columns.
    for (k, v) in config
        df[!, String(k)] = fill(v, n)
    end

    summary_path = joinpath(outdir, "predictions_$(tag).csv")
    CSV.write(summary_path, df)

    # ------------------------------------------------------------------
    # 4. Clade trajectories side-car (Step C)
    # ------------------------------------------------------------------
    traj_path = ""
    if clade_trajectories !== nothing
        traj_path = joinpath(outdir, "clade_trajectories_$(tag).csv")
        CSV.write(traj_path, clade_trajectories)
    end

    return (
        summary            = summary_path,
        h_corr             = h_corr_path,
        h_nocorr           = h_nocorr_path,
        outputs            = method_to_output_path,
        clade_trajectories = traj_path,
    )
end


# ===========================================================================
# STEP C: clade-frequency trajectories through time
# ===========================================================================

# Resolve a metadata input to a DataFrame. Accepts either a DataFrame or a
# filesystem path to a CSV (FluStudy's metadata_<...>.csv layout).
_resolve_metadata(md::DataFrame) = md
_resolve_metadata(md::AbstractString) = DataFrame(CSV.File(md))
_resolve_metadata(::Nothing) =
    error("compute_clade_trajectories: metadata is required (either a " *
          "DataFrame or a path to FluStudy's metadata_*.csv).")

"""
    _majority_clade(clades, weights) -> String

Most common clade in `clades`, weighted by `weights`. Ties broken by
first-seen order. Empty / `unassigned` / `unknown` are treated as missing
and only returned when no real label is present.
"""
function _majority_clade(clades::AbstractVector{<:AbstractString},
                         weights::AbstractVector{<:Real};
                         missing_labels = ("", "unassigned", "unknown"))
    miss = Set(String.(missing_labels))
    totals = Dict{String,Float64}()
    for (c, w) in zip(clades, weights)
        totals[c] = get(totals, c, 0.0) + Float64(w)
    end
    real = filter(p -> !(p.first in miss), totals)
    if !isempty(real)
        return first(sort(collect(real), by = x -> -x[2])).first
    end
    return isempty(totals) ? String(first(missing_labels)) :
                              first(sort(collect(totals), by = x -> -x[2])).first
end

"""
    _resolve_sequence_clade_map(m)
        -> Dict{String, Vector{Tuple{Int,String}}}

Accept either a `DataFrame` with columns `day, seq, clade` (as emitted
by `FluStudy.create_study` / `save_study`), or a path to such a CSV.
Build a `Dict{seq -> sorted list of (day, clade) pairs}`.

The map is intentionally NOT collapsed: a single encoded sequence can
appear at many different days with potentially different FASTA clade
labels (nomenclature drift over years is the common case). Callers
typically pick the (day, clade) pair closest to a query time.
"""
function _resolve_sequence_clade_map(m::DataFrame)
    @assert "day"   in names(m) "sequence_to_clade DataFrame missing :day column."
    @assert "seq"   in names(m) "sequence_to_clade DataFrame missing :seq column."
    @assert "clade" in names(m) "sequence_to_clade DataFrame missing :clade column."
    out = Dict{String, Vector{Tuple{Int,String}}}()
    for (d, s, c) in zip(m.day, m.seq, m.clade)
        seq_str   = ismissing(s) ? "" : String(s)
        clade_str = ismissing(c) ? "" : String(c)
        v = get!(out, seq_str, Tuple{Int,String}[])
        push!(v, (Int(d), clade_str))
    end
    for v in values(out)
        sort!(v, by = first)        # ascending by day, for binary-search-style lookups
    end
    return out
end
_resolve_sequence_clade_map(path::AbstractString) =
    _resolve_sequence_clade_map(DataFrame(CSV.File(path)))
_resolve_sequence_clade_map(::Nothing) =
    error("compute_clade_trajectories: `sequence_to_clade` is required " *
          "(either the DataFrame returned by `FluStudy.create_study` or " *
          "the path to `sequence_to_clade_*.csv`).")

# Pick the (day, clade) pair whose `day` is closest to `t_query`. Ties
# in distance go to the more recent day. `pairs` must be sorted by day.
function _clade_at_nearest_day(pairs::Vector{Tuple{Int,String}},
                                t_query::Int)
    best_idx = 1
    best_dist = abs(pairs[1][1] - t_query)
    @inbounds for i in 2:length(pairs)
        d = abs(pairs[i][1] - t_query)
        # `<= ` (not `<`): later occurrences with equal distance win,
        # which gives us "most recent label when nomenclature drifted".
        if d <= best_dist
            best_dist = d
            best_idx  = i
        end
    end
    return pairs[best_idx][2]
end

"""
    _build_species_clades(pop1_sVec, seq_to_clade, t1) -> Vector{String}

For each species in `pop1_sVec`, look up the FASTA clade label that
applied to that encoded sequence at time `t1` (the observation day).
When the sequence has multiple `(day, clade)` observations in the
data, the one whose `day` is closest to `t1` wins.

Errors loudly if any species is missing from the map -- this would
mean the study and the prediction run are out of sync, and is a bug
we want to surface rather than silently mask as "unknown".
"""
function _build_species_clades(pop1_sVec::Matrix{Float64},
                               seq_to_clade::Dict{String, Vector{Tuple{Int,String}}},
                               t1::Int)
    L = size(pop1_sVec, 2)
    out = Vector{String}(undef, size(pop1_sVec, 1))
    missing_idx = Int[]
    for r in 1:size(pop1_sVec, 1)
        # round-then-int: pop1_sVec is Float64 because that's how the
        # pipeline stores it, but every entry should be an integer state
        # in 1..q. `round` tolerates tiny FP noise from upstream
        # arithmetic.
        key = join((round(Int, pop1_sVec[r, j]) for j in 1:L), ',')
        pairs = get(seq_to_clade, key, nothing)
        if pairs === nothing
            push!(missing_idx, r)
            out[r] = "unassigned"
        else
            out[r] = _clade_at_nearest_day(pairs, t1)
        end
    end
    if !isempty(missing_idx)
        error("$(length(missing_idx)) / $(size(pop1_sVec, 1)) pop1 species " *
              "could not be matched against the sequence_to_clade map. " *
              "This means the study was not produced from the same `data` " *
              "matrix as the prediction run. Regenerate the study (or pass " *
              "the matching one).")
    end
    return out
end


# ---------------------------------------------------------------------------
# Snapshot-capable WF run with clade aggregation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Snapshot trajectories: medoid distributions at intermediate times
# ---------------------------------------------------------------------------

"""
    _aggregate_dict_by_clade(d, clade_lookup) -> Dict{String,Float64}

Sum the frequency mass of `d :: Dict{BioSequence,Float64}` into
clade buckets. `clade_lookup` is a callable `enc_str -> clade` (the
encoded-sequence string -> clade label resolver). The result is
normalised so frequencies sum to 1.
"""
function _aggregate_dict_by_clade(d, clade_lookup)
    out = Dict{String,Float64}()
    total = 0.0
    for (seq, f) in d
        enc_str = join((Main.aaToint[a] for a in seq), ',')
        c = clade_lookup(enc_str)
        out[c] = get(out, c, 0.0) + f
        total += f
    end
    total > 0 && (for k in keys(out); out[k] /= total; end)
    return out
end

"""
    _wf_clade_snapshots(pop_result, map_result, snapshot_gens, clade_lookup;
                        noise, gamma, q, N, K, M,
                        freq_threshold, medoid_threshold, verbose)
        -> Vector{Dict{String,Float64}}

For each snapshot generation `g` in `snapshot_gens`, run
`simulate_medoid_distribution` with `T_gen = g` (i.e. forecast g
generations forward from `t1`) and aggregate the medoid distribution
by clade.

`snapshot_gens` is the vector of generation indices (0 = t1,
t2 - t1 = t2). When `g == 0` the snapshot is just the smoothed
observation `pop_result.d1`. When `g > 0` we re-run the medoid pipeline
with the requested horizon -- mirroring exactly what GenModel does at
the forecast time, just with a shorter horizon.

`clade_lookup` is `_lookup_clade` from `compute_clade_trajectories`.
"""
function _wf_clade_snapshots(pop_result::NamedTuple,
                             map_result::NamedTuple,
                             snapshot_gens::Vector{Int},
                             clade_lookup;
                             noise::Bool               = false,
                             gamma::Float64            = 20.0,
                             q::Int                    = 21,
                             N::Int                    = 1000,
                             K::Int                    = 1,
                             M::Int                    = 50,
                             freq_threshold::Float64   = 0.0,
                             medoid_threshold::Float64 = 1.0e-6,
                             verbose::Bool             = true)
    snaps = Vector{Dict{String,Float64}}(undef, length(snapshot_gens))
    for (i, g) in enumerate(snapshot_gens)
        if g == 0
            # At t1 there's nothing to simulate: the predicted population
            # IS the observation. Aggregate d1 directly.
            snaps[i] = _aggregate_dict_by_clade(pop_result.d1, clade_lookup)
            continue
        end
        verbose && @info "GenModel snapshot $i/$(length(snapshot_gens)): " *
                         "T_gen=$g, M=$M, K=$K."
        # Build a per-snapshot pop_result NamedTuple where t2 has been
        # set to (t1 + g). `simulate_medoid_distribution` uses
        # `pop_result.t2 - pop_result.t1` to derive its T_gen.
        pop_at_g = merge(pop_result, (t2 = pop_result.t1 + g,))
        sim = Main.simulate_medoid_distribution(
            pop_at_g, map_result;
            noise = noise, gamma = gamma, q = q,
            N = N, K = K, M = M,
            freq_threshold = freq_threshold,
            medoid_threshold = medoid_threshold,
            verbose = false,
        )
        snaps[i] = _aggregate_dict_by_clade(sim.output, clade_lookup)
    end
    return snaps
end

"""
    _slm_clade_snapshots(pop_result, h_nocorr, snapshot_gens, clade_lookup;
                         q, M, K_slm, freq_threshold, medoid_threshold,
                         verbose) -> Vector{Dict{String,Float64}}

SLM analogue of `_wf_clade_snapshots`: at each snapshot generation, run
`simulate_slm_medoid_distribution` with the requested horizon and
aggregate the medoid distribution by clade.
"""
function _slm_clade_snapshots(pop_result::NamedTuple,
                              h_nocorr::Vector{Float64},
                              snapshot_gens::Vector{Int},
                              clade_lookup;
                              q::Int                    = 21,
                              M::Int                    = 50,
                              K_slm::Int                = 1,
                              freq_threshold::Float64   = 0.0,
                              medoid_threshold::Float64 = 1.0e-6,
                              verbose::Bool             = true)
    snaps = Vector{Dict{String,Float64}}(undef, length(snapshot_gens))
    for (i, g) in enumerate(snapshot_gens)
        if g == 0
            snaps[i] = _aggregate_dict_by_clade(pop_result.d1, clade_lookup)
            continue
        end
        verbose && @info "SLM snapshot $i/$(length(snapshot_gens)): " *
                         "T_gen=$g, M=$M, K_slm=$K_slm."
        sim = simulate_slm_medoid_distribution(
            pop_result, h_nocorr;
            T_gen = g, M = M, K_slm = K_slm, q = q,
            freq_threshold = freq_threshold,
            medoid_threshold = medoid_threshold,
            verbose = false,
        )
        snaps[i] = _aggregate_dict_by_clade(sim.output, clade_lookup)
    end
    return snaps
end

"""
    _mlr_clade_snapshots(mlr_result, snapshot_days, clade_lookup;
                         freq_threshold) -> Vector{Dict{String,Float64}}

For each snapshot day, evaluate the trained MLR model via
`predict_both_probs_counts` and aggregate the resulting
distribution by clade. This is the same recipe used to build
`d_mlr` at the main forecast date, but evaluated at every snapshot
along the way.
"""
function _mlr_clade_snapshots(mlr_result::NamedTuple,
                              snapshot_days::Vector{Int},
                              clade_lookup;
                              freq_threshold::Float64 = 0.0)
    snaps = Vector{Dict{String,Float64}}(undef, length(snapshot_days))
    for (i, t) in enumerate(snapshot_days)
        out = Main.predict_both_probs_counts(mlr_result.mlrModel, Float64(t))
        seq_last = mlr_result.u_seqs0[out.last.idx, :]
        d_mlr_t  = Main.sequence_distribution(seq_last, out.last.p)
        d_mlr_t  = _safe_select_subset(d_mlr_t, freq_threshold)
        snaps[i] = _aggregate_dict_by_clade(d_mlr_t, clade_lookup)
    end
    return snaps
end


# ---------------------------------------------------------------------------
# Empirical clade trajectory (smoothed)
# ---------------------------------------------------------------------------

"""
    _empirical_clade_trajectory(metadata, snapshot_days; tsmooth, missing_labels)
        -> Vector{Dict{String,Float64}}

Gaussian-smoothed observed clade frequencies at each day in
`snapshot_days`. Reads `metadata.day`, `metadata.count`, `metadata.clade`.
"""
function _empirical_clade_trajectory(metadata::DataFrame,
                                      snapshot_days::Vector{Int};
                                      tsmooth::Real = 60.0,
                                      missing_labels = ("", "unassigned", "unknown"))
    days   = Float64.(metadata.day)
    counts = Float64.(metadata.count)
    clades = String[ismissing(c) ? "" : String(c) for c in metadata.clade]

    miss = Set(String.(missing_labels))
    norm = 1.0 / (tsmooth * sqrt(2π))
    tcut = 3.0 * tsmooth

    out = Vector{Dict{String,Float64}}(undef, length(snapshot_days))
    for (i, t) in enumerate(snapshot_days)
        agg = Dict{String,Float64}()
        @inbounds for k in eachindex(days)
            dt = days[k] - t
            abs(dt) > tcut && continue
            w = counts[k] * exp(-(dt / tsmooth)^2 / 2) * norm
            c = clades[k]
            agg[c] = get(agg, c, 0.0) + w
        end
        # Drop missing-label buckets ONLY if any real label has positive mass.
        total_real = sum((v for (k, v) in agg if !(k in miss)); init = 0.0)
        if total_real > 0
            for k in collect(keys(agg))
                k in miss && delete!(agg, k)
            end
            total = total_real
        else
            total = sum(values(agg); init = 0.0)
        end
        total > 0 && (for k in keys(agg); agg[k] /= total; end)
        out[i] = agg
    end
    return out
end


# ---------------------------------------------------------------------------
# Top-level: compute trajectories for every method
# ---------------------------------------------------------------------------

"""
    compute_clade_trajectories(result, sequence_to_clade, metadata;
                               n_snapshots_past   = 10,
                               n_snapshots_models = 10,
                               tsmooth     = 60.0,
                               noise       = false,
                               gamma       = 20.0,
                               N           = 1000,
                               q           = 21,
                               K           = 1,
                               M           = 50,
                               K_slm       = 1,
                               freq_threshold   = 0.0,
                               medoid_threshold = 1.0e-6,
                               include_methods  = nothing,
                               clade_initial_date = nothing,
                               verbose          = true) -> DataFrame

Build a long-format DataFrame of clade frequencies through time, one row
per (method, snapshot_day, clade).

By default, snapshots are taken in the FORECAST window only -- evenly
spaced between `t1` (inclusive) and `t2` (inclusive). When
`clade_initial_date` is supplied, the function also records
`n_snapshots_past` empirical-only snapshots in the PAST window --
evenly spaced between `clade_initial_date` and `t1`.

NOTE: `clade_initial_date` controls only the trajectory plot's past
panel. It is distinct from the pipeline's `user_start_date` (a.k.a.
`initial_date` on the config struct), which anchors the MAP time axis
during the prediction itself.

`n_snapshots_past` and `n_snapshots_models` are independent: a typical
configuration is a dense empirical past (e.g. 20 snapshots, cheap
Gaussian smoothing) and a sparser forecast (e.g. 4 snapshots, since
each forecast snapshot re-runs an entire medoid simulation). Set
`n_snapshots_past = 0` to disable the past panel entirely. Passing
the legacy `n_snapshots` kwarg sets both to the same value.

At each forecast snapshot, model-based methods re-run their full
prediction machinery with the snapshot's `T_gen`, so the trajectory is
"what would the model have predicted if the forecast horizon were
shorter":

  * GenModel -- `Main.simulate_medoid_distribution(pop_at_g, map_result;
                M, K, ...)` with `T_gen = g`, then aggregate the medoid
                Dict by clade.
  * SLM      -- `simulate_slm_medoid_distribution(pop_result, h_nocorr;
                T_gen = g, M, K_slm, ...)`, aggregate medoid by clade.
  * MLR      -- `Main.predict_both_probs_counts(mlr_result.mlrModel,
                Float64(snapshot_day))`, aggregate the resulting
                distribution by clade.
  * Null / Null-Emp / Lower Bound -- constant over time
                (`d1` / `bin_d1` / `dlb` aggregated by clade once).
  * Empirical -- Gaussian-smoothed `metadata.clade` evaluated at each
                snapshot day; emitted in both past and forecast periods.

`result` must be a `run_prediction_pipeline_short_ext` return value.

`sequence_to_clade` is the per-row encoded-sequence -> FASTA clade
table emitted by `FluStudy.create_study` (the
`study.sequence_to_clade` DataFrame, columns `day`, `seq`, `clade`)
or `FluStudy.save_study` (the `sequence_to_clade_*.csv` file). Either
a `DataFrame` or a filesystem path is accepted.

The map is NOT collapsed across days: the same encoded sequence
typically appears multiple times with the same clade label, but can
appear with different labels if nextstrain nomenclature was refined
between observations. For each `pop1` species, the clade label at the
`day` closest to `t1` is used (this captures "what was this sequence
labelled as around observation time").

`metadata` is the FluStudy metadata (DataFrame or path to
`metadata_*.csv`). Used only to compute the smoothed EMPIRICAL clade
trajectory.

Columns:
  * `method`        -- one of `Empirical`, `Null`, `Null-Emp`,
                        `Lower Bound`, `GenModel`, `MLR`, `SLM`.
  * `period`        -- `"past"` for snapshots between `clade_initial_date`
                        and `t1`, `"forecast"` for snapshots between `t1`
                        and `t2`. The two windows share the boundary date
                        `result.date1` (no duplication: the boundary
                        belongs to the forecast period).
  * `snapshot_day`  -- integer day (compatible with `data[:, 1]`).
  * `snapshot_date` -- the calendar date for that snapshot.
  * `clade`         -- clade label.
  * `freq`          -- frequency at that (method, snapshot) pair.

Past snapshots have `method = "Empirical"` only (no models predict the
past). Forecast snapshots have all method labels (including a forward
Empirical evaluation, which is the smoothed truth).

`include_methods` filters the FORECAST methods; the past-period
`Empirical` rows are always emitted when `clade_initial_date` is supplied.

# Computational cost
For each non-zero forecast snapshot, GenModel does `M*K` WF runs of
length `T_gen = g`, and SLM does `M*K_slm * L` single-locus WF runs.
With `n_snapshots_models = 10, M = 50, K = K_slm = 1`, the total work
is roughly `n_snapshots_models` times the cost of the main forecast
for each of GenModel and SLM. MLR is cheap (one regression evaluation
per snapshot). Empirical is cheap (one Gaussian-smoothing pass).
"""
function compute_clade_trajectories(result,
                                    sequence_to_clade,
                                    metadata;
                                    # Two separate snapshot densities. Pass
                                    # `n_snapshots` to set both at once (legacy
                                    # convenience); pass `n_snapshots_past` and
                                    # `n_snapshots_models` separately to
                                    # distinguish (e.g. dense empirical past
                                    # but sparse, expensive forecast medoids).
                                    n_snapshots::Union{Int,Nothing}        = nothing,
                                    n_snapshots_past::Union{Int,Nothing}   = nothing,
                                    n_snapshots_models::Union{Int,Nothing} = nothing,
                                    tsmooth::Real      = 60.0,
                                    # Forwarded to the per-snapshot
                                    # `simulate_medoid_distribution` /
                                    # `simulate_slm_medoid_distribution` /
                                    # `predict_both_probs_counts` calls.
                                    # Defaults match the main forecast.
                                    noise::Bool               = false,
                                    gamma::Float64            = 20.0,
                                    N::Int                    = 1000,
                                    q::Int                    = 21,
                                    K::Int                    = 1,
                                    M::Int                    = 50,
                                    K_slm::Int                = 1,
                                    freq_threshold::Float64   = 0.0,
                                    medoid_threshold::Float64 = 1.0e-6,
                                    include_methods           = nothing,
                                    # The start date of the PAST empirical
                                    # trajectory panel. NOT to be confused
                                    # with the pipeline's `user_start_date`
                                    # (a.k.a. `initial_date` on the config
                                    # struct), which anchors the MAP time
                                    # axis. The two are independent: you
                                    # can train the pipeline with a long
                                    # historical window but only plot the
                                    # last year or two of past empirical
                                    # clades.
                                    clade_initial_date::Union{Date,Nothing} = nothing,
                                    # Deprecated alias; kept for one cycle.
                                    initial_date::Union{Date,Nothing}       = nothing,
                                    verbose::Bool             = true)
    # Backward compat: `initial_date` was the previous name.
    if initial_date !== nothing
        if clade_initial_date !== nothing
            error("compute_clade_trajectories: pass `clade_initial_date`, " *
                  "not both `clade_initial_date` and the deprecated alias " *
                  "`initial_date`.")
        end
        @warn "compute_clade_trajectories: the `initial_date` kwarg has " *
              "been renamed to `clade_initial_date` to distinguish it from " *
              "the pipeline's `user_start_date`. Please update your call."
        clade_initial_date = initial_date
    end
    # Resolve snapshot counts. Either pass the single legacy `n_snapshots`
    # (applied to both periods) or both `n_snapshots_past` and
    # `n_snapshots_models`. Mixing is an error.
    if n_snapshots !== nothing
        if n_snapshots_past !== nothing || n_snapshots_models !== nothing
            error("compute_clade_trajectories: pass either `n_snapshots` or " *
                  "the separate `n_snapshots_past` / `n_snapshots_models` " *
                  "kwargs, not both.")
        end
        n_snapshots_past   = n_snapshots
        n_snapshots_models = n_snapshots
    else
        n_snapshots_past   = something(n_snapshots_past,   10)
        n_snapshots_models = something(n_snapshots_models, 10)
    end
    @assert n_snapshots_past   >= 0
    @assert n_snapshots_models >= 1 "Need at least 1 forecast snapshot (t1)."
    md = _resolve_metadata(metadata)
    @assert "day"   in names(md)
    @assert "count" in names(md)
    @assert "clade" in names(md)

    seq_to_clade = _resolve_sequence_clade_map(sequence_to_clade)

    # ----- Snapshot grid (in days) -----
    # Use the exact day numbers stored in `data[:, 1]` (those are the
    # ones `unique_sequences` was applied to). Reconstructing them from
    # date arithmetic is error-prone because `data[:, 1]` doesn't
    # necessarily start at 1.
    t1_day = result.t1
    t2_day = result.t2
    snapshot_days  = collect(round.(Int,
                          range(t1_day, t2_day, length = n_snapshots_models)))
    # Map snapshot day-numbers back to calendar dates via the known
    # (date1, t1_day) correspondence.
    snapshot_dates = result.date1 .+ Day.(snapshot_days .- t1_day)
    snapshot_gens  = snapshot_days .- t1_day            # 0 at t1, t2-t1 at t2

    @info "Forecast snapshot grid: $n_snapshots_models time points between " *
          "$(result.date1) (day $t1_day) and $(result.date2) (day $t2_day)."

    # ----- Aligned species clades for pop1 -----
    pop1_sVec, _ = result.pop1
    species_clades = _build_species_clades(pop1_sVec, seq_to_clade, t1_day)
    @assert length(species_clades) == size(pop1_sVec, 1)

    # Pick which methods to emit. `Empirical` is always available; the
    # others depend on the result fields actually being populated.
    available = ["Empirical", "Null", "Null-Emp", "Lower Bound", "GenModel", "MLR"]
    result.slm !== nothing && push!(available, "SLM")
    selected = include_methods === nothing ? available :
               [m for m in include_methods if m in available]

    # Containers ------------------------------------------------------------
    rows = Vector{NamedTuple{(:method, :period, :snapshot_day, :snapshot_date,
                              :clade, :freq),
                             Tuple{String,String,Int,Date,String,Float64}}}()

    # --- Past empirical snapshots (only if `clade_initial_date` was
    # provided and `n_snapshots_past >= 1`). The past window runs
    # [clade_initial_date, t1_day - 1], strictly before the forecast
    # window's t1. We deliberately leave t1_day to the forecast period
    # so the boundary date is unambiguous. -----------------------------
    if clade_initial_date !== nothing && n_snapshots_past >= 1
        # Convert calendar date to day-number compatible with `data[:, 1]`.
        # `result.date1` corresponds to `t1_day`, so:
        clade_initial_day = t1_day - Dates.value(result.date1 - clade_initial_date)
        if clade_initial_day >= t1_day
            @warn "clade_initial_date ($clade_initial_date) is not before " *
                  "observation date ($(result.date1)); skipping past trajectory."
        else
            past_days  = collect(round.(Int,
                              range(clade_initial_day, t1_day - 1,
                                    length = n_snapshots_past)))
            past_dates = result.date1 .+ Day.(past_days .- t1_day)
            past_emp   = _empirical_clade_trajectory(md, past_days;
                                                    tsmooth = tsmooth)
            for (i, dct) in enumerate(past_emp)
                for (c, f) in dct
                    push!(rows, (method        = "Empirical",
                                 period        = "past",
                                 snapshot_day  = past_days[i],
                                 snapshot_date = past_dates[i],
                                 clade         = c,
                                 freq          = f))
                end
            end
            @info "Past empirical trajectory: $n_snapshots_past snapshots " *
                  "between $clade_initial_date (day $clade_initial_day) and " *
                  "$(result.date1 - Day(1)) (day $(t1_day - 1))."
        end
    end

    # --- Forecast snapshots (all selected methods, period = "forecast") ----
    _push_dict! = (method, day_idx, dct) -> begin
        for (c, f) in dct
            push!(rows, (method        = method,
                         period        = "forecast",
                         snapshot_day  = snapshot_days[day_idx],
                         snapshot_date = snapshot_dates[day_idx],
                         clade         = c,
                         freq          = f))
        end
    end

    # 1) Empirical (forward, smoothed truth over the forecast window) -------
    if "Empirical" in selected
        emp = _empirical_clade_trajectory(md, snapshot_days; tsmooth = tsmooth)
        for (i, dct) in enumerate(emp); _push_dict!("Empirical", i, dct); end
    end

    # Helper: convert a `Dict{BioSequence,Float64}` (Null/LB/Null-Emp use
    # this) into clade-aggregated frequencies. Each BioSequence's integer
    # encoding is looked up first against `pop1`'s species table (fast
    # path, already time-aware for t1) and then against the per-row
    # `seq_to_clade` map (using the nearest-day clade label, with t1 as
    # the reference time). Anything that still misses lands in
    # `"unknown"` -- this is rare and indicates a sequence not present
    # in the training data `data` matrix at all.
    enc_to_clade = Dict{String,String}()
    for i in axes(pop1_sVec, 1)
        key = join((round(Int, pop1_sVec[i, j]) for j in axes(pop1_sVec, 2)), ',')
        enc_to_clade[key] = species_clades[i]
    end

    function _lookup_clade(enc_str::AbstractString)
        # Fast path: same species table that `_build_species_clades`
        # used, so already time-aware.
        c = get(enc_to_clade, enc_str, nothing)
        c !== nothing && return c
        # Fall back to the full seq->day-clade table.
        pairs = get(seq_to_clade, enc_str, nothing)
        pairs === nothing && return "unknown"
        return _clade_at_nearest_day(pairs, t1_day)
    end

    function dict_to_clade_freqs(d)
        out = Dict{String,Float64}()
        total = 0.0
        for (seq, f) in d
            enc_str = join((Main.aaToint[a] for a in seq), ',')
            c = _lookup_clade(enc_str)
            out[c] = get(out, c, 0.0) + f
            total += f
        end
        total > 0 && (for k in keys(out); out[k] /= total; end)
        return out
    end

    # 2) Null (constant = clade-mapped d1) ----------------------------------
    if "Null" in selected
        agg = dict_to_clade_freqs(result.d1)
        for i in 1:n_snapshots_models; _push_dict!("Null", i, agg); end
    end

    # 3) Null-Emp (constant = clade-mapped bin_d1) --------------------------
    if "Null-Emp" in selected
        agg = dict_to_clade_freqs(result.bin_d1)
        for i in 1:n_snapshots_models; _push_dict!("Null-Emp", i, agg); end
    end

    # 4) Lower Bound (constant = clade-mapped dlb) --------------------------
    if "Lower Bound" in selected
        agg = dict_to_clade_freqs(result.dlb)
        for i in 1:n_snapshots_models; _push_dict!("Lower Bound", i, agg); end
    end

    # 5) GenModel ------------------------------------------------------------
    # At each snapshot generation g, re-run `simulate_medoid_distribution`
    # with T_gen = g. This produces exactly the same kind of medoid
    # distribution the main forecast uses, just at intermediate horizons.
    if "GenModel" in selected
        if result.pop_result === nothing || result.map_result === nothing
            @warn "GenModel trajectory requested but pop_result / map_result " *
                  "is not exposed on the short result. Skipping."
        else
            snaps = _wf_clade_snapshots(
                result.pop_result, result.map_result,
                Vector{Int}(snapshot_gens), _lookup_clade;
                noise = noise, gamma = gamma, q = q,
                N = N, K = K, M = M,
                freq_threshold = freq_threshold,
                medoid_threshold = medoid_threshold,
                verbose = verbose,
            )
            for (i, dct) in enumerate(snaps); _push_dict!("GenModel", i, dct); end
        end
    end

    # 6) MLR -----------------------------------------------------------------
    # Evaluate `predict_both_probs_counts` at each snapshot day; this is
    # the same recipe used to build `d_mlr` for the main forecast, just
    # repeated at every horizon along the way.
    if "MLR" in selected
        if result.mlr_result === nothing
            @warn "MLR trajectory requested but mlr_result is not on the " *
                  "short result. Skipping."
        else
            snaps = _mlr_clade_snapshots(
                result.mlr_result, Vector{Int}(snapshot_days), _lookup_clade;
                freq_threshold = freq_threshold,
            )
            for (i, dct) in enumerate(snaps); _push_dict!("MLR", i, dct); end
        end
    end

    # 7) SLM -----------------------------------------------------------------
    # SLM analogue of GenModel: at each snapshot generation, run the
    # SLM medoid pipeline with T_gen = g.
    if "SLM" in selected && result.slm !== nothing
        if result.h_no_correlation === nothing
            @warn "SLM trajectory requested but h_no_correlation is nothing."
        else
            snaps = _slm_clade_snapshots(
                result.pop_result, result.h_no_correlation,
                Vector{Int}(snapshot_gens), _lookup_clade;
                q = q, M = M, K_slm = K_slm,
                freq_threshold = freq_threshold,
                medoid_threshold = medoid_threshold,
                verbose = verbose,
            )
            for (i, dct) in enumerate(snaps); _push_dict!("SLM", i, dct); end
        end
    end

    return DataFrame(rows)
end


# ===========================================================================
# STEP D: configuration struct + study cache
# ===========================================================================

# Reserved struct -- DON'T add fields lightly. Every field is part of the
# user-facing API, persisted to TOML, and round-tripped through
# `save_config` / `load_config`. Group the fields logically so the TOML
# stays readable.

"""
    PredictionConfig

Bundles every parameter needed to run the prediction pipeline (study
preprocessing through trajectory plotting). Mutable, so a user can tweak
a single field after construction.

Construction follows Julia's keyword-only-args convention: all fields
have defaults except the bare minimum (`fasta_path`, `subtype`,
`protein`, `start_date`, `stop_date`, `observation_date`, `initial_date`,
`Δ`). Pass anything else as `name = value`.

```julia
config = PredictionConfig(
    fasta_path        = "data/raw/H3N2_HA.fasta",
    subtype           = "H3N2",
    protein           = "HA",
    region            = "all",
    start_date        = Date("2009-01-01"),
    stop_date         = Date("2020-03-01"),
    observation_date  = Date("2013-02-03"),
    initial_date      = Date("2012-02-03"),
    Δ                 = 270,
    study_dir         = "data/derived/study",
    output_dir        = "outputs/emd_prediction",
)
```

Clade labels are read from the `clade` field of the FASTA headers.

Save / load:

```julia
save_config(config, "config.toml")
config2 = load_config("config.toml")
```
"""
Base.@kwdef mutable struct PredictionConfig
    # --- Study identifiers (used to build filename tags) ---------------
    subtype::String
    protein::String
    region::String                              = "all"

    # --- Study inputs (used by FluStudy.create_study) ------------------
    fasta_path::String
    start_date::Date
    stop_date::Date
    tstep::Float64                              = 10.0
    tsmooth::Float64                            = 60.0
    gap_threshold::Float64                      = 0.05
    mut_threshold::Float64                      = 20.0
    window_days::Float64                        = 360.0
    q::Int                                      = 21
    deduplicate_by_name::Bool                   = true
    legacy_dedup::Bool                          = false
    header_fields::Vector{String}               = ["strain", "isolate_id",
                                                    "date", "segment", "hash",
                                                    "region", "clade"]

    # --- Filesystem layout ---------------------------------------------
    study_dir::String                           = "data"
    output_dir::String                          = "outputs"

    # --- Prediction-pipeline parameters --------------------------------
    observation_date::Date
    initial_date::Date
    Δ::Int                                      # horizon in days
    gamma::Float64                              = 20.0
    noise::Bool                                 = false
    K::Int                                      = 100
    M::Int                                      = 50
    K_slm::Int                                  = 1
    N::Int                                      = 1000
    # Posterior spread scale for the h-sampling when `noise = true`:
    # sigma = 1/sqrt(Ne*gamma) is the prior width on selection
    # coefficients (0.01 at Ne=500, gamma=20). Shared by GenModel and SLM.
    # Distinct from `N`, the WF resampling population size.
    Ne::Int                                     = 500
    correlation::Bool                           = true
    slm::Bool                                   = true
    freq_threshold::Float64                     = 1.0e-6
    medoid_threshold::Float64                   = 1.0e-6
    verbose::Bool                               = true

    # --- Trajectory parameters -----------------------------------------
    n_snapshots_past::Int                       = 20
    n_snapshots_models::Int                     = 4
    # Start date of the past empirical trajectory panel. Independent of
    # `initial_date`: `initial_date` shifts the MAP time axis during the
    # prediction, while `clade_initial_date` only controls how far back
    # the past trajectory plot reaches. Defaults to `initial_date` when
    # left as `nothing` (see `compute_clade_trajectories` call in the
    # driver).
    clade_initial_date::Union{Date,Nothing}     = nothing
    traj_methods::Union{Vector{String},Nothing} = nothing
    min_clade_freq::Float64                     = 0.02
end

# Convenience: filename tag shared across study artifacts.
_study_tag(c::PredictionConfig) = "$(c.subtype)_$(c.protein)_$(c.region)"

# The set of `create_study`-relevant fields that go into the manifest.
# Only these affect the cache validity.
const _STUDY_INPUT_FIELDS = (
    :subtype, :protein, :region,
    :fasta_path,
    :start_date, :stop_date,
    :tstep, :tsmooth,
    :gap_threshold, :mut_threshold, :window_days,
    :q, :deduplicate_by_name, :legacy_dedup,
    :header_fields,
)

# Helper: serialize a single field to a TOML-safe value.
function _to_toml(v)
    if v isa Date
        return string(v)
    elseif v isa Nothing
        return ""           # TOML has no `null`; reload as `nothing` below.
    elseif v isa Vector
        return collect(v)   # ensure a proper TOML array
    else
        return v
    end
end

# Helper: parse a TOML value back to the struct's expected type.
function _from_toml(::Type{Date}, v)
    if v isa Date
        return v
    elseif v isa AbstractString
        return Date(v)
    else
        error("Cannot coerce $(typeof(v)) to Date.")
    end
end
_from_toml(::Type{Union{String,Nothing}}, v) =
    (v isa AbstractString && isempty(v)) ? nothing :
    (v isa AbstractString)                ? String(v) :
    v === nothing                         ? nothing :
    error("Cannot coerce $(typeof(v)) to Union{String,Nothing}.")
_from_toml(::Type{Union{Date,Nothing}}, v) =
    v === nothing                         ? nothing :
    (v isa AbstractString && isempty(v)) ? nothing :
    (v isa AbstractString)                ? Date(v) :
    v isa Date                            ? v :
    error("Cannot coerce $(typeof(v)) to Union{Date,Nothing}.")
_from_toml(::Type{Union{Vector{String},Nothing}}, v) =
    v === nothing                                  ? nothing :
    (v isa AbstractString && isempty(v))           ? nothing :
    (v isa AbstractVector && isempty(v))           ? nothing :
    (v isa AbstractVector)                         ? String.(v) :
    error("Cannot coerce $(typeof(v)) to Union{Vector{String},Nothing}.")
_from_toml(::Type{Vector{String}}, v) = String.(v)
_from_toml(::Type{T}, v) where {T} = convert(T, v)


"""
    save_config(config, path)

Serialize a `PredictionConfig` to `path` as TOML. The file is human-
readable and can be edited by hand before re-loading.
"""
function save_config(config::PredictionConfig, path::AbstractString)
    out = Dict{String,Any}()
    for f in fieldnames(PredictionConfig)
        out[String(f)] = _to_toml(getfield(config, f))
    end
    open(path, "w") do io
        println(io, "# PredictionConfig (auto-generated, safe to edit by hand).")
        TOML.print(io, out; sorted = true)
    end
    @info "Saved config to $path."
    return path
end

"""
    load_config(path) -> PredictionConfig

Read a TOML file produced by `save_config` (or hand-written) and build a
`PredictionConfig`. Fields not present in the TOML keep their struct
defaults.
"""
function load_config(path::AbstractString)
    raw = TOML.parsefile(path)
    kwargs = Dict{Symbol,Any}()
    for f in fieldnames(PredictionConfig)
        key = String(f)
        haskey(raw, key) || continue
        ftype = fieldtype(PredictionConfig, f)
        kwargs[f] = _from_toml(ftype, raw[key])
    end
    return PredictionConfig(; kwargs...)
end


# ---------------------------------------------------------------------------
# Study cache
# ---------------------------------------------------------------------------

# Filenames that `FluStudy.save_study` emits for one (subtype, protein,
# region) tuple. All five must be present (plus the manifest) for the
# cache to be considered valid.
_study_artifact_paths(c::PredictionConfig) = begin
    tag = _study_tag(c)
    Dict(
        :data              => joinpath(c.study_dir, "data_$(tag).txt"),
        :frequency         => joinpath(c.study_dir, "frequency_$(tag).txt"),
        :metadata          => joinpath(c.study_dir, "metadata_$(tag).csv"),
        :name_index        => joinpath(c.study_dir, "name_index_$(tag).csv"),
        :sequence_to_clade => joinpath(c.study_dir, "sequence_to_clade_$(tag).csv"),
        :manifest          => joinpath(c.study_dir, "study_manifest_$(tag).toml"),
    )
end

# Manifest = serialized snapshot of the kwargs that determined the
# cached study. Compared field-by-field against the current config when
# loading; any mismatch refuses to load.
function _write_manifest(config::PredictionConfig, path::AbstractString)
    out = Dict{String,Any}()
    for f in _STUDY_INPUT_FIELDS
        out[String(f)] = _to_toml(getfield(config, f))
    end
    open(path, "w") do io
        println(io, "# Study manifest -- auto-generated by load_or_create_study.")
        println(io, "# Records the create_study inputs that built this cache.")
        println(io, "# Don't edit; clear the study_dir to invalidate.")
        TOML.print(io, out; sorted = true)
    end
end

function _check_manifest(config::PredictionConfig, path::AbstractString)
    raw = TOML.parsefile(path)
    mismatches = String[]
    for f in _STUDY_INPUT_FIELDS
        key      = String(f)
        expected = _to_toml(getfield(config, f))
        cached   = get(raw, key, missing)
        # Normalise the "absent / empty" cases to a single sentinel so a
        # round-tripped `nothing` (stored as "") compares equal to a key
        # that's missing entirely.
        if (cached === missing || cached == "") && (expected === nothing || expected == "")
            continue
        end
        if cached != expected
            push!(mismatches, "$key: cached=$(repr(cached)) config=$(repr(expected))")
        end
    end
    return mismatches
end

"""
    load_or_create_study(config; force_rebuild = false) -> NamedTuple

Either load a previously-saved FluStudy from `config.study_dir`, or
build one from scratch by calling `Main.FluStudy.create_study` and
persisting it via `Main.FluStudy.save_study`.

Return shape matches `FluStudy.create_study`:
    (data, frequency, metadata, name_index, sequence_to_clade)

Cache-validity rule: every artifact file must exist AND every
`create_study` kwarg recorded in the manifest must match the current
config. Any deviation triggers a rebuild (so changing `start_date` or
swapping the FASTA invalidates the cache automatically). Pass
`force_rebuild = true` to skip the check entirely.

`Main.FluStudy` must be loaded in the caller's scope (typically via
`include("FluStudy.jl"); using .FluStudy` in the driver) before
calling this function.
"""
function load_or_create_study(config::PredictionConfig;
                              force_rebuild::Bool = false)
    @assert isdefined(Main, :FluStudy) "FluStudy is not loaded. " *
        "Run `include(\"FluStudy.jl\"); using .FluStudy` before " *
        "`load_or_create_study`."

    paths = _study_artifact_paths(config)
    all_present = all(isfile, values(paths))

    needs_build = force_rebuild || !all_present
    if !needs_build
        mismatches = _check_manifest(config, paths[:manifest])
        if !isempty(mismatches)
            error("""
                Cached study in `$(config.study_dir)` was built with
                different `create_study` kwargs than the current config:

                  $(join(mismatches, "\n  "))

                Clear the directory (or run with `force_rebuild = true`)
                to regenerate.
                """)
        end
    end

    if needs_build
        force_rebuild && @info "load_or_create_study: force_rebuild = true; " *
                               "rebuilding study from FASTA."
        if !all_present
            missing_files = [k for (k, p) in paths if !isfile(p)]
            @info "load_or_create_study: missing artifact(s) " *
                  "$(missing_files); rebuilding study from FASTA."
        end

        # Resolve `region` and `country` filters. Our convention: the
        # FluStudy kwarg `region` accepts `"all"` (no filter), a single
        # value, or a vector. The config holds it as a single String,
        # so we map "all" -> nothing (skip filter) and pass everything
        # else verbatim.
        region_arg = config.region == "all" ? nothing : config.region

        study = Main.FluStudy.create_study(
            config.fasta_path;
            protein             = config.protein,
            start_date          = config.start_date,
            stop_date           = config.stop_date,
            tstep               = config.tstep,
            tsmooth             = config.tsmooth,
            gap_threshold       = config.gap_threshold,
            mut_threshold       = config.mut_threshold,
            window_days         = config.window_days,
            q                   = config.q,
            deduplicate_by_name = config.deduplicate_by_name,
            legacy_dedup        = config.legacy_dedup,
            header_fields       = config.header_fields,
            region              = region_arg,
        )

        Main.FluStudy.save_study(study, config.study_dir;
            subtype = config.subtype,
            protein = config.protein,
            region  = config.region,
        )
        _write_manifest(config, paths[:manifest])
        @info "load_or_create_study: built and saved study to " *
              "`$(config.study_dir)`."
        return study
    end

    # Load -----------------------------------------------------------------
    @info "load_or_create_study: loading cached study from " *
          "`$(config.study_dir)`."
    data       = readdlm(paths[:data])
    frequency  = readdlm(paths[:frequency])
    metadata   = DataFrame(CSV.File(paths[:metadata]))
    name_index = DataFrame(CSV.File(paths[:name_index]))
    seq_clade  = DataFrame(CSV.File(paths[:sequence_to_clade]))
    return (
        data              = data,
        frequency         = frequency,
        metadata          = metadata,
        name_index        = name_index,
        sequence_to_clade = seq_clade,
    )
end


end # module PredictionPipeline
