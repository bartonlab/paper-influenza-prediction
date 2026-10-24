include(joinpath(@__DIR__, "..", "shared", "mpl_inference.jl"))   # MPL kernel shared with MAP estimation
include("misc_EMD.jl")   # EMD (HiGHS), consensus sequences, medoid selection
include("MLR.jl")        # multinomial logistic-regression baseline

using Base.Threads
using Dates

# ---------------------------------------------------------------------------
# unique_sequences  (used by MLRmodel, bins_distributions_at_t1_t2, etc.)
# ---------------------------------------------------------------------------

"""
unique_sequences(seqs, counts)

Collapse duplicate rows in `seqs` summing their normalised counts.
Returns (unique_seq_matrix, normalised_counts).
"""
function unique_sequences(seqs::Matrix{Float64}, counts::Vector{Float64})
    M, L = size(seqs)
    inv_sum = inv(sum(counts))
    # Dict preserves insertion order; same first-seen ordering as the nested-loop version.
    acc = Dict{Tuple, Float64}()
    sizehint!(acc, min(M, 1024))
    @inbounds for j in 1:M
        key = ntuple(i -> seqs[j, i], L)
        w = counts[j] * inv_sum
        acc[key] = get(acc, key, 0.0) + w
    end
    nu = length(acc)
    out = Matrix{Float64}(undef, nu, L)
    newCounts = Vector{Float64}(undef, nu)
    idx = 1
    for (key, w) in acc
        @inbounds for c in 1:L
            out[idx, c] = key[c]
        end
        newCounts[idx] = w
        idx += 1
    end
    return out, newCounts
end


# ---------------------------------------------------------------------------
# smoothed_distribution_at_time  (used by distributions_at_t1_t2)
# ---------------------------------------------------------------------------

"""
smoothed_distribution_at_time(t, times, counts, seqs, tstep, tsmooth)

Return the unique sequences and their normalised smoothed weights at time t.
"""
function smoothed_distribution_at_time(
    t::Int,
    times::Vector{Float64},
    counts::Vector{Float64},
    seqs::Matrix{Float64},
    tstep::Float64,
    tsmooth::Float64,
)
    times_int         = Int.(times)
    position_in_time  = findlast(==(t), times_int)

    counts_obs            = counts[1:position_in_time]
    times_obs             = times[1:position_in_time]
    seqs_observed         = seqs[1:position_in_time, :]

    normalized_counts_obs = normalizeCounts(times_obs, counts_obs)

    times_measure = collect(1.0:tstep:t)
    times_measure[end] != t && push!(times_measure, Float64(t))

    effectiveCounts_list = Vector{Vector{Float64}}(undef, length(times_measure))
    for (i, t_) in enumerate(times_measure)
        effectiveCounts_list[i] = smoothCounts_normalized(
            normalized_counts_obs, times_obs, t_, tsmooth)
    end

    ncounts   = last(effectiveCounts_list)
    mask      = ncounts .> 0.0
    counts_   = ncounts[mask]
    sequences = seqs_observed[mask, :]

    return unique_sequences(sequences, counts_)
end


# ---------------------------------------------------------------------------
# closest_future_time_and_date  (used by distributions_at_t1_t2, bins_*)
# ---------------------------------------------------------------------------

"""
closest_future_time_and_date(t1, x, u_times, start_date, end_date; throw_error)

Find the time in u_times closest to t1 + x and return it together with its date.
"""
function closest_future_time_and_date(
    t1::Int, x::Int,
    u_times::Vector{Int},
    start_date::Date,
    end_date::Date;
    throw_error::Bool = true,
)
    target  = t1 + x
    max_day = Dates.value(end_date - start_date)

    if target < 0 || target > max_day
        msg = "Target time t1 + x = $target is outside [0, $max_day]."
        throw_error ? error(msg) : (@warn msg; return nothing)
    end

    idx = searchsortedfirst(u_times, target)

    if idx == 1
        t2 = u_times[1]
    elseif idx > length(u_times)
        t2 = u_times[end]
    else
        left  = u_times[idx-1]
        right = u_times[idx]
        t2 = abs(left - target) <= abs(right - target) ? left : right
    end

    return t2, start_date + Day(t2)
end


# ---------------------------------------------------------------------------
# distributions_at_t1_t2  (smoothed populations at observation and target times)
# ---------------------------------------------------------------------------

"""
distributions_at_t1_t2(data, date_, Δ, subtype; tstep, tsmooth, freq_threshold)

Build smoothed sequence distributions at t1 (observation date) and t2 = t1 + Δ.
Also returns the lower-bound distribution (nearest-neighbour assignment).
"""
function distributions_at_t1_t2(
    data::Matrix{Float64},
    date_::Date,
    Δ::Int,
    subtype::String;
    tstep::Float64 = 10.0,
    tsmooth::Float64 = 15.0,
    freq_threshold::Float64 = 0.0,
)
    times  = data[:, 1]
    counts = data[:, 2]
    seqs   = data[:, 3:end]

    u_times = sort!(unique(Int.(times)))

    start_date =
        subtype == "H3N2"  ? Date(2009, 1, 1) :
        subtype == "H1N1"  ? Date(2009, 5, 1) :
        error("Unknown subtype: $subtype")

    end_date = Date(2020, 3, 1)

    target_day = Dates.value(date_ - start_date)
    println("Target day (t1): ", target_day)
    t1 = closest_utime(target_day, u_times)
    println("Closest utime (t1): ", t1)
    u_sequences_1, u_counts_1 = smoothed_distribution_at_time(
        t1, times, counts, seqs, tstep, tsmooth)
    d1 = sequence_distribution(u_sequences_1, u_counts_1)
    d1 = select_distribution_subset(d1, freq_threshold)

    t2, d2_date = closest_future_time_and_date(t1, Δ, u_times, start_date, end_date)
    println("Target day + Δ (t2): ", target_day + Δ)
    println("Closest utime (t2): ", t2) 
    println("Target date (t2): ", d2_date)
    u_sequences_2, u_counts_2 = smoothed_distribution_at_time(
        t2, times, counts, seqs, tstep, tsmooth)
    d2 = sequence_distribution(u_sequences_2, u_counts_2)
    d2 = select_distribution_subset(d2, freq_threshold)

    dlb = lower_bound_dist(d1, d2)
    dlb = select_distribution_subset(dlb, freq_threshold)

    return (
        t1    = t1,
        date1 = start_date + Day(t1),
        pop1  = (u_sequences_1, u_counts_1),
        d1    = d1,
        t2    = t2,
        date2 = d2_date,
        d2    = d2,
        dlb   = dlb,
    )
end


# ---------------------------------------------------------------------------
# distributions_at_t1_t2_causal  (causal / leakage-free counterpart)
# ---------------------------------------------------------------------------

"""
distributions_at_t1_t2_causal(data, date_, Δ, subtype; tstep, tsmooth,
                              freq_threshold, q, weird_threshold,
                              data_t1, data_t2)

Causal counterpart to `distributions_at_t1_t2`. `pop1`/`d1` are built from
`data` with weird low-frequency mutations corrected using ONLY information
available up to `t1` (via `_denoise_data_upto`); `d2` is built
independently, using its OWN correction based ONLY on information available
up to `t2`. Neither distribution ever sees information from beyond its own
anchor date -- see the discussion around `run_prediction_pipeline_denoised`.

`data_t1`/`data_t2` may be passed in pre-computed (e.g. by
`run_prediction_pipeline_denoised`, which needs the same two denoised
datasets for `bins_distributions_at_t1_t2_causal` and `h`-fitting) to avoid
recomputing the expensive `_denoise_data_upto` step twice; when omitted
(`nothing`, the default), they're computed here as before -- this function
remains fully self-sufficient when called directly.

Also returns `data_t1`, the t1-denoised data matrix, so callers (`h`-fitting,
MLR-fitting) can reuse the same corrected data without recomputing it.

Requires `FluStudy.jl` to be loaded (for `frequency_trajectory`).
"""
function distributions_at_t1_t2_causal(
    data::Matrix{Float64},
    date_::Date,
    Δ::Int,
    subtype::String;
    tstep::Float64 = 10.0,
    tsmooth::Float64 = 15.0,
    freq_threshold::Float64 = 0.0,
    q::Int = 21,
    weird_threshold::Float64 = 0.01,
    data_t1::Union{Matrix{Float64},Nothing} = nothing,
    data_t2::Union{Matrix{Float64},Nothing} = nothing,
)
    u_times = sort!(unique(Int.(data[:, 1])))

    start_date =
        subtype == "H3N2"  ? Date(2009, 1, 1) :
        subtype == "H1N1"  ? Date(2009, 5, 1) :
        subtype == "H3N2_2023"  ? Date(2023, 1, 1) :
        error("Unknown subtype: $subtype")

    end_date =
        subtype == "H3N2"  ? Date(2020, 3, 1) :
        subtype == "H1N1"  ? Date(2020, 3, 1) :
        subtype == "H3N2_2023"  ? Date(2026, 3, 1) :
        error("Unknown subtype: $subtype")

    target_day = Dates.value(date_ - start_date)
    t1 = closest_utime(target_day, u_times)
    t2, d2_date = closest_future_time_and_date(t1, Δ, u_times, start_date, end_date)

    data_t1 = data_t1 !== nothing ? data_t1 :
        _denoise_data_upto(data, t1, q; tstep=tstep, tsmooth=tsmooth,
                           threshold=weird_threshold)
    data_t2 = data_t2 !== nothing ? data_t2 :
        _denoise_data_upto(data, t2, q; tstep=tstep, tsmooth=tsmooth,
                           threshold=weird_threshold)

    u_sequences_1, u_counts_1 = smoothed_distribution_at_time(
        t1, data_t1[:, 1], data_t1[:, 2], data_t1[:, 3:end], tstep, tsmooth)
    d1 = sequence_distribution(u_sequences_1, u_counts_1)
    d1 = select_distribution_subset(d1, freq_threshold)

    u_sequences_2, u_counts_2 = smoothed_distribution_at_time(
        t2, data_t2[:, 1], data_t2[:, 2], data_t2[:, 3:end], tstep, tsmooth)
    d2 = sequence_distribution(u_sequences_2, u_counts_2)
    d2 = select_distribution_subset(d2, freq_threshold)

    dlb = lower_bound_dist(d1, d2)
    dlb = select_distribution_subset(dlb, freq_threshold)

    return (
        t1      = t1,
        date1   = start_date + Day(t1),
        pop1    = (u_sequences_1, u_counts_1),
        d1      = d1,
        t2      = t2,
        date2   = d2_date,
        d2      = d2,
        dlb     = dlb,
        data_t1 = data_t1,
    )
end


# ---------------------------------------------------------------------------
# bins_distributions_at_t1_t2  (empirical bin-based populations)
# ---------------------------------------------------------------------------

"""
bins_distributions_at_t1_t2(data, date_, Δ, subtype; bin_size, freq_threshold)

Build empirical sequence distributions by collecting raw observations in a
30-day window around t1 and t2.
"""
function bins_distributions_at_t1_t2(
    data::Matrix{Float64},
    date_::Date,
    Δ::Int,
    subtype::String;
    bin_size::Int = 30,
    freq_threshold::Float64 = 0.0,
)
    times  = Int.(data[:, 1])
    counts = data[:, 2]
    seqs   = data[:, 3:end]

    u_times = sort!(unique(times))

    start_date =
        subtype == "H3N2"  ? Date(2009, 1, 1) :
        subtype == "H1N1"  ? Date(2009, 5, 1) :
        subtype == "H3N2_2023"  ? Date(2023, 1, 1) :
        error("Unknown subtype: $subtype")

    end_date = subtype == "H3N2_2023" ? Date(2026, 3, 1) : Date(2020, 3, 1)

    target_day = Dates.value(date_ - start_date)
    t1 = closest_utime(target_day, u_times)
    println("Target day (t1): ", target_day)
    println("Closest utime (t1): ", t1)
    left_1            = t1 - bin_size
    index_time_1_start = findfirst(t -> t >= left_1, times)
    index_time_1_end   = findlast(==(t1), times)

    isnothing(index_time_1_start) && error("No observations in bin ending at t1 = $t1.")
    isnothing(index_time_1_end)   && error("No observations exactly at t1 = $t1.")

    seqs_1   = seqs[index_time_1_start:index_time_1_end, :]
    counts_1 = counts[index_time_1_start:index_time_1_end]

    u_sequences_1, u_counts_1 = unique_sequences(seqs_1, counts_1)
    d1 = sequence_distribution(u_sequences_1, u_counts_1)
    d1 = select_distribution_subset(d1, freq_threshold)

    t2, d2_date = closest_future_time_and_date(t1, Δ, u_times, start_date, end_date)
    println("Target day + Δ (t2): ", target_day + Δ)
    println("Closest utime (t2): ", t2)
    println("Target date (t2): ", d2_date)

    left_2            = t2 - bin_size
    index_time_2_start = findfirst(t -> t >= left_2, times)
    index_time_2_end   = findlast(==(t2), times)

    isnothing(index_time_2_start) && error("No observations in bin ending at t2 = $t2.")
    isnothing(index_time_2_end)   && error("No observations exactly at t2 = $t2.")

    seqs_2   = seqs[index_time_2_start:index_time_2_end, :]
    counts_2 = counts[index_time_2_start:index_time_2_end]

    u_sequences_2, u_counts_2 = unique_sequences(seqs_2, counts_2)
    d2 = sequence_distribution(u_sequences_2, u_counts_2)
    d2 = select_distribution_subset(d2, freq_threshold)

    return (
        t1              = t1,
        date1           = start_date + Day(t1),
        index_time_1    = (index_time_1_start, index_time_1_end),
        bin_d1          = d1,
        t2              = t2,
        date2           = d2_date,
        index_time_2    = (index_time_2_start, index_time_2_end),
        bin_d2          = d2,
    )
end


# ---------------------------------------------------------------------------
# bins_distributions_at_t1_t2_causal  (causal / leakage-free counterpart)
# ---------------------------------------------------------------------------

"""
bins_distributions_at_t1_t2_causal(data, date_, Δ, subtype; bin_size,
                                   freq_threshold, q, weird_threshold,
                                   tstep, tsmooth, data_t1, data_t2)

Causal counterpart to `bins_distributions_at_t1_t2`: `bin_d1` is built from
data denoised using ONLY information up to `t1`; `bin_d2` independently,
using ONLY information up to `t2`. `tstep`/`tsmooth` are used only for the
weird-mutation-detection trajectory (`_denoise_data_upto`), not for the
bin extraction itself, which stays a plain time-window count exactly like
`bins_distributions_at_t1_t2`.

`data_t1`/`data_t2` may be passed in pre-computed (e.g. by
`run_prediction_pipeline_denoised`, which already computed them for
`distributions_at_t1_t2_causal`) to avoid recomputing `_denoise_data_upto`
a second time; when omitted (`nothing`, the default), they're computed
here as before -- this function remains fully self-sufficient when called
directly.

Requires `FluStudy.jl` to be loaded (for `frequency_trajectory`).
"""
function bins_distributions_at_t1_t2_causal(
    data::Matrix{Float64},
    date_::Date,
    Δ::Int,
    subtype::String;
    bin_size::Int = 30,
    freq_threshold::Float64 = 0.0,
    q::Int = 21,
    weird_threshold::Float64 = 0.01,
    tstep::Float64 = 10.0,
    tsmooth::Float64 = 60.0,
    data_t1::Union{Matrix{Float64},Nothing} = nothing,
    data_t2::Union{Matrix{Float64},Nothing} = nothing,
)
    u_times = sort!(unique(Int.(data[:, 1])))

    start_date =
        subtype == "H3N2"  ? Date(2009, 1, 1) :
        subtype == "H1N1"  ? Date(2009, 5, 1) :
        subtype == "H3N2_2023"  ? Date(2023, 1, 1) :
        error("Unknown subtype: $subtype")

    end_date =
        subtype == "H3N2"  ? Date(2020, 3, 1) :
        subtype == "H1N1"  ? Date(2020, 3, 1) :
        subtype == "H3N2_2023"  ? Date(2026, 3, 1) :
        error("Unknown subtype: $subtype")

    target_day = Dates.value(date_ - start_date)
    t1 = closest_utime(target_day, u_times)
    t2, d2_date = closest_future_time_and_date(t1, Δ, u_times, start_date, end_date)

    data_t1 = data_t1 !== nothing ? data_t1 :
        _denoise_data_upto(data, t1, q; tstep=tstep, tsmooth=tsmooth,
                           threshold=weird_threshold)
    data_t2 = data_t2 !== nothing ? data_t2 :
        _denoise_data_upto(data, t2, q; tstep=tstep, tsmooth=tsmooth,
                           threshold=weird_threshold)

    times_1  = Int.(data_t1[:, 1])
    counts_1 = data_t1[:, 2]
    seqs_1   = data_t1[:, 3:end]

    left_1              = t1 - bin_size
    index_time_1_start  = findfirst(t -> t >= left_1, times_1)
    index_time_1_end    = findlast(==(t1), times_1)
    isnothing(index_time_1_start) && error("No observations in bin ending at t1 = $t1.")
    isnothing(index_time_1_end)   && error("No observations exactly at t1 = $t1.")

    seqs_1w   = seqs_1[index_time_1_start:index_time_1_end, :]
    counts_1w = counts_1[index_time_1_start:index_time_1_end]
    u_sequences_1, u_counts_1 = unique_sequences(seqs_1w, counts_1w)
    d1 = sequence_distribution(u_sequences_1, u_counts_1)
    d1 = select_distribution_subset(d1, freq_threshold)

    times_2  = Int.(data_t2[:, 1])
    counts_2 = data_t2[:, 2]
    seqs_2   = data_t2[:, 3:end]

    left_2              = t2 - bin_size
    index_time_2_start  = findfirst(t -> t >= left_2, times_2)
    index_time_2_end    = findlast(==(t2), times_2)
    isnothing(index_time_2_start) && error("No observations in bin ending at t2 = $t2.")
    isnothing(index_time_2_end)   && error("No observations exactly at t2 = $t2.")

    seqs_2w   = seqs_2[index_time_2_start:index_time_2_end, :]
    counts_2w = counts_2[index_time_2_start:index_time_2_end]
    u_sequences_2, u_counts_2 = unique_sequences(seqs_2w, counts_2w)
    d2 = sequence_distribution(u_sequences_2, u_counts_2)
    d2 = select_distribution_subset(d2, freq_threshold)

    return (
        t1              = t1,
        date1           = start_date + Day(t1),
        index_time_1    = (index_time_1_start, index_time_1_end),
        bin_d1          = d1,
        t2              = t2,
        date2           = d2_date,
        index_time_2    = (index_time_2_start, index_time_2_end),
        bin_d2          = d2,
    )
end


# ---------------------------------------------------------------------------
# Active-position detection  (for noise estimation in simulate_medoid)
# ---------------------------------------------------------------------------

"""
find_active_positions(observed_path_freq, ref_s, q; lost_thr, fix_thr)

Return all (position, state) indices that are polymorphic at the last time point.
"""
function find_active_positions(
    observed_path_freq::Matrix{Float64},
    ref_s::Vector{Int},
    q::Int;
    lost_thr::Float64 = 0.05,
    fix_thr::Float64  = 0.95,
)
    L         = length(ref_s)
    last_freq = @view observed_path_freq[:, end]
    active_pos = Int[]

    for pos in 1:L
        for state in 1:q
            if state != ref_s[pos]
                f = last_freq[(pos-1)*q + state]
                if lost_thr < f < fix_thr
                    push!(active_pos, (pos-1)*q + state)
                end
            end
        end
    end

    return active_pos
end


# ---------------------------------------------------------------------------
# Covariance submatrix helper
# ---------------------------------------------------------------------------

"""
get_submatrix(CC, mutations)

Extract the sub-covariance matrix at the given mutation indices.
"""
function get_submatrix(CC::Matrix{Float64}, mutations::Vector{Int64})
    return CC[mutations, mutations]
end


# ---------------------------------------------------------------------------
# Gaussian perturbation for selection coefficients
# ---------------------------------------------------------------------------

"""
GaussianVector(sqrtC, mu)

Draw a multivariate Gaussian sample: sqrtC * N(0,I) + mu.
"""
function GaussianVector(sqrtC::Array{Float64,2}, mu::Array{Float64,1})
    Y = randn(Float64, size(sqrtC, 1))
    return sqrtC * Y + mu
end


# ---------------------------------------------------------------------------
# generate_sample_distribution  (one forward simulation → sequence distribution)
# ---------------------------------------------------------------------------

"""
generate_sample_distribution(init_seqs_, init_counts_, h, mut_pos, sq_isC,
    perturb_h, q, L, N, K, T_gen, freq_threshold; parallel_k)

- `perturb_h`: if true, add Gaussian draw to `h_0[mut_pos]` (caller should pass
  `false` when `mut_pos` is empty).
- `K`: must be ≥ 1 (`K == 0` used to hit `reduce(vcat, …)` on an empty set and error).
- `parallel_k`: parallelize the `K` WF runs; set `false` when an outer loop is
  already using `Threads.@threads` (e.g. `simulate_medoid_distribution` with `M > 1`).
"""
function generate_sample_distribution(
    init_seqs_::Matrix{Float64},
    init_counts_::Vector{Float64},
    h::Vector{Float64},
    mut_pos,
    sq_isC,
    perturb_h::Bool,
    q::Int,
    L::Int,
    N::Int,
    K::Int,
    T_gen::Int,
    freq_threshold::Float64;
    parallel_k::Bool = true,   # kept for signature compatibility; unused now
)
    K >= 1 || throw(ArgumentError("K must be at least 1, got K=$K"))

    h_0 = copy(h)
    if perturb_h && !isempty(mut_pos)
        h_0[mut_pos] .+= GaussianVector(sq_isC, zeros(length(mut_pos)))
    end

    i_pop, i_sVec, i_nVec = initial_population(init_seqs_, init_counts_, Int64(q), h_0)

    # `evolve_pop_WF` deep-copies `i_pop` and only ever mutates each
    # species' `.n` (count) via the WF resampling -- `.s` (the haplotype
    # itself) never changes. So `sVec_T` is identical to `i_sVec[1]` on
    # every one of the K replicates; only `nVec_T` (a fixed-size vector)
    # varies. Summing counts directly avoids repeatedly `vcat`-ing (and
    # thus re-copying) a growing K*npop-row matrix, which OOMs for large
    # K on HA-scale data (L~566, hundreds of unique haplotypes).
    full_counts = zeros(length(i_nVec[1]))
    for k in 1:K
        _, _, nVec_T = evolve_pop_WF(
            T_gen, N, q, L, h_0, i_pop, i_sVec, i_nVec; record=T_gen)
        full_counts .+= nVec_T
    end
    full_seqs = i_sVec[1]

    # Normalize ONCE, after all K replicates are pooled, so every replicate
    # gets equal weight 1/K (a true uniform average). Renormalizing inside
    # the loop instead turns this into a recency-weighted (exponential
    # decay) combination -- the last replicate gets weight 1/2, the one
    # before it 1/4, etc. -- with an effective sample size of ~3 regardless
    # of K, which defeats the point of averaging over K replicates.
    full_seqs, full_counts = unique_sequences(full_seqs, full_counts)

    d2_full = sequence_distribution(full_seqs, full_counts)
    return select_distribution_subset(d2_full, freq_threshold)
end


# ---------------------------------------------------------------------------
# simulate_medoid_distribution
# ---------------------------------------------------------------------------

"""
simulate_medoid_distribution(pop_result, map_result; noise, gamma, q, N, K, M,
                              freq_threshold, medoid_threshold, verbose)

Run M forward simulations and return the medoid distribution together with
pairwise EMD distances.
"""
function simulate_medoid_distribution(
    pop_result::NamedTuple,
    map_result::NamedTuple;
    noise::Bool           = false,
    gamma::Float64        = 20.0,
    q::Int                = 21,
    N::Int                = 1000,
    K::Int                = 1,
    M::Int                = 100,
    freq_threshold::Float64   = 0.0,
    medoid_threshold::Float64 = 1.0e-6,
    verbose::Bool         = true,
    Ne::Int               = 500,
)
    h                    = map_result.h
    observed_path_freq   = map_result.observed_path_freq
    ref_s                = map_result.ref_s
    seqs_observed        = map_result.seqs_observed
    effectiveCounts_list = map_result.effectiveCounts_list
    times_measure        = map_result.times_measure
    t_c                  = map_result.t_c

    L     = size(seqs_observed, 2)
    T_gen = pop_result.t2 - pop_result.t1
    println("t_gen-: ", T_gen)
    mut_pos = Int[]
    sq_isC  = Matrix{Float64}(undef, 0, 0)

    if noise
        verbose && println("Computing error bar for selection coefficients...")

        Cint_ = Cintegrate(
            seqs_observed, effectiveCounts_list, times_measure, L, q, gamma)

        mut_pos = find_active_positions(observed_path_freq, ref_s, q)

        if !isempty(mut_pos)
            # Posterior covariance of h:  Σ = (1/Ne)·(C_int + γI)⁻¹,
            # using the SAME gamma as the point estimate in `mpl.jl`
            # (h_MAP = (C_int + γI)⁻¹·Δx). Both fall out of the same
            # Gaussian posterior, whose precision is Ne·C_int + σ_s⁻²I:
            # factoring out Ne gives this Σ with γ = 1/(Ne·σ_s²).
            #
            # So `Ne` is NOT a free noise knob -- given γ it fixes the
            # prior width on selection coefficients,
            #     σ_s = 1/sqrt(Ne·γ),
            # which is the quantity to reason about when changing it.
            # With Ne=500, γ=20 this is σ_s = 0.01, matching the observed
            # scale of the coefficients (~0.015). Ne never enters h_MAP,
            # so changing it rescales the error bars only.
            #
            # Note gamma must scale WITH Ne here: writing
            # `inv(Ne*sC_raw + gamma*I)` instead would leave the prior at
            # σ_s = 1/sqrt(γ) = 0.22 in every direction where C_int is
            # weak (short windows), which is the regime this matters in.
            #
            # `Cint_` already has `gamma` added to its diagonal (see
            # `_cintegrate_core!`), so back it out of the submatrix first
            # to recover the RAW integrated covariance -- otherwise gamma
            # would be counted twice, at the wrong relative weight.
            sC     = get_submatrix(Cint_, mut_pos)
            sC_raw = sC - gamma * I
            isC    = inv(sC_raw + gamma * I) / Ne
            isC    = (isC + isC') / 2
            sq_isC = sqrt(isC)
        end
    end

    verbose && println("Performing simulations...")

    init_seqs_, init_counts_ = pop_result.pop1
    samples = Vector{Any}(undef, M)
    parallel_k = (M == 1)

    Threads.@threads for m in 1:M
        samples[m] = generate_sample_distribution(
            init_seqs_, init_counts_,
            h, mut_pos, sq_isC,
            noise && !isempty(mut_pos),
            q, L, N, K, T_gen, freq_threshold;
            parallel_k = parallel_k,
        )
    end

    medoid_index, output, D, scores = medoid_by_emd(
        samples, medoid_threshold; verbose=verbose)

    return (
        output            = output,
        medoid_index      = medoid_index,
        samples           = samples,
        pairwise_distances = D,
        medoid_scores     = scores,
    )
end


# ---------------------------------------------------------------------------
# run_prediction_pipeline  (full pipeline)
# ---------------------------------------------------------------------------

"""
run_prediction_pipeline(data, date_, Δ, subtype; ...)

Full prediction pipeline:
1. Build smoothed (d1, d2) and empirical (bin_d1, bin_d2) distributions.
2. Fit MPL selection coefficients (MAPestimate).
3. Fit MLR model (MLRmodel).
4. Run forward WF simulations and select the medoid (simulate_medoid_distribution).
5. Compute EMD and Hamming distances between all method predictions and the target.
"""
function run_prediction_pipeline(
    data::Matrix{Float64},
    date_::Date,
    Δ::Int,
    subtype::String;
    gamma::Float64            = 20.0,
    q::Int                    = 21,
    tstep::Float64            = 10.0,
    tsmooth::Float64          = 60.0,
    noise::Bool               = false,
    N::Int                    = 1000,
    # Posterior spread scale for the h-sampling (noise=true); see the Σ
    # derivation in `simulate_medoid_distribution`. Distinct from `N`,
    # which is the WF resampling population size.
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
    pop_result = distributions_at_t1_t2(
        data, date_, Δ, subtype;
        tstep=tstep, tsmooth=tsmooth, freq_threshold=freq_threshold)

    map_result = MAPestimate(
        data, date_, subtype;
        tstep=tstep, gamma=gamma, tsmooth=tsmooth, q=q,
        correlation=correlation, user_start_date=user_start_date)

    start_day_date = map_result.start_day_date

    mlr_result = MLRmodel(
        data, date_, subtype;
        tstep=tstep, gamma=gamma, tsmooth=tsmooth, q=q,
        user_start_date=user_start_date)

    emp_result = bins_distributions_at_t1_t2(
        data, date_, Δ, subtype;
        bin_size=30, freq_threshold=freq_threshold)

    medoid_sim_result = simulate_medoid_distribution(
        pop_result, map_result;
        noise=noise, gamma=gamma, q=q, N=N, Ne=Ne, K=K, M=M,
        freq_threshold=freq_threshold,
        medoid_threshold=medoid_threshold, verbose=verbose)

    # Average across the same M samples used for the medoid (no extra
    # simulation cost -- see `average_distribution` in misc_EMD.jl).
    output_average = average_distribution(medoid_sim_result.samples)

    # MLR prediction
    t_future  = Float64(pop_result.t2)
    out       = predict_both_probs_counts(mlr_result.mlrModel, t_future)
    seq_last  = mlr_result.u_seqs0[out.last.idx, :]
    d_mlr     = sequence_distribution(seq_last, out.last.p)
    d_mlr     = select_distribution_subset(d_mlr, freq_threshold)

    # Consensus sequences
    pop_d1_cons      = consensus_sequence(pop_result.d1)
    pop_d2_cons      = consensus_sequence(pop_result.d2)
    pop_dlb_cons     = consensus_sequence(pop_result.dlb)
    emp_d1_cons      = consensus_sequence(emp_result.bin_d1)
    emp_d2_cons      = consensus_sequence(emp_result.bin_d2)
    sim_output_cons  = consensus_sequence(medoid_sim_result.output)
    output_average_cons = consensus_sequence(output_average)
    mlr_cons         = consensus_sequence(d_mlr)

    ham_naive      = BioTools.hamming(pop_d1_cons, pop_d2_cons)
    ham_prediction = BioTools.hamming(pop_d2_cons, sim_output_cons)
    ham_average    = BioTools.hamming(pop_d2_cons, output_average_cons)
    ham_lowerbound = BioTools.hamming(pop_dlb_cons, pop_d2_cons)
    ham_naive_emp  = BioTools.hamming(emp_d1_cons, emp_d2_cons)
    ham_mlr        = BioTools.hamming(pop_d2_cons, mlr_cons)

    # Initialise corrected quantities as nothing
    pop_d1_cons_corrected = pop_d2_cons_corrected = pop_dlb_cons_corrected = nothing
    emp_d1_cons_corrected = emp_d2_cons_corrected = output_cons_corrected = nothing
    output_average_cons_corrected = nothing
    mlr_cons_corrected = nothing
    output_corrected = d1_corrected = d2_corrected = dlb_corrected = nothing
    output_average_corrected = nothing
    result_prediction_corrected = result_naive_corrected = result_lb_corrected = nothing
    result_average_corrected = nothing
    result_naive_emp_corrected = d_mlr_corrected = result_mlr_corrected = nothing
    ham_naive_corrected = ham_prediction_corrected = ham_lowerbound_corrected = nothing
    ham_average_corrected = nothing
    ham_naive_emp_corrected = ham_mlr_corrected = nothing

    result_prediction = EMD_HiGHS(pop_result.d2, medoid_sim_result.output)
    result_average    = EMD_HiGHS(pop_result.d2, output_average)
    result_naive_emp  = EMD_HiGHS(emp_result.bin_d1, emp_result.bin_d2)
    result_mlr        = EMD_HiGHS(pop_result.d2, d_mlr)

    if full_frequency !== nothing
        L_corr = size(map_result.seqs_observed, 2)
        weird_df = find_weird_mutations(full_frequency, L_corr, q)
        (output_corrected, _)  = correct_weird_mutations(medoid_sim_result.output, weird_df)
        (output_average_corrected, _) = correct_weird_mutations(output_average, weird_df)
        (d1_corrected, _)      = correct_weird_mutations(pop_result.d1, weird_df)
        (d2_corrected, _)      = correct_weird_mutations(pop_result.d2, weird_df)
        (dlb_corrected, _)     = correct_weird_mutations(pop_result.dlb, weird_df)
        (emp_d1_corrected, _)  = correct_weird_mutations(emp_result.bin_d1, weird_df)
        (emp_d2_corrected, _)  = correct_weird_mutations(emp_result.bin_d2, weird_df)
        (d_mlr_corrected, _)   = correct_weird_mutations(d_mlr, weird_df)

        result_prediction_corrected = EMD_HiGHS(d2_corrected, output_corrected)
        result_average_corrected   = EMD_HiGHS(d2_corrected, output_average_corrected)
        result_naive_corrected      = EMD_HiGHS(d1_corrected, d2_corrected)
        result_lb_corrected          = EMD_HiGHS(dlb_corrected, d2_corrected)
        result_naive_emp_corrected   = EMD_HiGHS(emp_d1_corrected, emp_d2_corrected)
        result_mlr_corrected        = EMD_HiGHS(d2_corrected, d_mlr_corrected)

        pop_d1_cons_corrected  = consensus_sequence(d1_corrected)
        pop_d2_cons_corrected  = consensus_sequence(d2_corrected)
        pop_dlb_cons_corrected = consensus_sequence(dlb_corrected)
        emp_d1_cons_corrected  = consensus_sequence(emp_d1_corrected)
        emp_d2_cons_corrected  = consensus_sequence(emp_d2_corrected)
        output_cons_corrected  = consensus_sequence(output_corrected)
        output_average_cons_corrected = consensus_sequence(output_average_corrected)
        mlr_cons_corrected     = consensus_sequence(d_mlr_corrected)

        ham_naive_corrected       = BioTools.hamming(pop_d1_cons_corrected, pop_d2_cons_corrected)
        ham_prediction_corrected  = BioTools.hamming(pop_d2_cons_corrected, output_cons_corrected)
        ham_average_corrected     = BioTools.hamming(pop_d2_cons_corrected, output_average_cons_corrected)
        ham_lowerbound_corrected  = BioTools.hamming(pop_dlb_cons_corrected, pop_d2_cons_corrected)
        ham_naive_emp_corrected   = BioTools.hamming(emp_d1_cons_corrected, emp_d2_cons_corrected)
        ham_mlr_corrected         = BioTools.hamming(pop_d2_cons_corrected, mlr_cons_corrected)
    end

    result_naive = EMD_HiGHS(pop_result.d1, pop_result.d2)
    result_lb    = EMD_HiGHS(pop_result.dlb, pop_result.d2)

    return (
        pop_result  = pop_result,
        emp_result  = emp_result,
        map_result  = map_result,
        sim_result  = medoid_sim_result,
        start_day_date = start_day_date,
        result_naive = result_naive,
        result_naive_emp = result_naive_emp,
        result_lb    = result_lb,
        result_prediction = result_prediction,
        result_prediction_corrected = result_prediction_corrected,
        result_average = result_average,
        result_average_corrected = result_average_corrected,
        result_naive_corrected      = result_naive_corrected,
        result_naive_emp_corrected   = result_naive_emp_corrected,
        result_lb_corrected         = result_lb_corrected,
        output           = medoid_sim_result.output,
        output_corrected = output_corrected,
        output_average           = output_average,
        output_average_corrected = output_average_corrected,
        d1_corrected  = d1_corrected,
        d2_corrected  = d2_corrected,
        dlb_corrected = dlb_corrected,
        pop_d1_cons  = pop_d1_cons,
        pop_d2_cons  = pop_d2_cons,
        pop_dlb_cons = pop_dlb_cons,
        emp_d1_cons  = emp_d1_cons,
        emp_d2_cons  = emp_d2_cons,
        sim_output_cons = sim_output_cons,
        output_average_cons = output_average_cons,
        pop_d1_cons_corrected  = pop_d1_cons_corrected,
        pop_d2_cons_corrected  = pop_d2_cons_corrected,
        pop_dlb_cons_corrected = pop_dlb_cons_corrected,
        emp_d1_cons_corrected  = emp_d1_cons_corrected,
        emp_d2_cons_corrected  = emp_d2_cons_corrected,
        output_cons_corrected  = output_cons_corrected,
        output_average_cons_corrected = output_average_cons_corrected,
        ham_naive      = ham_naive,
        ham_prediction = ham_prediction,
        ham_average    = ham_average,
        ham_lowerbound = ham_lowerbound,
        ham_naive_emp  = ham_naive_emp,
        ham_naive_corrected      = ham_naive_corrected,
        ham_prediction_corrected = ham_prediction_corrected,
        ham_average_corrected    = ham_average_corrected,
        ham_lowerbound_corrected = ham_lowerbound_corrected,
        ham_naive_emp_corrected  = ham_naive_emp_corrected,
        mlr_result       = mlr_result,
        d_mlr            = d_mlr,
        d_mlr_corrected  = d_mlr_corrected,
        result_mlr       = result_mlr,
        result_mlr_corrected = result_mlr_corrected,
        mlr_cons         = mlr_cons,
        mlr_cons_corrected = mlr_cons_corrected,
        ham_mlr          = ham_mlr,
        ham_mlr_corrected = ham_mlr_corrected,
        medoid_index      = medoid_sim_result.medoid_index,
        samples           = medoid_sim_result.samples,
        pairwise_distances = medoid_sim_result.pairwise_distances,
        medoid_scores     = medoid_sim_result.medoid_scores,
    )
end


# ---------------------------------------------------------------------------
# run_prediction_pipeline_denoised  (causal / leakage-free denoising)
# ---------------------------------------------------------------------------

"""
_causal_weird_df(data, anchor_day, q; tstep, tsmooth, threshold) -> DataFrame

Restrict `data` to rows with `time <= anchor_day`, compute a frequency
trajectory from ONLY that restricted slice (via `FluStudy.frequency_trajectory`
-- so the Gaussian smoothing kernel never sees anything past `anchor_day`),
then run `find_weird_mutations` on it. This is the causal counterpart to
using a precomputed, whole-dataset `full_frequency` matrix: it never uses
information from beyond `anchor_day`, even accounting for the fact that
`frequency_trajectory`'s smoothing kernel is symmetric (it just never gets
future data to be symmetric around, because that data isn't in its input).

Requires `FluStudy.jl` to be loaded.
"""
function _causal_weird_df(data::Matrix{Float64}, anchor_day::Int, q::Int;
                          tstep::Float64 = 10.0, tsmooth::Float64 = 60.0,
                          threshold::Float64 = 0.01)
    times_int = Int.(data[:, 1])
    position_in_time = findlast(==(anchor_day), times_int)
    position_in_time === nothing &&
        error("_causal_weird_df: no observations exactly at anchor_day = $anchor_day.")

    seqs_upto   = Int.(data[1:position_in_time, 3:end])
    counts_upto = data[1:position_in_time, 2]
    times_upto  = Float64.(times_int[1:position_in_time])

    freq, _ = Main.FluStudy.frequency_trajectory(
        seqs_upto, counts_upto, times_upto; tstep = tstep, tsmooth = tsmooth, q = q)
    L = size(seqs_upto, 2)
    return find_weird_mutations(freq, L, q; threshold = threshold)
end

"""
_denoise_data_upto(data, anchor_day, q; tstep, tsmooth, threshold) -> Matrix{Float64}

A priori correction, restricted to information available up to `anchor_day`:
compute a causal weird-mutation list via `_causal_weird_df`, then apply
`correct_weird_mutations_data` ONLY to the rows of `data` with
`time <= anchor_day` -- rows after `anchor_day` are returned untouched, so
callers building a distribution anchored at a LATER date from the same
`data` are unaffected by this correction.
"""
function _denoise_data_upto(data::Matrix{Float64}, anchor_day::Int, q::Int;
                            tstep::Float64 = 10.0, tsmooth::Float64 = 60.0,
                            threshold::Float64 = 0.01)
    weird_df = _causal_weird_df(data, anchor_day, q; tstep=tstep, tsmooth=tsmooth,
                                threshold=threshold)
    isempty(weird_df) && return copy(data)

    times_int = Int.(data[:, 1])
    position_in_time = findlast(==(anchor_day), times_int)

    data_corrected = copy(data)
    data_corrected[1:position_in_time, :] =
        correct_weird_mutations_data(data[1:position_in_time, :], weird_df, q)
    return data_corrected
end

"""
run_prediction_pipeline_denoised(data, date_, Δ, subtype; q=21, weird_threshold=0.01, kwargs...)

Causal / leakage-free a priori counterpart to `run_prediction_pipeline`.
Unlike the earlier single-global-correction design (which used one
`full_frequency` matrix spanning the whole study, corrected `data` once, and
called the unmodified `run_prediction_pipeline`), this:

  * builds `d1`/`pop1` from `data` denoised using ONLY information up to
    `t1` (`distributions_at_t1_t2_causal`),
  * builds `d2` INDEPENDENTLY, denoised using ONLY information up to `t2`,
  * builds the Null-Emp baseline (`bin_d1`/`bin_d2`) from the raw,
    undenoised observations (`bins_distributions_at_t1_t2`, as in
    run_prediction_emp_main.jl), plus a denoised variant built the same way
    as d1/d2 (`bins_distributions_at_t1_t2_causal`), returned with a
    `_denoised` suffix (`emp_result_denoised`, `result_naive_emp_denoised`,
    `ham_naive_emp_denoised`, ...),
  * fits `h` (`MAPestimate`) and the MLR model on the SAME t1-denoised data
    used for `pop1`/`d1` (both already internally restrict their own
    computation to `<= t1`, so feeding them pre-denoised data is safe),
  * seeds the forward simulation from the denoised `pop1`.

No step here ever sees information from beyond its own anchor date, so
there is nothing left to patch after the fact -- `_corrected` fields are
simply `nothing` throughout, same shape as `run_prediction_pipeline`'s
return value (plus the `_denoised` Null-Emp fields and an extra `data_t1`
field, exposing the t1-denoised data
for reuse, e.g. by `run_prediction_pipeline_ext_denoised`'s `h_nocorr` fit).

Requires `FluStudy.jl` to be loaded (for `frequency_trajectory`).

Purely additive: does not modify `run_prediction_pipeline` or change its
default behaviour.
"""
function run_prediction_pipeline_denoised(
    data::Matrix{Float64},
    date_::Date,
    Δ::Int,
    subtype::String;
    gamma::Float64            = 20.0,
    q::Int                    = 21,
    tstep::Float64            = 10.0,
    tsmooth::Float64          = 60.0,
    noise::Bool               = false,
    N::Int                    = 1000,
    # Posterior spread scale for the h-sampling (noise=true); see the Σ
    # derivation in `simulate_medoid_distribution`. Distinct from `N`,
    # which is the WF resampling population size.
    Ne::Int                   = 500,
    K::Int                    = 1000,
    M::Int                    = 100,
    freq_threshold::Float64   = 1.0e-10,
    medoid_threshold::Float64 = 1.0e-6,
    correlation::Bool         = true,
    verbose::Bool             = true,
    weird_threshold::Float64  = 0.01,
    user_start_date::Union{Date,Nothing} = nothing,
)
    # Resolve t1/t2 and denoise ONCE here, then hand the same two denoised
    # datasets to both `distributions_at_t1_t2_causal` (pop1/d1/d2) and
    # `bins_distributions_at_t1_t2_causal` (bin_d1/bin_d2) below -- avoids
    # running the expensive `_denoise_data_upto` step (a Gaussian-smoothed
    # frequency trajectory over the whole history up to each anchor) twice
    # per anchor for no reason.
    u_times_ = sort!(unique(Int.(data[:, 1])))
    start_date_ =
        subtype == "H3N2"  ? Date(2009, 1, 1) :
        subtype == "H1N1"  ? Date(2009, 5, 1) :
        subtype == "H3N2_2023"  ? Date(2023, 1, 1) :
        error("Unknown subtype: $subtype")
    end_date_ =
        subtype == "H3N2"  ? Date(2020, 3, 1) :
        subtype == "H1N1"  ? Date(2020, 3, 1) :
        subtype == "H3N2_2023"  ? Date(2026, 3, 1) :
        error("Unknown subtype: $subtype")
    target_day_ = Dates.value(date_ - start_date_)
    t1_ = closest_utime(target_day_, u_times_)
    t2_, _ = closest_future_time_and_date(t1_, Δ, u_times_, start_date_, end_date_)

    data_t1_shared = _denoise_data_upto(data, t1_, q; tstep=tstep, tsmooth=tsmooth,
                                        threshold=weird_threshold)
    data_t2_shared = _denoise_data_upto(data, t2_, q; tstep=tstep, tsmooth=tsmooth,
                                        threshold=weird_threshold)

    pop_result = distributions_at_t1_t2_causal(
        data, date_, Δ, subtype;
        tstep=tstep, tsmooth=tsmooth, freq_threshold=freq_threshold,
        q=q, weird_threshold=weird_threshold,
        data_t1=data_t1_shared, data_t2=data_t2_shared)

    map_result = MAPestimate(
        pop_result.data_t1, date_, subtype;
        tstep=tstep, gamma=gamma, tsmooth=tsmooth, q=q,
        correlation=correlation, user_start_date=user_start_date)

    start_day_date = map_result.start_day_date

    mlr_result = MLRmodel(
        pop_result.data_t1, date_, subtype;
        tstep=tstep, gamma=gamma, tsmooth=tsmooth, q=q,
        user_start_date=user_start_date)

    # Null-Emp from the raw observations (as in run_prediction_emp_main.jl),
    # plus its denoised variant from the same t1/t2-denoised data as d1/d2.
    emp_result = bins_distributions_at_t1_t2(
        data, date_, Δ, subtype;
        bin_size=30, freq_threshold=freq_threshold)
    emp_result_denoised = bins_distributions_at_t1_t2_causal(
        data, date_, Δ, subtype;
        bin_size=30, freq_threshold=freq_threshold, q=q,
        weird_threshold=weird_threshold, tstep=tstep, tsmooth=tsmooth,
        data_t1=data_t1_shared, data_t2=data_t2_shared)

    medoid_sim_result = simulate_medoid_distribution(
        pop_result, map_result;
        noise=noise, gamma=gamma, q=q, N=N, Ne=Ne, K=K, M=M,
        freq_threshold=freq_threshold,
        medoid_threshold=medoid_threshold, verbose=verbose)

    # Average across the same M samples used for the medoid (no extra
    # simulation cost -- see `average_distribution` in misc_EMD.jl).
    output_average = average_distribution(medoid_sim_result.samples)

    # MLR prediction
    t_future  = Float64(pop_result.t2)
    out       = predict_both_probs_counts(mlr_result.mlrModel, t_future)
    seq_last  = mlr_result.u_seqs0[out.last.idx, :]
    d_mlr     = sequence_distribution(seq_last, out.last.p)
    d_mlr     = select_distribution_subset(d_mlr, freq_threshold)

    # Consensus sequences
    pop_d1_cons      = consensus_sequence(pop_result.d1)
    pop_d2_cons      = consensus_sequence(pop_result.d2)
    pop_dlb_cons     = consensus_sequence(pop_result.dlb)
    emp_d1_cons      = consensus_sequence(emp_result.bin_d1)
    emp_d2_cons      = consensus_sequence(emp_result.bin_d2)
    emp_d1_cons_denoised = consensus_sequence(emp_result_denoised.bin_d1)
    emp_d2_cons_denoised = consensus_sequence(emp_result_denoised.bin_d2)
    sim_output_cons  = consensus_sequence(medoid_sim_result.output)
    output_average_cons = consensus_sequence(output_average)
    mlr_cons         = consensus_sequence(d_mlr)

    ham_naive      = BioTools.hamming(pop_d1_cons, pop_d2_cons)
    ham_prediction = BioTools.hamming(pop_d2_cons, sim_output_cons)
    ham_average    = BioTools.hamming(pop_d2_cons, output_average_cons)
    ham_lowerbound = BioTools.hamming(pop_dlb_cons, pop_d2_cons)
    ham_naive_emp  = BioTools.hamming(emp_d1_cons, emp_d2_cons)
    ham_naive_emp_denoised = BioTools.hamming(emp_d1_cons_denoised, emp_d2_cons_denoised)
    ham_mlr        = BioTools.hamming(pop_d2_cons, mlr_cons)

    result_prediction = EMD_HiGHS(pop_result.d2, medoid_sim_result.output)
    result_average    = EMD_HiGHS(pop_result.d2, output_average)
    result_naive_emp  = EMD_HiGHS(emp_result.bin_d1, emp_result.bin_d2)
    result_naive_emp_denoised = EMD_HiGHS(emp_result_denoised.bin_d1, emp_result_denoised.bin_d2)
    result_mlr        = EMD_HiGHS(pop_result.d2, d_mlr)
    result_naive      = EMD_HiGHS(pop_result.d1, pop_result.d2)
    result_lb         = EMD_HiGHS(pop_result.dlb, pop_result.d2)

    return (
        pop_result  = pop_result,
        emp_result  = emp_result,
        emp_result_denoised = emp_result_denoised,
        map_result  = map_result,
        sim_result  = medoid_sim_result,
        start_day_date = start_day_date,
        result_naive = result_naive,
        result_naive_emp = result_naive_emp,
        result_naive_emp_denoised = result_naive_emp_denoised,
        result_lb    = result_lb,
        result_prediction = result_prediction,
        result_prediction_corrected = nothing,
        result_average = result_average,
        result_average_corrected = nothing,
        result_naive_corrected      = nothing,
        result_naive_emp_corrected   = nothing,
        result_lb_corrected         = nothing,
        output           = medoid_sim_result.output,
        output_corrected = nothing,
        output_average           = output_average,
        output_average_corrected = nothing,
        d1_corrected  = nothing,
        d2_corrected  = nothing,
        dlb_corrected = nothing,
        pop_d1_cons  = pop_d1_cons,
        pop_d2_cons  = pop_d2_cons,
        pop_dlb_cons = pop_dlb_cons,
        emp_d1_cons  = emp_d1_cons,
        emp_d2_cons  = emp_d2_cons,
        emp_d1_cons_denoised = emp_d1_cons_denoised,
        emp_d2_cons_denoised = emp_d2_cons_denoised,
        sim_output_cons = sim_output_cons,
        output_average_cons = output_average_cons,
        pop_d1_cons_corrected  = nothing,
        pop_d2_cons_corrected  = nothing,
        pop_dlb_cons_corrected = nothing,
        emp_d1_cons_corrected  = nothing,
        emp_d2_cons_corrected  = nothing,
        output_cons_corrected  = nothing,
        output_average_cons_corrected = nothing,
        ham_naive      = ham_naive,
        ham_prediction = ham_prediction,
        ham_average    = ham_average,
        ham_lowerbound = ham_lowerbound,
        ham_naive_emp  = ham_naive_emp,
        ham_naive_emp_denoised = ham_naive_emp_denoised,
        ham_naive_corrected      = nothing,
        ham_prediction_corrected = nothing,
        ham_average_corrected    = nothing,
        ham_lowerbound_corrected = nothing,
        ham_naive_emp_corrected  = nothing,
        mlr_result       = mlr_result,
        d_mlr            = d_mlr,
        d_mlr_corrected  = nothing,
        result_mlr       = result_mlr,
        result_mlr_corrected = nothing,
        mlr_cons         = mlr_cons,
        mlr_cons_corrected = nothing,
        ham_mlr          = ham_mlr,
        ham_mlr_corrected = nothing,
        medoid_index      = medoid_sim_result.medoid_index,
        samples           = medoid_sim_result.samples,
        pairwise_distances = medoid_sim_result.pairwise_distances,
        medoid_scores     = medoid_sim_result.medoid_scores,
        data_t1           = pop_result.data_t1,
    )
end

"""
run_prediction_pipeline_short_denoised(data, date_, Δ, subtype; q=21, weird_threshold=0.01, kwargs...)

Causal a priori counterpart to `run_prediction_pipeline_short` -- see
`run_prediction_pipeline_denoised`.
"""
function run_prediction_pipeline_short_denoised(
    data::Matrix{Float64},
    date_::Date,
    Δ::Int,
    subtype::String;
    gamma::Float64            = 20.0,
    q::Int                    = 21,
    tstep::Float64            = 10.0,
    tsmooth::Float64          = 60.0,
    noise::Bool               = false,
    N::Int                    = 1000,
    # Posterior spread scale for the h-sampling (noise=true); see the Σ
    # derivation in `simulate_medoid_distribution`. Distinct from `N`,
    # which is the WF resampling population size.
    Ne::Int                   = 500,
    K::Int                    = 1000,
    M::Int                    = 100,
    freq_threshold::Float64   = 1.0e-10,
    medoid_threshold::Float64 = 1.0e-6,
    correlation::Bool         = true,
    verbose::Bool             = true,
    weird_threshold::Float64  = 0.01,
    user_start_date::Union{Date,Nothing} = nothing,
)
    res = run_prediction_pipeline_denoised(
        data, date_, Δ, subtype;
        gamma=gamma, q=q, tstep=tstep, tsmooth=tsmooth,
        noise=noise, N=N, Ne=Ne, K=K, M=M,
        freq_threshold=freq_threshold,
        medoid_threshold=medoid_threshold,
        correlation=correlation, verbose=verbose,
        weird_threshold=weird_threshold,
        user_start_date=user_start_date)

    return (
        naive              = res.result_naive,
        lower_bound        = res.result_lb,
        prediction         = res.result_prediction,
        prediction_corrected = res.result_prediction_corrected,
        average            = res.result_average,
        average_corrected  = res.result_average_corrected,
        naive_corrected    = res.result_naive_corrected,
        lower_bound_corrected = res.result_lb_corrected,
        naive_emp          = res.result_naive_emp,
        naive_emp_corrected = res.result_naive_emp_corrected,
        naive_emp_denoised = res.result_naive_emp_denoised,
        output             = res.output,
        output_corrected   = res.output_corrected,
        output_average           = res.output_average,
        output_average_corrected = res.output_average_corrected,
        start_day_date     = res.start_day_date,
        date1              = res.pop_result.date1,
        date2              = res.pop_result.date2,
        ham_naive          = res.ham_naive,
        ham_prediction     = res.ham_prediction,
        ham_average        = res.ham_average,
        ham_lowerbound     = res.ham_lowerbound,
        ham_naive_emp      = res.ham_naive_emp,
        ham_naive_emp_denoised = res.ham_naive_emp_denoised,
        ham_naive_corrected      = res.ham_naive_corrected,
        ham_prediction_corrected = res.ham_prediction_corrected,
        ham_average_corrected    = res.ham_average_corrected,
        ham_lowerbound_corrected = res.ham_lowerbound_corrected,
        ham_naive_emp_corrected  = res.ham_naive_emp_corrected,
        mlr                = res.result_mlr,
        mlr_corrected      = res.result_mlr_corrected,
        ham_mlr            = res.ham_mlr,
        ham_mlr_corrected  = res.ham_mlr_corrected,
    )
end


# ---------------------------------------------------------------------------
# run_prediction_pipeline_short  (slim wrapper — primary entry point)
# ---------------------------------------------------------------------------

"""
run_prediction_pipeline_short(data, date_, Δ, subtype; ...)

Thin wrapper around run_prediction_pipeline that returns only the distances,
distributions, and Hamming scores needed for downstream reporting.
"""
function run_prediction_pipeline_short(
    data::Matrix{Float64},
    date_::Date,
    Δ::Int,
    subtype::String;
    gamma::Float64            = 20.0,
    q::Int                    = 21,
    tstep::Float64            = 10.0,
    tsmooth::Float64          = 60.0,
    noise::Bool               = false,
    N::Int                    = 1000,
    # Posterior spread scale for the h-sampling (noise=true); see the Σ
    # derivation in `simulate_medoid_distribution`. Distinct from `N`,
    # which is the WF resampling population size.
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
    res = run_prediction_pipeline(
        data, date_, Δ, subtype;
        gamma=gamma, q=q, tstep=tstep, tsmooth=tsmooth,
        noise=noise, N=N, Ne=Ne, K=K, M=M,
        freq_threshold=freq_threshold,
        medoid_threshold=medoid_threshold,
        correlation=correlation, verbose=verbose,
        full_frequency=full_frequency,
        user_start_date=user_start_date)

    return (
        naive              = res.result_naive,
        lower_bound        = res.result_lb,
        prediction         = res.result_prediction,
        prediction_corrected = res.result_prediction_corrected,
        average            = res.result_average,
        average_corrected  = res.result_average_corrected,
        naive_corrected    = res.result_naive_corrected,
        lower_bound_corrected = res.result_lb_corrected,
        naive_emp          = res.result_naive_emp,
        naive_emp_corrected = res.result_naive_emp_corrected,
        output             = res.output,
        output_corrected   = res.output_corrected,
        output_average           = res.output_average,
        output_average_corrected = res.output_average_corrected,
        start_day_date     = res.start_day_date,
        date1              = res.pop_result.date1,
        date2              = res.pop_result.date2,
        ham_naive          = res.ham_naive,
        ham_prediction     = res.ham_prediction,
        ham_average        = res.ham_average,
        ham_lowerbound     = res.ham_lowerbound,
        ham_naive_emp      = res.ham_naive_emp,
        ham_naive_corrected      = res.ham_naive_corrected,
        ham_prediction_corrected = res.ham_prediction_corrected,
        ham_average_corrected    = res.ham_average_corrected,
        ham_lowerbound_corrected = res.ham_lowerbound_corrected,
        ham_naive_emp_corrected  = res.ham_naive_emp_corrected,
        mlr                = res.result_mlr,
        mlr_corrected      = res.result_mlr_corrected,
        ham_mlr            = res.ham_mlr,
        ham_mlr_corrected  = res.ham_mlr_corrected,
    )
end


# ---------------------------------------------------------------------------
# distance_summary_table_short  (reporting helper)
# ---------------------------------------------------------------------------

"""
distance_summary_table_short(result; digits)

Format the distances returned by run_prediction_pipeline_short into a
sorted DataFrame.
"""
function distance_summary_table_short(result; digits=4)
    df = DataFrame(
        "StartDate" => fill(result.start_day_date, 6),
        "ObsDate" => fill(result.date1, 6),
        "PredDate" => fill(result.date2, 6),
        "Method" => ["Null", "GenModel", "GenModel-Average", "Lower Bound", "Null-Emp", "MLR"],
        "EMD(Pop)" => round.(
            [result.naive.emd, result.prediction.emd, result.average.emd, result.lower_bound.emd,
             result.naive_emp.emd, result.mlr.emd],
            digits=digits),
        "EMD(Pop-filt.)" => round.(
            [isnothing(result.naive_corrected)        ? result.naive.emd        : result.naive_corrected.emd,
             isnothing(result.prediction_corrected)   ? result.prediction.emd   : result.prediction_corrected.emd,
             isnothing(result.average_corrected)      ? result.average.emd      : result.average_corrected.emd,
             isnothing(result.lower_bound_corrected)  ? result.lower_bound.emd  : result.lower_bound_corrected.emd,
             isnothing(result.naive_emp_corrected)    ? result.naive_emp.emd    : result.naive_emp_corrected.emd,
             isnothing(result.mlr_corrected)          ? result.mlr.emd          : result.mlr_corrected.emd],
            digits=digits),
        "Hamming(ConsSeq)" => [result.ham_naive, result.ham_prediction, result.ham_average, result.ham_lowerbound,
                   result.ham_naive_emp, result.ham_mlr],
        "Hamming(ConsSeq-filt.)" => [
            isnothing(result.ham_naive_corrected)      ? result.ham_naive      : result.ham_naive_corrected,
            isnothing(result.ham_prediction_corrected) ? result.ham_prediction : result.ham_prediction_corrected,
            isnothing(result.ham_average_corrected)    ? result.ham_average    : result.ham_average_corrected,
            isnothing(result.ham_lowerbound_corrected) ? result.ham_lowerbound : result.ham_lowerbound_corrected,
            isnothing(result.ham_naive_emp_corrected)  ? result.ham_naive_emp  : result.ham_naive_emp_corrected,
            isnothing(result.ham_mlr_corrected)        ? result.ham_mlr        : result.ham_mlr_corrected],
    )
    if hasproperty(result, :naive_emp_denoised)   # denoised pipelines only
        emd = round(result.naive_emp_denoised.emd, digits=digits)
        push!(df, (result.start_day_date, result.date1, result.date2, "Null-Emp-Denoised",
                   emd, emd, result.ham_naive_emp_denoised, result.ham_naive_emp_denoised))
    end

    sort!(df, "EMD(Pop-filt.)")
    return df
end
