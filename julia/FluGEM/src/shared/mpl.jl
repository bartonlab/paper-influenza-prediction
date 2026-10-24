using LinearAlgebra
using Statistics
using Dates

# ---------------------------------------------------------------------------
# Covariance / mutation update helpers
# ---------------------------------------------------------------------------

"""
updateCovarianceIntegrate(dg, p1_0, p2_0, p1_1, p2_1)

Add the piecewise-linear integrated covariance contribution for one interval.
"""
function updateCovarianceIntegrate(dg::Float64, p1_0::Array{Float64,1}, p2_0::Array{Float64,2},
                                   p1_1::Array{Float64,1}, p2_1::Array{Float64,2})
    return -dg .* ((2 .* p1_0 * p1_0') .+ (2 .* p1_1 * p1_1') .+
                   (p1_0 * p1_1') .+ (p1_1 * p1_0')) ./ 6 .+
           dg .* 0.5 .* (p2_0 .+ p2_1)
end

"""
    updateCovarianceIntegrate!(C, dg, p1_0, p2_0, p1_1, p2_1)

Add the same matrix as `updateCovarianceIntegrate` in-place to `C`.
"""
function updateCovarianceIntegrate!(
    C::Matrix{Float64},
    dg::Float64,
    p1_0::Vector{Float64},
    p2_0::Matrix{Float64},
    p1_1::Vector{Float64},
    p2_1::Matrix{Float64},
)
    n = size(C, 1)
    α = -dg / 6
    β = dg * 0.5
    @inbounds for j in 1:n
        p0j = p1_0[j]
        p1j = p1_1[j]
        for i in 1:n
            C[i, j] += α * (2 * p1_0[i] * p0j + 2 * p1_1[i] * p1j +
                            p1_0[i] * p1j + p1_1[i] * p0j)
        end
    end
    @inbounds for ij in eachindex(C, p2_0, p2_1)
        C[ij] += β * (p2_0[ij] + p2_1[ij])
    end
    return C
end

"""
updateMuIntegrate(dg, mu, p1_0, p1_1)

Add the integrated mutation flux contribution for one interval.
"""
function updateMuIntegrate(dg::Float64, mu::Float64, p1_0::Array{Float64,1}, p1_1::Array{Float64,1})
    return -mu * dg .* (1 .- (p1_0 .+ p1_1))
end


# ---------------------------------------------------------------------------
# MPL inference
# ---------------------------------------------------------------------------

"""
processStandard(seqs, counts, times, mu, L, q, gamma)

Compute MPL selection coefficients from smoothed effective counts
(Vector{Vector{Float64}} variant).
"""
function processStandard(
    seqs::Matrix{Float64},
    counts::Vector{Vector{Float64}},
    times::Vector{Float64},
    mu::Float64,
    L::Int64,
    q::Int64,
    gamma::Float64,
)
    n = q * L
    lastp1 = Vector{Float64}(undef, n)
    lastp2 = Matrix{Float64}(undef, n, n)
    p1 = Vector{Float64}(undef, n)
    p2 = Matrix{Float64}(undef, n, n)
    comp_freqs!(lastp1, lastp2, seqs, counts[1], Int(q))

    dx = -lastp1
    totalCov = zeros(n, n)
    totalMu  = zeros(n)

    for k in 2:length(times)
        comp_freqs!(p1, p2, seqs, counts[k], Int(q))
        dt = times[k] - times[k - 1]
        updateCovarianceIntegrate!(totalCov, dt, lastp1, lastp2, p1, p2)
        if mu != 0.0
            c = -mu * dt
            @. totalMu += c * (1 - (lastp1 + p1))
        end

        if k == length(times)
            dx .+= p1
        else
            lastp1 .= p1
            lastp2 .= p2
        end
    end

    dx .+= totalMu
    @inbounds for i in 1:n
        totalCov[i, i] += gamma
    end
    return totalCov \ dx
end


"""
processStandardNoCorr(seqs, counts, times, mu, L, q, gamma)

Compute SL (single-locus) selection coefficients: same as processStandard
but zeroes out off-diagonal covariance entries.
"""
function processStandardNoCorr(
    seqs::Matrix{Float64},
    counts::Vector{Vector{Float64}},
    times::Vector{Float64},
    mu::Float64,
    L::Int64,
    q::Int64,
    gamma::Float64,
)
    n = q * L
    lastp1 = Vector{Float64}(undef, n)
    lastp2 = Matrix{Float64}(undef, n, n)
    p1 = Vector{Float64}(undef, n)
    p2 = Matrix{Float64}(undef, n, n)
    comp_freqs!(lastp1, lastp2, seqs, counts[1], Int(q))

    dx = -lastp1
    diag_cov = zeros(Float64, n)
    totalMu  = zeros(Float64, n)

    for k in 2:length(times)
        comp_freqs!(p1, p2, seqs, counts[k], Int(q))
        dt = times[k] - times[k - 1]
        c1 = -dt / 6
        c2 = dt * 0.5
        μdt = mu * dt

        @inbounds for i in 1:n
            a0 = lastp1[i]
            a1 = p1[i]
            diag_cov[i] += c1 * (2 * a0 * a0 + 2 * a1 * a1 + 2 * a0 * a1) +
                           c2 * (lastp2[i, i] + p2[i, i])
            totalMu[i] += -μdt * (1 - (a0 + a1))
        end

        if k == length(times)
            dx .+= p1
        else
            lastp1 .= p1
            lastp2 .= p2
        end
    end

    dx .+= totalMu
    @. dx = dx / (diag_cov + gamma)
    return dx
end

"""
diagonal_matrix(A)

Return a matrix with the same diagonal as A and zeros elsewhere.
"""
function diagonal_matrix(A::Matrix{Float64})
    n, m = size(A)
    B = zeros(n, m)
    for i in 1:min(n, m)
        B[i, i] = A[i, i]
    end
    return B
end


"""
Cintegrate(seqs, counts, times, L, q, gamma)

Compute the integrated (regularised) covariance matrix for active-mutation
noise estimation.
"""
function _cintegrate_core!(
    seqs::Matrix{Float64},
    counts::Vector{Vector{Float64}},
    times::AbstractVector{<:Real},
    L::Int,
    q::Int,
    gamma::Real,
)
    n = q * L
    lastp1 = Vector{Float64}(undef, n)
    lastp2 = Matrix{Float64}(undef, n, n)
    p1 = Vector{Float64}(undef, n)
    p2 = Matrix{Float64}(undef, n, n)
    comp_freqs!(lastp1, lastp2, seqs, counts[1], Int(q))

    totalCov = zeros(n, n)

    for k in 2:length(times)
        comp_freqs!(p1, p2, seqs, counts[k], Int(q))
        dt = Float64(times[k] - times[k - 1])
        updateCovarianceIntegrate!(totalCov, dt, lastp1, lastp2, p1, p2)
        if k != length(times)
            lastp1 .= p1
            lastp2 .= p2
        end
    end

    g = Float64(gamma)
    @inbounds for i in 1:n
        totalCov[i, i] += g
    end
    return totalCov
end

function Cintegrate(
    seqs::Matrix{Float64},
    counts::Vector{Vector{Float64}},
    times::AbstractVector{<:Real},
    L::Int,
    q::Int,
    gamma::Real,
)
    return _cintegrate_core!(seqs, counts, times, L, q, gamma)
end

function Cintegrate(
    seqs::AbstractMatrix{<:Real},
    counts::Vector{<:AbstractVector{<:Real}},
    times::AbstractVector{<:Real},
    L::Int,
    q::Int,
    gamma::Real,
)
    seqs_f = Matrix{Float64}(seqs)
    counts_f = [Vector{Float64}(c) for c in counts]
    return _cintegrate_core!(seqs_f, counts_f, times, L, q, gamma)
end


"""
normalize_sc(h, ref_seq, q, L)

Subtract the reference-allele selection coefficient at each site so
WT alleles have sc = 0.
"""
function normalize_sc(h::Vector{Float64}, ref_seq::Vector{Int}, q::Int, L::Int)
    selec_coeff = copy(h)
    for i in 1:L
        href = h[(i-1)*q + ref_seq[i]]
        for j in 1:q
            selec_coeff[(i-1)*q + j] -= href
        end
    end
    return selec_coeff
end


# ---------------------------------------------------------------------------
# High-level wrappers called by run_prediction_pipeline
# ---------------------------------------------------------------------------

"""
MAPestimate(data, date, subtype; ...)

Fit MPL selection coefficients from sequence data up to `date`.
Returns h, frequency paths, smoothed counts, and time metadata.
"""
function MAPestimate(
    data::Matrix{Float64},
    date::Date,
    subtype::String;
    tstep = 10.0,
    gamma = 20.0,
    tsmooth = 15.0,
    q = 21,
    correlation = true,
    user_start_date::Union{Date,Nothing} = nothing,
)
    times  = Int.(data[:, 1])
    counts = data[:, 2]
    seqs   = data[:, 3:end]

    u_times = sort!(unique(times))
    L = size(seqs, 2)

    base_start_date =
        subtype == "H3N2"  ? Date(2009, 1, 1) :
        subtype == "H1N1"  ? Date(2009, 5, 1) :
        error("Unknown subtype: $subtype")

    target_day = Dates.value(date - base_start_date)
    t_c = closest_utime(target_day, u_times)
    println("Target day: ", target_day)
    println("Closest utime: ", t_c)

    position_in_time_T0 = findlast(==(t_c), times)
    println("Position in time for T0: ", position_in_time_T0)
    if user_start_date === nothing
        start_day = minimum(times)
        start_day_date = base_start_date + Day(start_day)
    else
        start_day = Dates.value(user_start_date - base_start_date)
        start_day = max(minimum(times), min(start_day, t_c))
        start_day_date = base_start_date + Day(start_day)
    end

    start_idx = findfirst(x -> x >= start_day, times)
    if start_idx === nothing
        start_idx = 1
    end

    if position_in_time_T0 < start_idx
        error("t_c ($t_c) occurs before computed start day ($start_day)")
    end

    counts_obs      = counts[start_idx:position_in_time_T0]
    times_obs       = Float64.(times[start_idx:position_in_time_T0])
    seqs_observed   = seqs[start_idx:position_in_time_T0, :]

    normalized_counts_obs = normalizeCounts(times_obs, counts_obs)

    times_measure = collect(times_obs[1]:tstep:Float64(t_c))
    if times_measure[end] != t_c
        push!(times_measure, Float64(t_c))
    end

    effectiveCounts_list = Vector{Vector{Float64}}(undef, length(times_measure))
    for (i, t_) in enumerate(times_measure)
        effectiveCounts_list[i] = smoothCounts_normalized(
            normalized_counts_obs, times_obs, t_, tsmooth)
    end

    observed_path_freq = compute_path_f_i(seqs_observed, effectiveCounts_list, q)

    fr_t0 = observed_path_freq[:, 1]
    f_mat = reshape(fr_t0, q, L)'
    ref_s = [argmax(@view f_mat[i, :]) for i in 1:L]

    mu = 0.0
    h = correlation ?
        processStandard(seqs_observed, effectiveCounts_list, times_measure, mu, L, q, gamma) :
        processStandardNoCorr(seqs_observed, effectiveCounts_list, times_measure, mu, L, q, gamma)

    h = normalize_sc(h, ref_s, q, L)

    return (
        h                    = h,
        observed_path_freq   = observed_path_freq,
        ref_s                = ref_s,
        seqs_observed        = seqs_observed,
        effectiveCounts_list = effectiveCounts_list,
        times_measure        = times_measure,
        t_c                  = t_c,
        start_day            = start_day,
        start_day_date       = start_day_date,
    )
end


"""
MLRmodel(data, date, subtype; ...)

Fit a multinomial logistic regression (MLR) model from smoothed variant counts
up to `date`. Returns the fitted model and unique sequences.
"""
function MLRmodel(
    data::Matrix{Float64},
    date::Date,
    subtype::String;
    tstep = 10.0,
    gamma = 20.0,
    tsmooth = 15.0,
    q = 21,
    user_start_date::Union{Date,Nothing} = nothing,
)
    times  = Int.(data[:, 1])
    counts = data[:, 2]
    seqs   = data[:, 3:end]

    u_times = sort!(unique(times))
    L = size(seqs, 2)

    base_start_date =
        subtype == "H3N2"  ? Date(2009, 1, 1) :
        subtype == "H1N1"  ? Date(2009, 5, 1) :
        error("Unknown subtype: $subtype")

    target_day = Dates.value(date - base_start_date)
    t_c = closest_utime(target_day, u_times)
    println("Target day: ", target_day)
    println("Closest utime: ", t_c)
    position_in_time_T0 = findlast(==(t_c), times)
    println("Position in time for T0: ", position_in_time_T0)
    if user_start_date === nothing
        start_day = minimum(times)
        start_day_date = base_start_date + Day(start_day)
    else
        start_day = Dates.value(user_start_date - base_start_date)
        start_day = max(minimum(times), min(start_day, t_c))
        start_day_date = base_start_date + Day(start_day)
    end

    start_idx = findfirst(x -> x >= start_day, times)
    if start_idx === nothing
        start_idx = 1
    end

    if position_in_time_T0 < start_idx
        error("t_c ($t_c) occurs before computed start day ($start_day)")
    end

    counts_obs    = counts[start_idx:position_in_time_T0]
    times_obs     = Float64.(times[start_idx:position_in_time_T0])
    seqs_observed = seqs[start_idx:position_in_time_T0, :]

    (u_seqs0, _) = unique_sequences(seqs_observed, ones(Float64, size(seqs_observed, 1)))
    K_var   = size(u_seqs0, 1)
    row2var = build_row2var(seqs_observed, u_seqs0)

    times_measure = collect(times_obs[1]:tstep:Float64(t_c))
    if times_measure[end] != t_c
        push!(times_measure, Float64(t_c))
    end

    effectiveCounts_list = Vector{Vector{Float64}}(undef, length(times_measure))
    for (i, t_) in enumerate(times_measure)
        effectiveCounts_list[i] = smooth_to_variants(
            counts_obs, times_obs, row2var, K_var, t_, tsmooth)
    end

    modelS = fit_mlr_from_smoothed(times_measure, effectiveCounts_list, u_seqs0;
                                   λ=0.1, ref=1, maxiter=20000)

    return (
        mlrModel = modelS,
        u_seqs0  = u_seqs0,
    )
end
