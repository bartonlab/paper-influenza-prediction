using LinearAlgebra
using Statistics
using Optim
using LineSearches

# ---------------------------------------------------------------------------
# Variant-ID helpers
# ---------------------------------------------------------------------------

"""
seq_to_u16(seq_row)

Convert a sequence row to a compact Vector{UInt16}.
"""
function seq_to_u16(seq_row::AbstractVector{<:Real})
    v = Vector{UInt16}(undef, length(seq_row))
    @inbounds for i in eachindex(seq_row)
        v[i] = UInt16(round(Int, seq_row[i]))
    end
    return v
end

"""
build_row2var(seqs_obs, u_seqs)

Map each row of `seqs_obs` to the index of its corresponding row in `u_seqs`.

Uses `Vector{UInt16}` keys instead of `NTuple{L,UInt16}` so large `L` (e.g. full
protein length) does not trigger compiler inference on huge tuple types / `Val{L}`.
"""
function build_row2var(
    seqs_obs::AbstractMatrix{<:Real},
    u_seqs::AbstractMatrix{<:Real},
)
    size(seqs_obs, 2) == size(u_seqs, 2) ||
        throw(DimensionMismatch("seqs_obs and u_seqs must have the same number of columns"))

    seq2k = Dict{Vector{UInt16}, Int}()
    sizehint!(seq2k, size(u_seqs, 1))
    @inbounds for k in 1:size(u_seqs, 1)
        key = seq_to_u16(@view u_seqs[k, :])
        seq2k[key] = k
    end

    row2var = Vector{Int}(undef, size(seqs_obs, 1))
    @inbounds for r in 1:size(seqs_obs, 1)
        key = seq_to_u16(@view seqs_obs[r, :])
        k = get(seq2k, key, 0)
        @assert k != 0 "Row $r sequence not found in unique set (u_seqs)."
        row2var[r] = k
    end
    return row2var
end

"""
smooth_to_variants(counts0, times0, row2var, Kvar, t_, tsmooth_)

Apply Gaussian smoothing and aggregate the resulting weights by variant ID.
"""
function smooth_to_variants(counts0, times0,
                            row2var::Vector{Int},
                            Kvar::Int,
                            t_::Real,
                            tsmooth_::Real)
    w = smoothCounts(counts0, times0, float(t_), float(tsmooth_))
    y = zeros(Float64, Kvar)
    @inbounds for r in eachindex(w)
        y[row2var[r]] += w[r]
    end
    return y
end


# ---------------------------------------------------------------------------
# MLR core: softmax, NLL + analytic gradient
# ---------------------------------------------------------------------------

"""
softmax_stable!(p, η)

In-place numerically stable softmax.
"""
function softmax_stable!(p::Vector{Float64}, η::Vector{Float64})
    m = maximum(η)
    @inbounds for i in eachindex(η)
        p[i] = exp(η[i] - m)
    end
    s = sum(p)
    invs = 1.0 / s
    @inbounds for i in eachindex(p)
        p[i] *= invs
    end
    return p
end

"Pack α, β (excluding reference class) into a single parameter vector."
function pack_params(α::Vector{Float64}, β::Vector{Float64}, ref::Int)
    K  = length(α)
    θ  = Vector{Float64}(undef, 2*(K-1))
    ii = 1
    @inbounds for k in 1:K
        k == ref && continue
        θ[ii] = α[k]; ii += 1
    end
    @inbounds for k in 1:K
        k == ref && continue
        θ[ii] = β[k]; ii += 1
    end
    return θ
end

"Unpack θ into α, β with α[ref] = β[ref] = 0."
function unpack_params(θ::Vector{Float64}, K::Int, ref::Int)
    α  = zeros(Float64, K)
    β  = zeros(Float64, K)
    ii = 1
    @inbounds for k in 1:K
        k == ref && continue
        α[k] = θ[ii]; ii += 1
    end
    @inbounds for k in 1:K
        k == ref && continue
        β[k] = θ[ii]; ii += 1
    end
    return α, β
end

"""
nll_and_grad!(G, θ, x, Y; ref, λ, p_floor)

Multinomial negative log-likelihood and its analytic gradient.
"""
function nll_and_grad!(
    G::Vector{Float64}, θ::Vector{Float64},
    x::Vector{Float64}, Y::Matrix{Float64};
    ref::Int, λ::Float64=0.1, p_floor::Float64=1e-15
)
    Tn, K = size(Y)
    α, β  = unpack_params(θ, K, ref)

    η  = Vector{Float64}(undef, K)
    p  = Vector{Float64}(undef, K)
    gα = zeros(Float64, K)
    gβ = zeros(Float64, K)

    nll = 0.0

    @inbounds for t in 1:Tn
        xt = x[t]
        for k in 1:K; η[k] = α[k] + β[k]*xt; end
        softmax_stable!(p, η)

        s2 = 0.0
        for k in 1:K
            pk = p[k] < p_floor ? p_floor : p[k]
            p[k] = pk; s2 += pk
        end
        invs2 = 1.0 / s2
        for k in 1:K; p[k] *= invs2; end

        Nt = sum(@view Y[t, :])
        Nt == 0.0 && continue

        for k in 1:K
            y = Y[t, k]
            y > 0 && (nll -= y * log(p[k]))
            r = y - Nt * p[k]
            gα[k] -= r
            gβ[k] -= xt * r
        end
    end

    # Ridge penalty on deviations of the slopes from their mean,
    # λ Σ_k (β_k − β̄)², with β̄ computed from the current slopes (β[ref] = 0
    # included). It is unchanged when the same constant is added to every
    # slope, so the fit does not depend on the choice of `ref`. The gradient
    # is 2λ(β_k − β̄): the term through β̄ vanishes because Σ_k (β_k − β̄) = 0.
    if λ > 0
        β̄ = sum(β) / K
        for k in 1:K
            d = β[k] - β̄
            nll += λ * d^2
            k == ref && continue
            gβ[k] += 2λ * d
        end
    end

    ii = 1
    for k in 1:K; k == ref && continue; G[ii] = gα[k]; ii += 1; end
    for k in 1:K; k == ref && continue; G[ii] = gβ[k]; ii += 1; end

    return nll
end


# ---------------------------------------------------------------------------
# CountMLRModel struct + prediction
# ---------------------------------------------------------------------------

"""Fitted MLR model over all K variants."""
struct CountMLRModel
    α::Vector{Float64}
    β::Vector{Float64}
    ref::Int
    t_mean::Float64
    t_scale::Float64
    t_unique::Vector{Float64}
    last_idx::Vector{Int}
    unique_seqs::Vector{Vector{UInt16}}
end

"""
predict_both_probs_counts(model, t_future; N_future, p_floor)

Return full (all-K) and last-observed-set probabilities (and optional counts)
at t_future.
"""
function predict_both_probs_counts(
    model::CountMLRModel,
    t_future::Float64;
    N_future::Union{Nothing,Float64}=nothing,
    p_floor::Float64=1e-15
)
    K = length(model.α)
    x = (t_future - model.t_mean) / model.t_scale

    η = Vector{Float64}(undef, K)
    p = Vector{Float64}(undef, K)

    @inbounds for k in 1:K; η[k] = model.α[k] + model.β[k]*x; end
    softmax_stable!(p, η)

    s2 = 0.0
    @inbounds for k in 1:K
        pk = p[k] < p_floor ? p_floor : p[k]
        p[k] = pk; s2 += pk
    end
    p ./= s2

    idx_last = model.last_idx
    p_last_un = p[idx_last]
    mass_last_full = sum(p_last_un)

    p_last = (mass_last_full > 0) ?
             (p_last_un ./ mass_last_full) :
             fill(1.0/length(idx_last), length(idx_last))

    if isnothing(N_future)
        return (
            full           = (p=p,),
            last           = (idx=idx_last, p=p_last),
            mass_last_full = mass_last_full,
            t_future       = t_future
        )
    else
        return (
            full           = (p=p, counts=N_future .* p),
            last           = (idx=idx_last, p=p_last, counts=N_future .* p_last),
            mass_last_full = mass_last_full,
            t_future       = t_future
        )
    end
end


# ---------------------------------------------------------------------------
# Fitting from smoothed effective counts
# ---------------------------------------------------------------------------

"""
fit_mlr_from_smoothed(times_measure, effectiveCounts_list, seqs_observed; λ, ref, maxiter)

Fit a CountMLRModel from pre-computed smoothed effective counts per variant.
"""
function fit_mlr_from_smoothed(
    times_measure::Vector{Float64},
    effectiveCounts_list::Vector{Vector{Float64}},
    seqs_observed;
    λ::Float64 = 0.1,
    t_scale::Float64 = 10.0,
    ref::Int = 1,
    maxiter::Int = 500,
    tol_last::Float64 = 0.0
)
    println(">>> ENTER fit_mlr_from_smoothed")
    flush(stdout)

    Tn = length(times_measure)
    @assert length(effectiveCounts_list) == Tn

    K = length(effectiveCounts_list[1])
    @assert 1 <= ref <= K
    for t in 1:Tn
        @assert length(effectiveCounts_list[t]) == K
    end

    Y = Matrix{Float64}(undef, Tn, K)
    @inbounds for t in 1:Tn
        yt = effectiveCounts_list[t]
        for k in 1:K; Y[t,k] = yt[k]; end
    end

    last_idx = findall(>(tol_last), @view Y[end, :])

    t_mean  = mean(times_measure)
    t_scale = std(times_measure) + eps()
    x = (times_measure .- t_mean) ./ t_scale

    α0 = zeros(Float64, K)
    β0 = zeros(Float64, K)
    θ0 = pack_params(α0, β0, ref)

    Y = max.(Y, 1e-12)

    function f(θ)
        Gtmp = zeros(Float64, length(θ))
        return nll_and_grad!(Gtmp, θ, x, Y; ref=ref, λ=λ)
    end
    function g!(G, θ)
        nll_and_grad!(G, θ, x, Y; ref=ref, λ=λ)
        return G
    end

    od     = Optim.OnceDifferentiable(f, g!, θ0; inplace=true)
    method = Optim.LBFGS(linesearch=LineSearches.BackTracking(order=3))
    opts   = Optim.Options(iterations=maxiter, g_tol=1e-6,
                           show_trace=true, show_every=100)
    res    = Optim.optimize(od, θ0, method, opts)

    iters = Optim.iterations(res)
    if Optim.converged(res)
        println("MLR converged | iters=$(iters) | f=$(Optim.minimum(res))")
    elseif iters >= maxiter
        println("MLR did NOT converge (hit maxiter) | iters=$(iters) | f=$(Optim.minimum(res))")
    else
        println("MLR did NOT converge (stopped early) | iters=$(iters) | f=$(Optim.minimum(res))")
    end

    G       = zeros(length(θ0))
    f_final = nll_and_grad!(G, Optim.minimizer(res), x, Y; ref=ref, λ=λ)
    println("Final diagnostics f=$(f_final) gnorm=$(norm(G, Inf)) g2=$(norm(G)) iters=$(iters)")

    θ̂ = Optim.minimizer(res)
    α̂, β̂ = unpack_params(θ̂, K, ref)

    L_seq = size(seqs_observed, 2)
    unique_seqs = Vector{Vector{UInt16}}(undef, K)
    @inbounds for k in 1:K
        v = Vector{UInt16}(undef, L_seq)
        for j in 1:L_seq
            v[j] = UInt16(round(Int, seqs_observed[k, j]))
        end
        unique_seqs[k] = v
    end

    return CountMLRModel(α̂, β̂, ref, t_mean, t_scale, times_measure, last_idx, unique_seqs)
end









