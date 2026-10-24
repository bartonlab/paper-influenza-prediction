using Distributions
using LinearAlgebra

# ---------------------------------------------------------------------------
# Single-site frequency computation
# ---------------------------------------------------------------------------

"""
compute_f1(Y, counts, q)

Compute single-site frequencies from a sequence matrix and float counts.
"""


function compute_f1(
    Y::AbstractMatrix{<:Real},
    counts::AbstractVector{<:Real};
    q::Int=21,
)
    M, L = size(Y)
    f1 = zeros(Float64, L * q)
    N = sum(counts)

    for m in 1:M
        cm = counts[m]
        if cm > 0
            for j in 1:L
                f1[(j - 1) * q + Int(Y[m, j])] += cm
            end
        end
    end

    f1 ./= N
    return f1
end


"""
comp_freqs(Y, counts, q; pc)

Compute single-site (f1) and pairwise (f2) allele frequencies.
Supports Int64 and Float64 counts.
"""
function comp_freqs(
    Y::AbstractMatrix{<:Real},
    counts::AbstractVector{<:Real},
    q::Int;
    pc::Float64 = 0.0,
)

    (M, L) = size(Y)
    N = sum(counts)
    f1 = zeros(Float64, L * q)
    f2 = zeros(Float64, L * q, L * q)

    for m in 1:M
        cm = counts[m]
        for j in 1:L
            yj = Int(Y[m, j])
            idx_j = (j - 1) * q + yj
            f1[idx_j] += cm
            f2[idx_j, idx_j] += cm
            for i in (j+1):L
                yi = Int(Y[m, i])
                idx_i = (i - 1) * q + yi
                f2[idx_i, idx_j] += cm
                f2[idx_j, idx_i] += cm
            end
        end
    end

    f2 = f2 ./ N
    f1 = f1 ./ N

    if pc != 0.0
        f1 = (1 - pc) * f1 .+ pc / q
        f2 = (1 - pc) * f2 .+ pc / q / q
        for i in 1:size(Y, 2)
            f2[(i-1)*q .+ (1:q), (i-1)*q .+ (1:q)] .= 0
            for a in 1:q
                f2[(i-1)*q+a, (i-1)*q+a] = f1[(i-1)*q+a]
            end
        end
    end

    return (f1, f2)
end

function comp_freqs(Y::Array{Float64,2}, counts::Array{Float64,1}, q; pc=0.0)
    (M, L) = size(Y)
    N = sum(counts)
    f1 = zeros(Float64, L * q)
    f2 = zeros(Float64, L * q, L * q)

    for m in 1:M
        cm = counts[m]
        if cm > 0
            for j in 1:L
                yj = Int(Y[m, j])
                idx_j = (j - 1) * q + yj
                f1[idx_j] += cm
                f2[idx_j, idx_j] += cm
                for i in (j+1):L
                    yi = Int(Y[m, i])
                    idx_i = (i - 1) * q + yi
                    f2[idx_i, idx_j] += cm
                    f2[idx_j, idx_i] += cm
                end
            end
        end
    end

    f2 = f2 ./ N
    f1 = f1 ./ N

    if pc != 0.0
        f1 = (1 - pc) * f1 .+ pc / q
        f2 = (1 - pc) * f2 .+ pc / q / q
        for i in 1:size(Y, 2)
            f2[(i-1)*q .+ (1:q), (i-1)*q .+ (1:q)] .= 0
            for a in 1:q
                f2[(i-1)*q+a, (i-1)*q+a] = f1[(i-1)*q+a]
            end
        end
    end

    return (f1, f2)
end

"""
    comp_freqs!(f1, f2, Y, counts, q; pc)

Same as `comp_freqs` for `counts::Vector{Float64}`, but writes into preallocated
`f1` (length `size(Y,2) * q`) and `f2` (`(length(f1), length(f1))`).
"""
function comp_freqs!(
    f1::Vector{Float64},
    f2::Matrix{Float64},
    Y::AbstractMatrix{<:Real},
    counts::AbstractVector{<:Real},
    q::Int;
    pc::Float64 = 0.0,
)
    M, L = size(Y)
    N = sum(counts)
    fill!(f1, 0.0)
    fill!(f2, 0.0)

    for m in 1:M
        cm = counts[m]
        if cm > 0
            for j in 1:L
                yj = Int(Y[m, j])
                idx_j = (j - 1) * q + yj
                f1[idx_j] += cm
                f2[idx_j, idx_j] += cm
                for i in (j + 1):L
                    yi = Int(Y[m, i])
                    idx_i = (i - 1) * q + yi
                    f2[idx_i, idx_j] += cm
                    f2[idx_j, idx_i] += cm
                end
            end
        end
    end

    invN = inv(N)
    rmul!(f1, invN)
    rmul!(f2, invN)

    if pc != 0.0
        @. f1 = (1 - pc) * f1 + pc / q
        @. f2 = (1 - pc) * f2 + pc / q / q
        for i in 1:L
            f2[(i - 1) * q .+ (1:q), (i - 1) * q .+ (1:q)] .= 0
            for a in 1:q
                f2[(i - 1) * q + a, (i - 1) * q + a] = f1[(i - 1) * q + a]
            end
        end
    end

    return f1, f2
end

"""
compute_path_f_i(seqs, counts, q)

Compute single-site frequency trajectories over a list of smoothed count vectors.
"""
function compute_path_f_i(
    seqs::AbstractMatrix{<:Real},
    counts::Vector{<:AbstractVector{<:Real}},
    q::Int
)
    _, L = size(seqs)
    bins = length(counts)
    freq_vs_time = Matrix{Float64}(undef, L * q, bins)

    for t0 in 1:bins
        freq_vs_time[:, t0] = compute_f1(seqs, counts[t0], q=q)
    end

    return freq_vs_time
end


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

"""
fitness(field, seq, q)

Compute fitness for a single sequence.
"""
function fitness(field::Vector{Float64}, seq::AbstractVector{<:Real}, q::Int64)
    h = 1.0
    for i in 1:length(seq)
        h += field[(i - 1) * q + Int(seq[i])]
    end
    return h
end

"""
fitness(selection, mat_seq, q)

Compute fitness for each row in mat_seq.
"""
function fitness(selection::Vector{Float64}, mat_seq::Matrix{Float64}, q::Int64)
    num_seq = size(mat_seq, 1)
    fitness_seq = zeros(num_seq)
    for i in 1:num_seq
        fitness_seq[i] = fitness(selection, @view(mat_seq[i, :]), q)
    end
    return fitness_seq
end


# ---------------------------------------------------------------------------
# Species structs and population initialisation
# ---------------------------------------------------------------------------

"""Mutable struct for a haplotype with integer counts."""
mutable struct Species
    n::Int64
    s::Array{Float64,1}
    f::Float64
end

"""Mutable struct for a haplotype with fractional (smoothed) counts."""
mutable struct smoothSpecies
    n::Float64
    s::Array{Float64,1}
    f::Float64
end

function init_Species(effectiveCount::Float64, q::Int64, L::Int64, field::Array{Float64,1}; seq=[])
    if isempty(seq)
        rand_seq = rand(1.0:q, L)
        return smoothSpecies(effectiveCount, rand_seq, fitness(field, rand_seq, q))
    else
        return smoothSpecies(effectiveCount, seq, fitness(field, seq, q))
    end
end

"""
initial_population(sVec_i, effectiveCounts, q, h)

Initialise a smoothSpecies population from sequences and fractional counts.
"""
function initial_population(sVec_i::Matrix{Float64}, effectiveCounts::Array{Float64,1}, q::Int64, h::Array{Float64,1})
    pop = smoothSpecies[]
    (M, L) = size(sVec_i)
    for i in 1:M
        push!(pop, init_Species(effectiveCounts[i], q, L, h, seq=sVec_i[i,:]))
    end
    return pop, [sVec_i], [effectiveCounts]
end


# ---------------------------------------------------------------------------
# Wright-Fisher evolution (smoothSpecies / fractional counts version)
# ---------------------------------------------------------------------------

"""
evolve_pop_WF(T, N, q, L, h, i_pop, i_sVec, i_nVec; record, ReturnLast)

Evolve a smoothSpecies population under Wright-Fisher dynamics.
"""
function evolve_pop_WF(T::Int64, N::Int64, q::Int64, L::Int64, h::Vector{Float64},
                       i_pop::Array{smoothSpecies,1}, i_sVec::Vector{Matrix{Float64}},
                       i_nVec::Vector{Vector{Float64}}; record=1, ReturnLast=true)

    pop  = deepcopy(i_pop)
    npop = length(pop)
    sVec = Array{Float64,2}[]
    nVec = Vector{Float64}[]
    push!(sVec, i_sVec[1])
    push!(nVec, i_nVec[1])

    r = Vector{Float64}(undef, npop)
    wrk_n = Vector{Float64}(undef, npop)

    for t in 1:T
        s_r = 0.0
        @inbounds for i in 1:npop
            ri = abs(pop[i].n * pop[i].f)
            r[i] = ri
            s_r += ri
        end
        inv_sr = inv(s_r)
        @inbounds for i in 1:npop
            wrk_n[i] = r[i] * inv_sr
        end
        n = rand(Multinomial(N, wrk_n)) ./ N

        @inbounds for i in 1:npop
            pop[i].n = n[i]
        end

        if t % record == 0
            ad_s = zeros(npop, L)
            @inbounds for (i, s) in enumerate(pop)
                ad_s[i, :] = s.s
            end
            push!(nVec, [s.n for s in pop])
            push!(sVec, ad_s)
        end
    end

    if ReturnLast
        return pop, sVec[end], nVec[end]
    else
        return pop, sVec, nVec
    end
end
