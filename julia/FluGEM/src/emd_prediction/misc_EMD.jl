using Base.Threads
using StatsBase, Dates, DataStructures, Statistics, DataFrames
using CSV
import JuMP
import MathOptInterface
include("sequence_distance.jl")   # BioTools.hamming (replaces FluPredictibility.BioTools)
using BioSequences: BioSequence, AminoAcid, AA_Gap, AA_A, AA_C, AA_D, AA_E, AA_F, AA_G, AA_H, AA_I, AA_K, AA_L, AA_M, AA_N, AA_P, AA_Q, AA_R, AA_S, AA_T, AA_V, AA_W, AA_Y, AA_Term, AA_B, AA_J, AA_X, AA_Z, @aa_str
using HiGHS
const MOI = MathOptInterface

# The amino-acid encoding dictionaries (intToaa, charToaa, aaToint) are defined in
# src/shared/aa_encoding.jl, loaded through src/shared/mpl_inference.jl.


# ---------------------------------------------------------------------------
# Sequence encoding helpers
# ---------------------------------------------------------------------------

function transform_seq(seq::Any, D::Dict{Int,AminoAcid})
    q = length(seq)
    let_seq = Vector{AminoAcid}(undef, q)
    for i in 1:q; let_seq[i] = D[seq[i]]; end
    return let_seq
end

function transform_seq(seq::Any, D::Dict{AminoAcid,Int64})
    q = length(seq)
    let_seq = Array{Int64}(undef, q)
    for i in 1:q; let_seq[i] = D[seq[i]]; end
    return let_seq
end

function transform_aa_seq(aa_seq_::Vector{AminoAcid})
    seq = deepcopy(aa"")
    for i in 1:length(aa_seq_); push!(seq, aa_seq_[i]); end
    return seq
end

"""
sequence_distribution(msa, weights)

Build a Dict{BioSequence, Float64} from a sequence matrix and weight vector.
"""
function sequence_distribution(msa::Matrix{Float64}, weigts::Vector{Float64})
    d = Dict{BioSequence,Float64}()
    (m, l) = size(msa)
    for i in 1:m
        aa_seq = transform_aa_seq(transform_seq(Int.(msa[i,:]), intToaa))
        d[aa_seq] = weigts[i]
    end
    return d
end


# ---------------------------------------------------------------------------
# Distribution subset selection
# ---------------------------------------------------------------------------

"""
select_distribution_subset(msa, thr)

Keep only sequences with frequency >= thr and renormalise.
"""
function select_distribution_subset(msa::Dict{BioSequence,Float64}, thr::Float64)
    d = Dict{BioSequence,Float64}()
    all_seqs   = collect(keys(msa))

    for s in all_seqs
        if msa[s] >= thr
            d[s] = msa[s]
        end
    end

    if isempty(d)
        if isempty(msa)
            return d                      # nothing we can do; _normalize_dist will flag it
        end
        kbest = argmax(msa)               # key with the largest value
        d[kbest] = msa[kbest]
    end

    norm = sum(values(d))
    if norm <= 0 || !isfinite(norm)
        return d                          # leave as-is; _normalize_dist flags it downstream
    end
    new_seqs = collect(keys(d))
    for s in new_seqs
        d[s] = d[s] / norm
    end

    d[new_seqs[end]] += 1.0 - sum(values(d))   # numerical fix (now safe: new_seqs non-empty)

    return d
end


# ---------------------------------------------------------------------------
# Consensus sequence
# ---------------------------------------------------------------------------

"""
consensus_sequence(data; q)

Return the most-frequent amino acid at every site as a BioSequence.
"""
function consensus_sequence(data::Dict{BioSequence,Float64}; q=21)
    M   = length(data)
    sqs = collect(keys(data))
    L   = length(sqs[1])

    msa   = zeros(M, L)
    freq_ = zeros(M)
    for i in 1:M
        msa[i,:]  .= transform_seq(sqs[i], aaToint)
        freq_[i]   = data[sqs[i]]
    end
    f_mat = reshape(compute_f1(msa, freq_), (q, L))
    ref_s = map(x -> findmax(f_mat[:, x])[2], 1:L)
    aa_s  = transform_seq(ref_s, intToaa)
    return transform_aa_seq(aa_s)
end


# ---------------------------------------------------------------------------
# Lower-bound distribution (nearest-neighbour re-assignment)
# ---------------------------------------------------------------------------

"""
lower_bound_dist(d0, dtarget)

Reassign each sequence in dtarget to its nearest neighbour in d0 (Hamming).
"""
function lower_bound_dist(d0::Dict{BioSequence,Float64}, dtarget::Dict{BioSequence,Float64})
    P = deepcopy(d0)
    Q = dtarget
    f = BioTools.hamming

    d = Array{Float64,2}(undef, length(P), length(Q))
    for (i, (x, _)) in enumerate(P)
        for (j, (y, _)) in enumerate(Q)
            d[i, j] = f(x, y)
        end
    end

    for key in keys(P); P[key] = 0.0; end

    (n1, n2) = size(d)
    for i in 1:n2
        (dis, pos) = findmin(d[:, i])
        s1 = collect(keys(P))[pos, :][1]
        P[s1] += collect(values(dtarget))[i]
    end

    return P
end


# ---------------------------------------------------------------------------
# Weird-mutation detection and correction
# ---------------------------------------------------------------------------

"""
find_weird_mutations(full_path_freq, L, q; threshold)

Return a DataFrame of (mut_index, position, allele) for mutations that never
appear in two consecutive time points above `threshold`.
"""
function find_weird_mutations(full_path_freq, L, q; threshold=0.01)
    weird = Int[]

    for mut_idx in 1:size(full_path_freq, 1)
        freq_traj = view(full_path_freq, mut_idx, :)
        has_high_pair = any(freq_traj[t] > threshold && freq_traj[t+1] > threshold
                            for t in 1:(length(freq_traj)-1))
        if !has_high_pair
            push!(weird, mut_idx)
        end
    end

    df = DataFrame(
        mut_index = weird,
        position  = [(div(m-1, q) + 1) for m in weird],
        allele    = map(x -> intToaa[x], [((m-1) % q) + 1 for m in weird])
    )

    return df
end

"""
correct_weird_mutations(seq_counts, weird_df)

Replace alleles marked as weird with the majority allele at that position,
then renormalise.
"""
function correct_weird_mutations(seq_counts::Dict{BioSequence,Float64}, weird_df::DataFrame)
    majority_map = Dict{Int,AminoAcid}()
    for pos in unique(weird_df.position)
        freq_map = Dict{AminoAcid,Float64}()
        for (seq, c) in seq_counts
            aa = seq[pos]
            freq_map[aa] = get(freq_map, aa, 0.0) + c
        end
        majority_map[pos] = argmax(freq_map)
    end

    weird_set  = Set(((row.position, row.allele) for row in eachrow(weird_df)))
    corrected  = DefaultDict{BioSequence,Float64}(0.0)
    imputations = 0

    for (seq, c) in seq_counts
        new_seq = deepcopy(seq)
        for (pos, maj) in majority_map
            if (pos, new_seq[pos]) in weird_set
                new_seq[pos] = maj
                imputations += 1
            end
        end
        corrected[new_seq] += c
    end

    total = sum(values(corrected))
    for k in keys(corrected); corrected[k] /= total; end

    return Dict(corrected), imputations
end

"""
correct_weird_mutations_data(data, weird_df, q) -> Matrix{Float64}

A priori counterpart to `correct_weird_mutations`. Instead of cleaning each
derived distribution (d1, d2, the simulated output, ...) independently after
the fact, this replaces weird alleles directly in the raw `data` matrix
(columns `[time, count, seq_1, ..., seq_L]`, Float64 state codes `1..q`),
*before* it is used to fit selection coefficients or seed the forward
simulation. Doing the correction here means `MAPestimate`, `MLRmodel`, and
`simulate_medoid_distribution` / `simulate_slm_medoid_distribution` all see
already-clean sequences, so no separate `_corrected` pass is needed
downstream.

The majority allele at each flagged position is computed once, pooling all
rows of `data` (weighted by `data[:,2]`) -- unlike `correct_weird_mutations`,
which recomputes a local majority separately for each distribution it's
applied to (d1's majority, d2's majority, etc. can differ). Pooling matches
the "clean the input once" intent of an a priori correction.
"""
function correct_weird_mutations_data(data::Matrix{Float64}, weird_df::DataFrame, q::Int)
    data_corr = copy(data)
    counts    = data[:, 2]

    for pos in unique(weird_df.position)
        col = 2 + pos

        state_counts = Dict{Float64,Float64}()
        for i in 1:size(data, 1)
            s = data[i, col]
            state_counts[s] = get(state_counts, s, 0.0) + counts[i]
        end
        majority = argmax(state_counts)

        weird_states = Set(((row.mut_index - 1) % q) + 1
                            for row in eachrow(weird_df) if row.position == pos)

        for i in 1:size(data, 1)
            if data_corr[i, col] in weird_states
                data_corr[i, col] = majority
            end
        end
    end

    return data_corr
end


# ---------------------------------------------------------------------------
# EMD solvers
# ---------------------------------------------------------------------------
"""
    _normalize_dist(D) -> (normalized_dict, total, ok)

Renormalise a distribution to sum to 1. Returns ok=false for empty /
all-zero / non-finite distributions (caller should treat EMD as Inf).
"""
function _normalize_dist(D)
    s = sum(v for (_, v) in D; init = 0.0)
    if isempty(D) || s <= 0 || !isfinite(s)
        return (D, s, false)
    end
    return (Dict(k => v / s for (k, v) in D), s, true)
end




"""
EMD_HiGHS(P, Q)

Solve the Earth Mover's Distance (Wasserstein-1) between distributions P and Q
using Hamming as the ground metric. Returns (emd, flow, status).
"""
function EMD_HiGHS(P, Q)
    f = BioTools.hamming

    Pn, sP, okP = _normalize_dist(P)
    Qn, sQ, okQ = _normalize_dist(Q)
    if !okP || !okQ
        @warn "EMD: degenerate input (sum(P)=$sP, sum(Q)=$sQ, " *
              "|P|=$(length(P)), |Q|=$(length(Q))). Returning emd=Inf."
        return (emd = Inf, flow = zeros(length(P), length(Q)),
                status = MathOptInterface.INFEASIBLE)   # EMD_HiGHS: include flow
    end
    P, Q = Pn, Qn


    d = Array{Float64,2}(undef, length(P), length(Q))
    for (i, (x, _)) in enumerate(P)
        for (j, (y, _)) in enumerate(Q)
            d[i, j] = f(x, y)
        end
    end

    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    JuMP.@variable(model, w[i=1:length(P), j=1:length(Q)] >= 0)
    for (i, (_, p)) in enumerate(P)
        JuMP.@constraint(model, sum(w[i,j] for j in 1:length(Q)) == p)
    end
    for (j, (_, q)) in enumerate(Q)
        JuMP.@constraint(model, sum(w[i,j] for i in 1:length(P)) == q)
    end
    JuMP.@objective(model, Min, sum(sum(w .* d)))
    JuMP.optimize!(model)

    status = JuMP.termination_status(model)
    if status != MathOptInterface.OPTIMAL
        @warn "EMD_HiGHS: solver returned $(status) " *
              "(sum(P)=$(sum(p for (_,p) in P)), sum(Q)=$(sum(q for (_,q) in Q)), " *
              "|P|=$(length(P)), |Q|=$(length(Q))). Returning emd=Inf."
        return (emd = Inf,
                flow = fill(NaN, length(P), length(Q)),
                status = status)
    end

    return (emd = JuMP.objective_value(model),
            flow = JuMP.value.(w),
            status = status)
end

"""
EMD_HiGHS_value(P, Q)

Lightweight EMD solver returning only the scalar distance (used internally
by safe_emd / medoid_by_emd).
"""
function EMD_HiGHS_value(P, Q)
    f  = BioTools.hamming

    Pn, sP, okP = _normalize_dist(P)
    Qn, sQ, okQ = _normalize_dist(Q)
    if !okP || !okQ
        @warn "EMD: degenerate input (sum(P)=$sP, sum(Q)=$sQ, " *
              "|P|=$(length(P)), |Q|=$(length(Q))). Returning emd=Inf."
        return (emd = Inf, status = MathOptInterface.INFEASIBLE)   # EMD_HiGHS: include flow
    end
    P, Q = Pn, Qn

    nP = length(P)
    nQ = length(Q)

    d = Matrix{Float64}(undef, nP, nQ)
    for (i, (x, _)) in enumerate(P)
        for (j, (y, _)) in enumerate(Q)
            d[i, j] = f(x, y)
        end
    end

    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    JuMP.@variable(model, w[1:nP, 1:nQ] >= 0)
    for (i, (_, p)) in enumerate(P)
        JuMP.@constraint(model, sum(w[i, j] for j in 1:nQ) == p)
    end
    for (j, (_, q)) in enumerate(Q)
        JuMP.@constraint(model, sum(w[i, j] for i in 1:nP) == q)
    end
    JuMP.@objective(model, Min, sum(w[i, j] * d[i, j] for i in 1:nP, j in 1:nQ))
    JuMP.optimize!(model)

    status = JuMP.termination_status(model)
    status != MOI.OPTIMAL && return (emd=Inf, status=status)

    return (emd=JuMP.objective_value(model), status=status)
end

"""
safe_emd(P, Q)

Wrapper around EMD_HiGHS_value that catches solver failures gracefully.
"""
function safe_emd(P, Q)
    result = EMD_HiGHS_value(P, Q)
    status = result.status
    ok = status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
    ok && isfinite(result.emd) ? (result.emd, true, status) : (Inf, false, status)
end

"""
medoid_by_emd(samples, medoid_threshold; min_success_fraction, verbose)

Find the medoid of a vector of distributions under the EMD metric.
"""
function medoid_by_emd(
    samples::Vector,
    medoid_threshold::Float64 = 1.0e-6;
    min_success_fraction::Float64 = 0.5,
    verbose::Bool = true,
)
    n        = length(samples)
    selected = [select_distribution_subset(s, medoid_threshold) for s in samples]

    D = fill(Inf, n, n)
    for i in 1:n
        D[i, i] = 0.0
    end

    # Thread each row i: pairs (i,j) with j>i are disjoint across i — no races on D.
    Threads.@threads for i in 1:n-1
        Pi = selected[i]
        for j in (i + 1):n
            Qj = selected[j]
            d, ok, status = safe_emd(Pi, Qj)
            D[i, j] = d
            D[j, i] = d
            if !ok && verbose
                @warn "EMD failed for pair ($i, $j) with status $status"
            end
        end
    end

    success_counts = zeros(Int, n)
    @inbounds for i in 1:n
        for j in 1:n
            if i != j && isfinite(D[i, j])
                success_counts[i] += 1
            end
        end
    end

    min_successes = max(1, floor(Int, min_success_fraction * (n - 1)))

    scores = fill(Inf, n)
    for i in 1:n
        finite_mask = isfinite.(D[i, :])
        n_success = count(finite_mask) - 1
        if n_success >= min_successes
            scores[i] = sum(D[i, finite_mask])
        end
    end

    all(isinf, scores) && error("No valid medoid: too many EMD failures.")

    medoid_index = argmin(scores)
    return medoid_index, samples[medoid_index], D, scores
end

"""
average_distribution(samples) -> Dict{BioSequence,Float64}

Companion to `medoid_by_emd`: instead of selecting one representative sample,
average all of them into a single distribution, weighting each of the `M`
samples equally. Sequences absent from a given sample contribute 0 to it.
"""
function average_distribution(samples::Vector)
    M   = length(samples)
    acc = DefaultDict{BioSequence,Float64}(0.0)
    for d in samples
        for (seq, w) in d
            acc[seq] += w / M
        end
    end
    return Dict(acc)
end


# ---------------------------------------------------------------------------
# Marginal computation (used by consensus / unlinked product, kept for completeness)
# ---------------------------------------------------------------------------

"""
compute_marginals(pop)

Return a vector of per-site amino-acid frequency dictionaries.
"""
function compute_marginals(pop::Dict{BioSequence,Float64})
    seq_len   = length(first(keys(pop)))
    marginals = [Dict{AminoAcid,Float64}() for _ in 1:seq_len]

    for (seq, freq) in pop
        for i in 1:seq_len
            aa = seq[i]
            marginals[i][aa] = get(marginals[i], aa, 0.0) + freq
        end
    end

    return marginals
end
