"""
    SyntheticFlu

Synthetic inputs for the notebooks and quick checks: a population evolved with
`PopulationSimulation` (negative-binomial or Wright-Fisher dynamics with mutation and random
selection coefficients), returned in the project's data-matrix format, plus a FASTA writer in
the header layout that `FluStudy.create_study` reads. No real sequences are needed, so the
checks run without the (non-redistributable) GISAID data.
"""
module SyntheticFlu

using Random
using Dates
using Distributions: Normal

include("population_simulation.jl")
using .PopulationSimulation

export simulate_population, normalize_to_reference, write_fasta

const AA_LETTERS = "-ACDEFGHIKLMNPQRSTVWY"   # integer state k <-> AA_LETTERS[k] (1 = gap)
const Q_DATA = 21                            # alphabet size of the pipelines

"""
    simulate_population(; model = :NB, q = 4, L = 20, T = 1000, N = 1000, init_pop = 100,
                        pop_limit = 10000, k = 0.1, R = 2.0, mu = 1e-3, μh = 0.0, σh = 0.05,
                        record = 1, n_sample = nothing, reset_R = true, seed = 1)

Simulate a population and return it in the pipelines' data-matrix format. With the global
random number generator seeded by `seed`, in this order:

1. selection coefficients `s ~ Normal(μh, σh)` for the `q·L` alleles, and a random reference
   sequence `ref_seq` whose alleles get `s = 0`;
2. initial population: `init_pop` random sequences sharing `N` individuals;
3. `T` generations of `evolve_pop_NB` (`model = :NB`: offspring dispersion `k`, reproductive
   number `R`, ceiling `pop_limit`) or `evolve_pop_WF` (`model = :WF`: constant size `N`), both
   with mutation rate `mu` per site per generation, recorded every `record` generations.
   One recorded generation is one day. `reset_R = true` lets the negative-binomial population
   regrow after it has been capped (see `evolve_pop_NB`); with the original behaviour
   (`false`) some seeds stall at the population floor.

Observation: `n_sample = nothing` reports the whole population each day (count = number of
individuals); an integer `n_sample` reports that many individuals per day, drawn with
replacement in proportion to their counts with a separate generator (seed `seed + 1`), so
the dynamics do not depend on the sampling.

States `1..q` are written as `2..q+1` (`A C D E ...`), the gap-free part of the 21-state
amino-acid encoding, so the pipelines run unchanged with `q = 21`.

Returns a NamedTuple:
* `data`      -- `Matrix{Float64}` with rows `[day, count, seq_1 .. seq_L]` (days `1:T`), one
                 row per distinct observed sequence and day, sorted by day;
* `s`         -- true coefficients, length `q·L`, simulation states (`(i-1)q + a`);
* `s21`       -- the same in the pipelines' indexing (`(i-1)·21 + a + 1`), length `21·L`;
* `ref_seq`   -- reference sequence in the data encoding;
* `pop_size`, `n_species` -- population size and number of distinct sequences per day;
* `q`, `L`.
"""
function simulate_population(; model::Symbol = :NB, q::Int = 4, L::Int = 20, T::Int = 1000, N::Int = 1000,
                             init_pop::Int = 100, pop_limit::Int = 10000, k::Float64 = 0.1, R::Float64 = 2.0,
                             mu::Float64 = 1e-3, μh::Float64 = 0.0, σh::Float64 = 0.05, record::Int = 1,
                             n_sample::Union{Int,Nothing} = nothing, reset_R::Bool = true, seed::Integer = 1)
    q + 1 <= Q_DATA || error("q must be at most $(Q_DATA - 1)")
    Random.seed!(seed)
    s = rand(Normal(μh, σh), q * L)
    ref_seq = rand(1.0:q, L)
    for i in 1:L
        s[(i - 1) * q + Int(ref_seq[i])] = 0
    end

    pop0, sVec0, nVec0 = initial_population(init_pop, s, N, q, L)
    _, sVec, nVec = if model == :NB
        evolve_pop_NB(T, N, q, L, mu, s, k, R, pop_limit, pop0, sVec0, nVec0; record = record, reset_R = reset_R)
    elseif model == :WF
        evolve_pop_WF(T, N, q, L, mu, s, pop0, sVec0, nVec0; record = record, ReturnLast = false)
    else
        error("model must be :NB or :WF")
    end

    rng = MersenneTwister(seed + 1)
    rows = Vector{Vector{Float64}}()
    for day in 1:(length(sVec) - 1)             # sVec[1] is the initial population
        seqs, counts = sVec[day + 1], nVec[day + 1]
        observed = Dict{Vector{Float64},Int}()
        if n_sample === nothing
            for j in eachindex(counts)
                counts[j] > 0 && (observed[seqs[j, :] .+ 1] = counts[j])
            end
        else
            cw = cumsum(counts)
            for _ in 1:n_sample
                j = searchsortedfirst(cw, rand(rng) * cw[end])
                observed[seqs[j, :] .+ 1] = get(observed, seqs[j, :] .+ 1, 0) + 1
            end
        end
        for (seq, c) in sort(collect(observed))
            push!(rows, [Float64(day); Float64(c); seq])
        end
    end

    s21 = zeros(Q_DATA * L)
    for i in 1:L, a in 1:q
        s21[(i - 1) * Q_DATA + a + 1] = s[(i - 1) * q + a]
    end
    return (data = permutedims(reduce(hcat, rows)), s = s, s21 = s21, ref_seq = ref_seq .+ 1,
            pop_size = sum.(nVec[2:end]), n_species = length.(nVec[2:end]), q = q, L = L)
end

"""
    normalize_to_reference(s21, ref; q = 21)

Coefficients relative to the allele of sequence `ref` at each site (`s[i,a] - s[i,ref[i]]`),
the normalisation `MAPestimate` uses (its reference is the consensus at the first time point).
"""
function normalize_to_reference(s21::Vector{Float64}, ref::AbstractVector; q::Int = Q_DATA)
    out = copy(s21)
    for i in 1:length(ref), a in 1:q
        out[(i - 1) * q + a] -= s21[(i - 1) * q + Int(ref[i])]
    end
    return out
end

"""
    write_fasta(path, syn; start_date = Date(2009, 1, 1), protein = "HA", region = "north_america")

Write the observed sequences of `syn` (from `simulate_population`) as one FASTA record per
individual, with the header layout `strain|isolate_id|date|segment|hash|region|clade` used by
`FluStudy`. A sequence observed on day `d` is dated `start_date + d` days. Build the study with
`FluStudy.create_study(path; protein, protein_length = syn.L, ...)`. Returns the number of
records written.
"""
function write_fasta(path::AbstractString, syn; start_date::Date = Date(2009, 1, 1),
                     protein::AbstractString = "HA", region::AbstractString = "north_america")
    n = 0
    open(path, "w") do io
        for r in axes(syn.data, 1)
            date = start_date + Day(Int(syn.data[r, 1]))
            seq = String([AA_LETTERS[Int(x)] for x in syn.data[r, 3:end]])
            for _ in 1:Int(syn.data[r, 2])
                n += 1
                println(io, ">A/Synthetic/$(n)/$(year(date))|EPI_ISL_SYN$(n)|$(date)|$(protein)|SYN$(n)|$(region)|synthetic")
                println(io, seq)
            end
        end
    end
    return n
end

end # module SyntheticFlu
