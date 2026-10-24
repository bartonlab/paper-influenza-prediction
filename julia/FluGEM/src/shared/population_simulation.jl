"""
    PopulationSimulation

Individual-based population dynamics with selection and mutation, used to generate synthetic
data (`synthetic_data.jl`). From `Population_dynamics.jl` of the original analysis code, with
the same random-number calls in the same order, so a given seed gives the same populations.

* `evolve_pop_NB` -- negative-binomial offspring numbers (overdispersion `k`, reproductive
  number `R`) with a population ceiling `max_pop_size`;
* `evolve_pop_WF` -- Wright-Fisher resampling at constant size `N`.

Both mutate each individual at rate `mu` per site per generation. Sequences are vectors of
integer states `1..q`; fitness is additive, `1 + Σ_i h[(i-1)q + s_i]`.

Kept in its own module because its `Species`, `fitness` and `init_Species` differ from the
methods of the same name in `src/shared/Population_dynamics.jl`.
"""
module PopulationSimulation

using Distributions: Binomial, Multinomial, NegativeBinomial

export Species, initial_population, evolve_pop_NB, evolve_pop_WF

"""A set of identical individuals: `n` copies of sequence `s`, with fitness `f`."""
mutable struct Species
    n::Int64
    s::Array{Float64,1}
    f::Float64
end

"""Additive fitness of `seq`: `1 + Σ_i field[(i-1)q + seq[i]]`."""
function fitness(field::Vector{Float64}, seq::Array{Float64,1}, q::Int64)
    h = 1
    for i in 1:length(seq)
        h += field[(i - 1) * q + Int(seq[i])]
    end
    return h
end

"""`Species` with `n` members and sequence `seq` (random when `seq` is empty)."""
function init_Species(n::Int64, q::Int64, L::Int64, field::Array{Float64,1}; seq = [])
    if isempty(seq)
        rand_seq = rand(1.0:q, L)
        return Species(n, rand_seq, fitness(field, rand_seq, q))
    end
    return Species(n, seq, fitness(field, seq, q))
end

"""Copy of `s` with a single member."""
function clone(s::Species)
    out = deepcopy(s)
    out.n = 1
    return out
end

"""
    mutate(s, mu, h, q, L) -> Vector

Each member of `s` mutates with probability `mu * L`, at a random site, to a different random
state; every mutant becomes a new single-member species. Returns the mutants followed by `s`
itself (if any members remain unmutated).
"""
function mutate(s::Species, mu::Float64, h::Array{Float64,1}, q::Int64, L::Int64)
    newSpecies = []
    if s.n > 0
        nMut = rand(Binomial(s.n, mu * L))   # number of members that mutate
        s.n -= nMut
        for i in rand(1:L, nMut)             # one mutation site per mutant
            new_s = clone(s)
            new_mut = rand(1.0:q)
            while new_mut == new_s.s[i]
                new_mut = rand(1.0:q)
            end
            new_s.s[i] = new_mut
            new_s.f = fitness(h, new_s.s, q)
            push!(newSpecies, new_s)
        end
    end
    s.n > 0 && push!(newSpecies, s)
    return newSpecies
end

"""
    merge_species!(newPop, species)

Add each species in `species` to `newPop`, merging it into an existing entry with the same
sequence.
"""
function merge_species!(newPop::Vector{Species}, species)
    for sp in species
        unique = true
        for k in 1:length(newPop)
            if sp.s == newPop[k].s
                unique = false
                newPop[k].n += sp.n
                break
            end
        end
        unique && push!(newPop, sp)
    end
    return newPop
end

"""Sequence matrix (species × sites) and count vector of `pop`."""
function snapshot(pop::Vector{Species}, L::Int64)
    ad_s = zeros(length(pop), L)
    for (i, s) in enumerate(pop)
        ad_s[i, :] = s.s
    end
    return ad_s, [s.n for s in pop]
end

"""
    initial_population(randomize, h, N, q, L) -> (pop, sVec, nVec)

`randomize` random sequences sharing `N` individuals (`floor(N/randomize)` each, the last one
takes the remainder); a single random sequence when `randomize <= 1`.
"""
function initial_population(randomize::Int64, h::Array{Float64,1}, N::Int64, q::Int64, L::Int64)
    pop, sVec, nVec = Species[], Array{Float64,2}[], Vector{Int64}[]
    if randomize > 1
        randomize = min(randomize, N)
        n_seqs    = Int64(floor(N / randomize))
        temp_sVec = zeros(randomize, L)
        temp_nVec = Int64[]
        for i in 1:randomize
            temp_seq = rand(1.0:q, L)
            i == randomize && (n_seqs = N - sum(temp_nVec))
            push!(pop, init_Species(n_seqs, q, L, h, seq = temp_seq))
            temp_sVec[i, :] .= temp_seq
            push!(temp_nVec, n_seqs)
        end
        push!(sVec, temp_sVec)
        push!(nVec, temp_nVec)
    else
        temp_seq = rand(1.0:q, L)
        push!(pop, init_Species(N, q, L, h, seq = temp_seq[1:end]))
        push!(sVec, reshape(temp_seq, 1, L))   # (the original pushed the vector itself, which errors)
        push!(nVec, N * ones(Int64, 1))
    end
    return pop, sVec, nVec
end

"""
    evolve_pop_NB(T, N, q, L, mu, h, k, R, max_pop_size, i_pop, i_sVec, i_nVec;
                  record = 1, min_pop = 1000, reset_R = false) -> (pop, sVec, nVec)

Evolve `i_pop` for `T` generations. Each species with `n` members and fitness `f` leaves a
negative-binomial number of offspring with mean `n R f` and dispersion `n k`. Whenever the
population reaches `max_pop_size`, `R` is rescaled to `(1 / mean fitness) * max_pop_size / size`,
so the expected size stays at the ceiling. A generation whose total is below `min_pop` is
redrawn. Then every species mutates (`mutate`). `sVec` / `nVec` start with the initial state
and record the population every `record` generations. `N` is not used.

`reset_R = false` (default, the original behaviour): a rescaled `R` is kept until the ceiling
is reached again. If it was pushed well below 1 and the population then falls to `min_pop`,
the expected next size is below `min_pop` and the redraw loop can run for an extremely long
time. `reset_R = true`: `R` returns to its initial value whenever the population is below the
ceiling, so the population can regrow.
"""
function evolve_pop_NB(T::Int64, N::Int64, q::Int64, L::Int64, mu::Float64, h::Vector{Float64}, k::Float64, R::Float64,
                       max_pop_size::Int64, i_pop::Array{Species,1}, i_sVec::Vector{Matrix{Float64}},
                       i_nVec::Vector{Vector{Int64}}; record = 1, min_pop::Int = 1000, reset_R::Bool = false)
    pop = deepcopy(i_pop)
    sVec = Array{Float64,2}[i_sVec[1]]
    nVec = Vector{Int64}[i_nVec[1]]
    total_population = sum([s.n for s in pop])
    R0 = R

    for t in 1:T
        average_fitness = sum([s.f * s.n for s in pop]) / sum([s.n for s in pop])
        if total_population >= max_pop_size
            R = (1 / average_fitness) * (max_pop_size / total_population)
        elseif reset_R
            R = R0
        end

        r = [s.n * k for s in pop]
        r[r .== 0.0] .= 0.00001
        p = k ./ (k .+ [R * s.f for s in pop])
        n = rand.(map(NegativeBinomial, r, p))
        total_population = sum(n)
        while total_population < min_pop   # redraw if the population would almost die out
            n = rand.(map(NegativeBinomial, r, p))
            total_population = sum(n)
        end

        newPop = Species[]
        for i in 1:length(pop)
            pop[i].n = n[i]
            merge_species!(newPop, mutate(pop[i], mu, h, q, L))
        end
        pop = newPop

        if t % record == 0
            ad_s, counts = snapshot(pop, L)
            push!(nVec, counts)
            push!(sVec, ad_s)
        end
    end
    return pop, sVec, nVec
end

"""
    evolve_pop_WF(T, N, q, L, mu, h, i_pop, i_sVec, i_nVec; record = 1, ReturnLast = true)

Wright-Fisher counterpart of `evolve_pop_NB`: each generation `N` individuals are drawn from a
multinomial with probabilities proportional to `n f`, then every species mutates. Returns
`(pop, sVec[end], nVec[end])` when `ReturnLast`, otherwise `(pop, sVec, nVec)`.
"""
function evolve_pop_WF(T::Int64, N::Int64, q::Int64, L::Int64, mu::Float64, h::Vector{Float64},
                       i_pop::Array{Species,1}, i_sVec::Vector{Matrix{Float64}},
                       i_nVec::Vector{Vector{Int64}}; record = 1, ReturnLast = true)
    pop = deepcopy(i_pop)
    sVec = Array{Float64,2}[i_sVec[1]]
    nVec = Vector{Int64}[i_nVec[1]]
    mutations_present = mu != 0.0

    for t in 1:T
        r = abs.([s.n * s.f for s in pop])
        n = rand(Multinomial(N, r / sum(r)))
        newPop = Species[]
        for i in 1:length(pop)
            pop[i].n = n[i]
            mutations_present && merge_species!(newPop, mutate(pop[i], mu, h, q, L))
        end
        mutations_present && (pop = deepcopy(newPop))

        if t % record == 0
            ad_s, counts = snapshot(pop, L)
            push!(nVec, counts)
            push!(sVec, ad_s)
        end
    end
    return ReturnLast ? (pop, sVec[end], nVec[end]) : (pop, sVec, nVec)
end

end # module PopulationSimulation
