"""
    FixationPrediction

Task 2: predict the fate (fixation / loss) of mutations observed at a prediction date.

For one prediction date the pipeline (i) infers MPL selection coefficients from the
sequences observed up to that date, (ii) labels each mutation's true future fate from the
full trajectory, and (iii) runs `M` forward Wright-Fisher simulations from the smoothed
population at that date to estimate P(fixation) and P(loss). `load_results` and
`compute_summary_curves` aggregate many dates into AUROC and fixation-probability curves.

Entry points: `run_simulation` (one date) and `run_fixation_prediction.jl` (CLI). It uses
the shared MPL kernel of `src/shared/`, like the MAP estimation and the EMD prediction.
"""
module FixationPrediction

using LinearAlgebra
using Statistics
using Random
using Dates
using CSV
using DataFrames
using StatsBase: tiedrank

export Mutation,
       init_traject,
       simulate_and_update_scores,
       build_results_dataframe,
       save_simulation_results,
       load_results,
       compute_summary_curves,
       date_to_day,
       select_time_window,
       observation_day,
       run_simulation,
       parse_string,
       score_simulations,
       score_frequency

include(joinpath("..", "shared", "mpl_inference.jl"))   # shared MPL kernel

"""Frequency trajectory of allele `state` at site `pos` (row `(pos-1)q + state`)."""
function select_traj(freq_vs_time::Matrix{Float64}, pos::Int64, state::Int64, q::Int64)
    state > q && return println("error state must be in [1:q] range")
    return freq_vs_time[(pos - 1) * q + state, 1:end]
end

"""Row indices of `data` with and without `state` at position `pos`."""
function select_seq_with_mut(data::Array{Float64,2}, pos::Int64, state::Int64)
    seqs_with_mut = []
    seqs_no_mut = []
    for i in 1:size(data, 1)
        if data[i, pos] == state
            push!(seqs_with_mut, i)
        else
            push!(seqs_no_mut, i)
        end
    end
    return seqs_with_mut, seqs_no_mut
end

"""
Compact mutation container.
Only fields used by the simulation output and the summary scoring are kept.
`T0` (observation day) and `sigma` (smoothing width) are integers, as in the
original `mutation` struct, so they are written to CSV as integers.
"""
mutable struct Mutation
    i::Int64
    val::Int64
    reference::Int64
    T0::Int64
    freq::Vector{Float64}
    emp_frequency::Float64
    time_active::Float64
    fixation::Symbol
    sc::Float64
    sigma::Int64
    gamma::Float64
    av_fitness_mut::Float64
    av_fitness_wt::Float64
    porcent_simulation_fix::Float64
    porcent_simulation_lost::Float64
    av_fixation_time::Float64
    av_lost_time::Float64
end

"""Parse a vector stored as a string like \"[0.1, 0.2, 0.3]\"."""
function parse_string(str::AbstractString)
    s = strip(str)
    s = startswith(s, "[") && endswith(s, "]") ? s[2:end-1] : s
    isempty(strip(s)) && return Float64[]
    return parse.(Float64, strip.(split(s, ',')))
end

"""
Return the time elapsed since the mutation was last clearly lost (<0.05)
or fixed (>0.95) in the observed trajectory.
"""
function obs_time(mut_vs_t::Vector{Float64}, times_::Vector{Float64})
    length(mut_vs_t) == length(times_) || error("frequency trajectory and times must have the same length")
    fix_t = findall(>(0.95), mut_vs_t)
    lost_t = findall(<(0.05), mut_vs_t)
    last_fix = isempty(fix_t) ? 1 : last(fix_t)
    last_lost = isempty(lost_t) ? 1 : last(lost_t)
    last_event = max(last_fix, last_lost)
    return last(times_) - times_[last_event]
end

"""
Day number of `date` counted from `origin` (day 0), as in run_prediction_main.jl.
Both the precomputed .txt data and FluStudy.create_study number days this way:
FluStudy drops sequences dated exactly on its start date, so day 1 = start date + 1 day.
"""
date_to_day(date::Date, origin::Date) = Dates.value(date - origin)

"""
Keep the rows of `data` dated between `start_date` and `stop_date` (inclusive) and
renumber their days so that `start_date` is day 0. `origin` is the date of day 0 in `data`.
"""
function select_time_window(data::Matrix{Float64}, origin::Date, start_date::Date, stop_date::Date)
    issorted(data[:, 1]) || error("data rows must be sorted by day")
    first_day, last_day = date_to_day(start_date, origin), date_to_day(stop_date, origin)
    window = data[first_day .<= data[:, 1] .<= last_day, :]
    window[:, 1] .-= first_day
    return window
end

"""Observed day of `data` closest to `date` (day 0 = `start_date`); ties go to the earlier day."""
function observation_day(data::Matrix{Float64}, date::Date, start_date::Date)
    u_times = sort(unique(data[:, 1]))
    return u_times[argmin(abs.(u_times .- date_to_day(date, start_date)))]
end

"""
Return the row interval in the raw `times` vector corresponding approximately
to the previous 30 days ending at the observation day `t_c`.
"""
function find_previous_time_bin(times::Vector{Float64}, t_c::Real)
    position_in_time_T0 = last(findall(==(t_c), times))
    obs_times = times[1:position_in_time_T0]
    diff = t_c .- obs_times
    valid = diff[(diff .- 30) .>= 0.0]
    if isempty(valid)
        return 1, position_in_time_T0
    else
        t = first(sort(valid))
        position_in_time_T1 = first(findall(==(t), diff))
        return position_in_time_T1, position_in_time_T0
    end
end

"""Gaussian-smoothed counts of the given rows on the grid `1:tstep:t_end`."""
function smoothed_counts(counts::Vector{Float64}, times::Vector{Float64}, t_end, tstep, tsmooth::Float64)
    grid = collect(1.0:tstep:t_end)
    return grid, [smoothCounts(counts, times, t_, tsmooth) for t_ in grid]
end

"""Normalize counts and merge identical sequence rows."""
function unique_sequences(seqs::Matrix{Float64}, counts::Vector{Float64})
    seq_list = [vec(seqs[i, :]) for i in 1:size(seqs, 1)]
    norm_counts = counts ./ sum(counts)
    new_seqs = Vector{Vector{Float64}}()
    new_counts = Float64[]

    for j in eachindex(seq_list)
        is_new = true
        for k in eachindex(new_seqs)
            if seq_list[j] == new_seqs[k]
                new_counts[k] += norm_counts[j]
                is_new = false
                break
            end
        end
        if is_new
            push!(new_seqs, seq_list[j])
            push!(new_counts, norm_counts[j])
        end
    end

    return stack(new_seqs; dims = 1), new_counts
end

"""Indices in `trajs` of non-reference mutations active at T0 (0.05 < f < 0.95)."""
active_mutations(trajs::Vector{Mutation}) = findall(x -> (0.05 < x.freq[end] < 0.95) && x.val != x.reference, trajs)

"""Diagonal noise amplitude decaying with time active."""
function DiagNoiseMat(trajs::Vector{Mutation}, σ0::Float64, τ::Float64)
    times = map(x -> x.time_active, trajs[active_mutations(trajs)]) ./ τ
    return σ0 .* diagm(exp.(-times))
end

"""
Initialize mutation trajectories from the matrix-form input, observed up to day `t_c`
(an observed day, see `observation_day`).
Input layout:
- column 1: time
- column 2: sequence count
- columns 3:end: encoded sequence states
"""
function init_traject(data::Matrix{Float64}, t_c::Real, gamma::Float64; tstep::Float64 = 10.0, tsmooth::Float64 = 15.0, q::Int64 = 21, correlation::Bool = true)
    times = data[:, 1]
    counts = data[:, 2]
    seqs = data[:, 3:end]
    u_times = unique(times)
    L = size(seqs, 2)

    # Full trajectory (all data): only used for the fate of each mutation after t_c
    _, effectiveCounts_full = smoothed_counts(counts, times, last(u_times), tstep, tsmooth)
    full_path_freq = compute_path_f_i(seqs, effectiveCounts_full, q)
    Tf = size(full_path_freq, 2)

    # Observed trajectory: rows up to t_c, grid 1:tstep:t_c
    position_in_time_T1, position_in_time_T0 = find_previous_time_bin(times, t_c)
    times_measure, effectiveCounts_list = smoothed_counts(counts[1:position_in_time_T0], times[1:position_in_time_T0], t_c, tstep, tsmooth)
    seqs_observed = seqs[1:position_in_time_T0, :]
    observed_path_freq = compute_path_f_i(seqs_observed, effectiveCounts_list, q)

    # Empirical frequency over the ~30 days ending at t_c
    femp = compute_f1(seqs[position_in_time_T1:position_in_time_T0, :], counts[position_in_time_T1:position_in_time_T0])

    # Reference = consensus at the first observed time point
    ref_s = [argmax(observed_path_freq[(i - 1) * q + 1:i * q, 1]) for i in 1:L]

    mu = 0.0
    h = if correlation
        processStandard(seqs_observed, effectiveCounts_list, times_measure, mu, L, q, gamma)
    else
        processStandardNoCorr(seqs_observed, effectiveCounts_list, times_measure, mu, L, q, gamma)
    end
    h = normalize_sc(h, ref_s, q, L)

    # Fitness-weighted share of each sequence at t_c (shared by all mutations)
    ncounts = last(effectiveCounts_list)
    sequences = seqs_observed[ncounts .> 0.0, :]
    r = ncounts[ncounts .> 0.0] .* fitness(h, sequences, q)
    p = r / sum(r)

    out = Mutation[]
    lost_thr = 0.05
    fix_threshold = 0.95
    Ti = length(times_measure)

    for pos in 1:L, state in 1:q
        k = (pos - 1) * q + state
        mut_vs_time = select_traj(observed_path_freq, pos, state, q)

        # Fate after t_c: first pair of consecutive grid points both lost or both fixed
        fixation = :poly
        for time in Ti:(Tf - 1)
            if full_path_freq[k, time] <= lost_thr && full_path_freq[k, time + 1] <= lost_thr
                fixation = :lost
                break
            elseif full_path_freq[k, time] >= fix_threshold && full_path_freq[k, time + 1] >= fix_threshold
                fixation = :fixed
                break
            end
        end

        seqs_mut, seqs_wt = select_seq_with_mut(sequences, pos, state)
        push!(out, Mutation(pos, state, ref_s[pos], t_c, mut_vs_time, femp[k], obs_time(mut_vs_time, times_measure),
                            fixation, h[k], tsmooth, gamma, sum(p[seqs_mut]), sum(p[seqs_wt]), 0.0, 0.0, 0.0, 0.0))
    end

    return out
end

"""
Forward Wright-Fisher simulations from the smoothed population at T0.
Each of the `M` runs evolves one generation at a time until the population
collapses into a single sequence (or 100*N generations) and records the first
generation at which each allele reaches the fixation / loss threshold.
`noise = 1` perturbs the selection coefficients of active mutations with the
MPL posterior covariance; `noise = 2` adds a diagonal term decaying with time active.
Returns updated copies of `trajs`.
"""
function simulate_and_update_scores(trajs::Vector{Mutation}, data::Matrix{Float64}; gamma::Float64 = 20.0, q::Int64 = 21, noise::Int64 = 0, σ0::Float64 = 0.01, τ::Float64 = 100.0, tstep::Int64 = 10, M::Int64 = 1000)
    println("Simulation Parameters:")
    println("gamma = $gamma")
    println("q = $q")
    println("noise = $noise")

    times = data[:, 1]
    counts = data[:, 2]
    seqs = data[:, 3:end]
    L = size(seqs, 2)

    fix_threshold = 0.95
    lost_threshold = 0.05
    N = 1000
    t_c = trajs[1].T0

    position_in_time_T0 = last(findall(==(t_c), times))
    times_measure, effectiveCounts_list = smoothed_counts(counts[1:position_in_time_T0], times[1:position_in_time_T0], t_c, tstep, Float64(trajs[1].sigma))
    seqs_observed = seqs[1:position_in_time_T0, :]

    h = map(x -> x.sc, trajs)
    h_0 = copy(h)
    fixMat = zeros(q * L, M)   # generation of first fixation per allele and run (0 = never)
    lostMat = zeros(q * L, M)  # generation of first loss per allele and run (0 = never)

    if noise != 0
        mut_pos = map(x -> (x.i - 1) * q + x.val, trajs[active_mutations(trajs)])
        Cint_ = Cintegrate(seqs_observed, effectiveCounts_list, times_measure, L, q, gamma)
        isC = inv(Cint_[mut_pos, mut_pos])
        isC = (1 / t_c) * ((isC + isC') / 2)
        if noise == 2
            isC .+= DiagNoiseMat(trajs, σ0, τ)
        end
        sq_isC = sqrt(isC)
    end

    # Population at T0 (identical for every run; only h_0 changes)
    ncounts = last(effectiveCounts_list)
    u_sequences, u_counts = unique_sequences(seqs_observed[ncounts .> 0.0, :], ncounts[ncounts .> 0.0])

    println("\nSimulating...")
    for m in 1:M
        println("\nTrajectory $(m)...")
        if noise != 0
            h_0[mut_pos] = h[mut_pos] .+ sq_isC * randn(length(mut_pos))
        end
        i_pop, i_sVec, i_nVec = initial_population(u_sequences, u_counts, q, h_0)

        t = 1
        while count(>(0), i_nVec[1]) != 1 && t < 100 * N
            i_pop, sVec_t, nVec_t = evolve_pop_WF(1, N, q, L, h_0, i_pop, i_sVec, i_nVec; record = 1)
            freq_ = compute_f1(sVec_t, nVec_t, q = q)
            i_sVec, i_nVec = [sVec_t], [nVec_t]

            fixMat[(freq_ .>= fix_threshold) .& (fixMat[:, m] .== 0.0), m] .= t
            lostMat[(freq_ .<= lost_threshold) .& (lostMat[:, m] .== 0.0), m] .= t
            t += 1
        end
    end

    return map(trajs) do ft
        k = (ft.i - 1) * q + ft.val
        fix_t = fixMat[k, fixMat[k, :] .> 0]
        lost_t = lostMat[k, lostMat[k, :] .> 0]
        ft_ = deepcopy(ft)
        ft_.porcent_simulation_fix = length(fix_t) / M
        ft_.porcent_simulation_lost = length(lost_t) / M
        ft_.av_fixation_time = mean(fix_t)   # NaN if never fixed
        ft_.av_lost_time = mean(lost_t)      # NaN if never lost
        ft_
    end
end

"""Build the tabular output exported after the simulation step."""
function build_results_dataframe(tr_sim::Vector{Mutation})
    return DataFrame(
        Position = getfield.(tr_sim, :i),
        Nucleotide = getfield.(tr_sim, :val),
        Reference = getfield.(tr_sim, :reference),
        Time = getfield.(tr_sim, :T0),
        Frequency = getfield.(tr_sim, :freq),
        Empirical_frequency = getfield.(tr_sim, :emp_frequency),
        Time_active = getfield.(tr_sim, :time_active),
        Selection_coefficient = getfield.(tr_sim, :sc),
        Av_fitness_mutant = getfield.(tr_sim, :av_fitness_mut),
        Av_fitness_wildtype = getfield.(tr_sim, :av_fitness_wt),
        Simulations_fixed = getfield.(tr_sim, :porcent_simulation_fix),
        Simulations_lost = getfield.(tr_sim, :porcent_simulation_lost),
        Mean_fixation_time = getfield.(tr_sim, :av_fixation_time),
        Mean_lost_time = getfield.(tr_sim, :av_lost_time),
        Fixation = string.(getfield.(tr_sim, :fixation)),
        Gamma = getfield.(tr_sim, :gamma),
        Sigma = getfield.(tr_sim, :sigma),
    )
end

"""Write simulation results to CSV."""
function save_simulation_results(tr_sim::Vector{Mutation}, out_file::String)
    mkpath(dirname(out_file))
    df = build_results_dataframe(tr_sim)
    CSV.write(out_file, df)
    return out_file
end

"""Simulation ranking: P(fix) - P(loss)."""
function score_simulations(df::DataFrame)
    out = zeros(size(df, 1), 3)
    for k in 1:size(df, 1)
        out[k, 1] = k
        out[k, 2] = df.Simulations_fixed[k] - df.Simulations_lost[k]
        out[k, 3] = df.Fixation[k] == "fixed" ? 1.0 : 0.0
    end
    return reverse(sortslices(out, dims = 1, by = x -> x[2]), dims = 1)
end

"""Null ranking: current observed frequency."""
function score_frequency(df::DataFrame)
    out = zeros(size(df, 1), 3)
    freq = last.(parse_string.(string.(df.Frequency)))
    for k in 1:size(df, 1)
        out[k, 1] = k
        out[k, 2] = freq[k]
        out[k, 3] = df.Fixation[k] == "fixed" ? 1.0 : 0.0
    end
    return reverse(sortslices(out, dims = 1, by = x -> x[2]), dims = 1)
end

"""
Area under the ROC curve of `scores` (higher = more likely positive) for `labels`
(1 = positive): the probability that a positive outscores a negative, ties counting 1/2.
NaN scores are skipped; NaN when there are no positives or no negatives. Equals the
trapezoidal AUC over the unique score thresholds (ROC.jl's `AUC`, used before).
"""
function auroc(scores::AbstractVector{<:Real}, labels::AbstractVector{<:Real})
    keep = .!isnan.(scores)
    s, y = scores[keep], labels[keep] .== 1
    P, N = count(y), count(.!y)
    (P == 0 || N == 0) && return NaN
    return (sum(tiedrank(s)[y]) - P * (P + 1) / 2) / (P * N)
end

"""
Load and concatenate all per-date CSV result files from a directory. File names
(`fixation_<subtype>_<protein>_<YYYY-MM-DD>_...`) carry the prediction date after a
common prefix, so sorting them gives chronological order, as in the original summary
script (this matters for ties in the top/bottom split).
"""
function load_results(results_dir::String)
    files = filter(f -> endswith(f, ".csv"), readdir(results_dir; join = true))
    isempty(files) && error("No CSV files found in $results_dir")
    dfs = DataFrame[]
    for f in sort(files)
        try
            push!(dfs, DataFrame(CSV.File(f)))
        catch err
            @warn "Skipping unreadable file" file = f exception = err
        end
    end
    isempty(dfs) && error("No readable CSV files found in $results_dir")
    return vcat(dfs...)
end

"""
Compute AUROC and fixation-probability summaries across frequency bins.
Returns:
(freqs, auc_sim, auc_null, fix_prob_top, fix_prob_bottom, fix_prob_all)
"""
function compute_summary_curves(df_all::DataFrame; delta = 0.05, top_fraction = 0.5)
    df = filter(row -> row.Fixation in ["lost", "fixed"], df_all)
    df = filter(row -> row.Nucleotide != 1, df)

    auc_sim = Float64[]
    auc_null = Float64[]
    fix_prob_top = Float64[]
    fix_prob_bottom = Float64[]
    fix_prob_all = Float64[]
    freq_last = last.(parse_string.(string.(df.Frequency)))

    for f0 in 0.1:0.1:0.9
        mask = (f0 + delta .> freq_last) .& (freq_last .>= f0 - delta)
        df_bin = df[mask, :]

        if nrow(df_bin) == 0
            push!(auc_sim, NaN)
            push!(auc_null, NaN)
            push!(fix_prob_top, NaN)
            push!(fix_prob_bottom, NaN)
            push!(fix_prob_all, NaN)
            continue
        end

        sc_sim = score_simulations(df_bin)
        sc_null = score_frequency(df_bin)
        push!(auc_sim, auroc(sc_sim[:, 2], sc_sim[:, 3]))
        push!(auc_null, auroc(sc_null[:, 2], sc_null[:, 3]))

        # mean of an empty slice is NaN, matching the original 0/0
        n_top = Int(round(top_fraction * nrow(df_bin)))
        push!(fix_prob_top, mean(sc_sim[1:n_top, 3]))
        push!(fix_prob_bottom, mean(sc_sim[n_top + 1:end, 3]))
        push!(fix_prob_all, mean(sc_sim[:, 3]))
    end

    return collect(0.1:0.1:0.9), auc_sim, auc_null, fix_prob_top, fix_prob_bottom, fix_prob_all
end

"""
Run the full simulation step for one observation date and optionally save the CSV.
`data` rows are `[day, count, seq...]` with day 0 = `origin`. Only rows dated between
`start_date` and `stop_date` are used, renumbered so that day 0 = `start_date`; the
prediction time is the observed day closest to `observation_date`.
"""
function run_simulation(data::Matrix{Float64}, observation_date::Date;
    start_date::Date,
    stop_date::Date,
    origin::Date = start_date,
    gamma::Float64 = 20.0,
    q::Int64 = 21,
    tsmooth::Float64 = 60.0,
    noise::Int64 = 1,
    M::Int64 = 1000,
    out_file::Union{Nothing,String} = nothing,
    correlation::Bool = true)

    start_date <= observation_date <= stop_date ||
        error("observation date $observation_date is outside [$start_date, $stop_date]")
    data = select_time_window(data, origin, start_date, stop_date)
    t_c = observation_day(data, observation_date, start_date)
    println("Observation date $observation_date -> day $(Int(t_c)) ($(start_date + Day(Int(t_c))))")

    tr = init_traject(data, t_c, gamma; tsmooth = tsmooth, q = q, correlation = correlation)
    tr_sim = simulate_and_update_scores(tr, data; q = q, noise = noise, M = M, gamma = gamma)

    if !isnothing(out_file)
        save_simulation_results(tr_sim, out_file)
    end
    return tr_sim
end

end
