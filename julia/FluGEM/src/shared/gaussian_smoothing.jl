using Distributions
using LinearAlgebra

"""
normalizeCounts(times::Vector{Float64}, counts::Vector{Float64})

Normalize counts within each unique time point so they sum to 1.
"""
function normalizeCounts(times::Vector{Float64}, counts::Vector{Float64})
    n = length(counts)
    norm_counts = similar(counts)

    i = 1
    while i <= n
        d = times[i]
        j = i
        s = 0.0

        while j <= n && times[j] == d
            s += counts[j]
            j += 1
        end

        if s == 0.0
            @inbounds for k in i:j-1
                norm_counts[k] = 0.0
            end
        else
            invs = 1.0 / s
            @inbounds for k in i:j-1
                norm_counts[k] = counts[k] * invs
            end
        end

        i = j
    end

    return norm_counts
end


"""
smoothCounts(counts, times, t, tsmooth)

Gaussian-smooth raw counts (with internal normalization per time point).
"""
function smoothCounts(counts::Vector{Float64}, times::Vector{Float64}, t::Float64, tsmooth::Float64; useDebug=false)
    counts = normalizeCounts(times, counts)
    effectiveCounts = zeros(length(counts))

    tcutoff = 3 * tsmooth
    norm = 1 / (tsmooth * sqrt(2 * π))

    for k in 1:length(counts)
        if abs(times[k] - t) > tcutoff
            effectiveCounts[k] = 0
        else
            effectiveCounts[k] = counts[k] * exp(-((times[k] - t) / tsmooth)^2 / 2) * norm
        end
    end

    if useDebug
        counted = sum(effectiveCounts .> 0)
        println("time $t\t$counted contributing sequences")
    end

    return effectiveCounts
end


"""
smoothCounts_normalized(counts, times, t, tsmooth)

Gaussian-smooth pre-normalized counts (no internal normalization step).
"""
function smoothCounts_normalized(
    counts::Vector{Float64},
    times::Vector{Float64},
    t::Float64,
    tsmooth::Float64;
    useDebug = false,
)
    effectiveCounts = zeros(Float64, length(counts))

    tcutoff = 3 * tsmooth
    norm = 1 / (tsmooth * sqrt(2π))

    for k in eachindex(counts, times)
        dt = times[k] - t
        if abs(dt) <= tcutoff
            effectiveCounts[k] = counts[k] * exp(-(dt / tsmooth)^2 / 2) * norm
        end
    end

    if useDebug
        counted = count(>(0), effectiveCounts)
        println("time $t\t$counted contributing sequences")
    end

    return effectiveCounts
end
