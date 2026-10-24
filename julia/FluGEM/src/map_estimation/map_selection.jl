# Task 1 -- table of MAP selection coefficients.
#
# Requires the shared kernel in scope (src/shared/mpl_inference.jl) for `intToaa`.
using DataFrames

"""HA domain along the alignment: signal peptide 1-17, HA1 18-344, HA2 345-."""
ha_domain(i) = i <= 17 ? "SigPep" : i <= 344 ? "HA1" : "HA2"

"""
    map_selection_table(map_result; q = 21, freq_threshold = 0.01, protein = "HA") -> DataFrame

Keep the non-gap alleles of `map_result` (the output of `MAPestimate`) whose frequency
exceeds `freq_threshold` at two consecutive time points at least once, and tabulate
`Position`, `Domain` (HA only), `Reference`, `Mutation` and `Selection_coefficient`.
Reference alleles are kept (their coefficient is 0 by normalisation).
"""
function map_selection_table(map_result; q::Int = 21, freq_threshold::Real = 0.01, protein::AbstractString = "HA")
    # Rows of `observed_path_freq` (and entries of `h`) are indexed by
    # (site i, allele a) -> (i - 1) * q + a; columns are time points.
    L = length(map_result.ref_s)
    idx(i, a) = (i - 1) * q + a

    above = map_result.observed_path_freq .> freq_threshold
    keep  = vec(any(above[:, 1:end-1] .& above[:, 2:end]; dims = 2))

    sites = [(i, a) for i in 1:L for a in 2:q if keep[idx(i, a)]]   # a = 1 is the gap

    df = DataFrame(
        Position              = first.(sites),
        Reference             = [string(intToaa[map_result.ref_s[i]]) for (i, _) in sites],
        Mutation              = [string(intToaa[a]) for (_, a) in sites],
        Selection_coefficient = [map_result.h[idx(i, a)] for (i, a) in sites],
    )
    protein == "HA" && insertcols!(df, 2, :Domain => ha_domain.(df.Position))
    return df
end
