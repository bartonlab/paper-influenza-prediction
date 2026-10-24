# Task 1 -- table of MAP selection coefficients.
#
# Requires the shared kernel in scope (src/shared/mpl_inference.jl) for `intToaa`.
using DataFrames

"""
    ha_domain(i, subtype) -> "SigPep" | "HA1" | "HA2"

Domain of position `i` of the 566-column HA alignment. The H3 signal peptide is one
residue shorter than the H1 one, so both boundaries shift between subtypes:

| Subtype | Signal peptide | HA1 | HA2 |
|---|---|---|---|
| `H3N2` | 1-16 | 17-345 | 346-566 |
| `H1N1` | 1-17 | 18-344 | 345-566 |
"""
function ha_domain(i::Integer, subtype::AbstractString)
    sigpep_end, ha1_end =
        subtype in ("H3N2", "H3N2_2023") ? (16, 345) :
        subtype == "H1N1"                ? (17, 344) :
        error("Unknown subtype \"$subtype\" (expected H3N2, H1N1 or H3N2_2023)")
    return i <= sigpep_end ? "SigPep" : i <= ha1_end ? "HA1" : "HA2"
end

"""
    map_selection_table(map_result; q = 21, freq_threshold = 0.01, protein = "HA",
                        subtype = nothing) -> DataFrame

Keep the non-gap alleles of `map_result` (the output of `MAPestimate`) whose frequency
exceeds `freq_threshold` at two consecutive time points at least once, and tabulate
`Position`, `Domain` (HA only), `Reference`, `Mutation` and `Selection_coefficient`.
Reference alleles are kept (their coefficient is 0 by normalisation). `protein = "HA"`
needs `subtype`, because the domain boundaries differ between subtypes (`ha_domain`).
"""
function map_selection_table(map_result; q::Int = 21, freq_threshold::Real = 0.01,
                             protein::AbstractString = "HA",
                             subtype::Union{Nothing,AbstractString} = nothing)
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
    if protein == "HA"
        subtype === nothing && error("protein = \"HA\" needs `subtype` for the Domain column")
        insertcols!(df, 2, :Domain => ha_domain.(df.Position, subtype))
    end
    return df
end
