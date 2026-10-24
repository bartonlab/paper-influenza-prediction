# Helpers shared by the entry-point scripts: command-line options, output file names and the
# study window of each subtype.
using Dates

"""
    parse_cli(args) -> (options::Dict{String,String}, positional::Vector{String})

`--name value` pairs become `options["name"] = value`; all other arguments are positional
(kept for the SLURM-array calling convention).
"""
function parse_cli(args)
    options, positional = Dict{String,String}(), String[]
    i = 1
    while i <= length(args)
        if startswith(args[i], "--")
            i < length(args) || error("option $(args[i]) needs a value")
            options[args[i][3:end]] = args[i + 1]
            i += 2
        else
            push!(positional, args[i])
            i += 1
        end
    end
    return options, positional
end

"""Value as written in file names: `60.0 -> "60"`, `0.01 -> "0.01"`, `true -> "true"`."""
name_value(x::AbstractFloat) = isinteger(x) ? string(Int(x)) : string(x)
name_value(x) = string(x)

"""
    run_name(kind, subtype, protein, dates...; params...) -> String

File-name stem `<kind>_<subtype>_<protein>_<date>[_<date>]_<key><value>...`, used by all
three tasks. For example

    run_name("fixation", "H3N2", "HA", Date(2010, 1, 3); noise = 1, tsmooth = 60.0, gamma = 20.0)

gives `fixation_H3N2_HA_2010-01-03_noise1_tsmooth60_gamma20`. An empty `kind` is left out.
"""
function run_name(kind::AbstractString, subtype::AbstractString, protein::AbstractString, dates...; params...)
    parts = String[string(d) for d in dates]
    pushfirst!(parts, subtype, protein)
    isempty(kind) || pushfirst!(parts, kind)
    append!(parts, ["$(k)$(name_value(v))" for (k, v) in params])
    return join(parts, "_")
end

"""
    study_window(subtype) -> (start_date, stop_date)

Day 0 of the data matrix and the last date of the study for each subtype. The start dates
must match the ones hard-coded in the pipeline functions (`MAPestimate`, the EMD pipeline).
"""
function study_window(subtype::AbstractString)
    subtype == "H3N2"      && return (Date(2009, 1, 1), Date(2020, 3, 1))
    subtype == "H1N1"      && return (Date(2009, 5, 1), Date(2020, 3, 1))
    subtype == "H3N2_2023" && return (Date(2023, 1, 1), Date(2026, 3, 1))
    error("Unknown subtype \"$subtype\" (expected H3N2, H1N1 or H3N2_2023)")
end
