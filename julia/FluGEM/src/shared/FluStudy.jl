"""
    FluStudy

Self-contained preprocessing pipeline for influenza HA / NA alignments.

`create_study` is the single entry point. Stages, in order:

  (a) read FASTA; drop unreadable records and strains without a date.
  (b) drop strains with more than `gap_threshold` (default 5%) gaps.
  (b') optionally deduplicate by `strain` header field
       (matches the legacy `FluPop` dict-based dedup).
  (b'') optionally filter by `region` and/or `country` header fields
        (mirrors `FluPredictibility.filter_by_region!` /
        `filter_by_country!`).
  (c) bin sequences by date.
  (d) within each bin: drop sequences whose length differs from the
      expected protein length, encode AAs to integers (1 = ambiguous /
      gap sentinel, 2..21 = canonical AAs), and DEDUPLICATE rows that
      share the same sequence -- `count` records how many strains on
      that day carried it, and per-row lists of strain names and
      regions are kept.
  (e) replace ambiguous residues with the locally most frequent state
      (Gaussian-smoothed over `tsmooth` days).
  (f) clip to the requested `[start_date, stop_date]` study window.
  (g) drop strains > `mut_threshold` mutations away from a yearly
      reference (the most common strain at the start of the window).
  (h) compute the smoothed position-frequency trajectory.
  (i) attach a clade label and a majority region to every row.

The module depends only on `BioSequences`, `FastaIO`, `Dates`,
`DataFrames`, `CSV` and `DelimitedFiles`.
"""
module FluStudy

using BioSequences
using FastaIO
using Dates
using DataFrames
using CSV
using DelimitedFiles

export create_study, save_study
export filter_by_region, filter_by_country
export NEXTSTRAIN_REGIONS

# ===========================================================================
# 1. Constants
# ===========================================================================

# AA -> integer encoding. Every ambiguous / gap / terminator symbol maps to
# 1, which is the sentinel that `impute_ambiguous` looks for.
const AA_TO_INT = Dict(
    AA_Term => 1, AA_B => 1, AA_J => 1, AA_X => 1, AA_Z => 1, AA_Gap => 1,
    AA_A => 2,  AA_C => 3,  AA_D => 4,  AA_E => 5,  AA_F => 6,
    AA_G => 7,  AA_H => 8,  AA_I => 9,  AA_K => 10, AA_L => 11,
    AA_M => 12, AA_N => 13, AA_P => 14, AA_Q => 15, AA_R => 16,
    AA_S => 17, AA_T => 18, AA_V => 19, AA_W => 20, AA_Y => 21,
)

const Q_DEFAULT       = 21      # alphabet size (20 canonical AAs + 1 sink)
const AMBIGUOUS_STATE = 1       # encoded value for every ambiguous AA

const PROTEIN_LENGTHS = Dict("HA" => 566, "NA" => 469)

const IGNORED_HEADER_FIELDS = ("?", "")
const DATE_FIELDS = ("date", "Date")

# Separator used inside the metadata CSV to glue multiple strain names that
# share the same (day, sequence) into a single cell.
const NAME_LIST_SEPARATOR = ";"

# Nextstrain regions, as used by FluPredictibility's `filter_by_region!`.
# These are the exact strings expected in the FASTA's `region` header
# field. Strain values are matched literally (case-sensitive) against
# this list.
const NEXTSTRAIN_REGIONS = [
    "north_america", "south_america", "europe", "china", "oceania",
    "southeast_asia", "japan_korea", "south_asia", "west_asia", "africa",
]

# ===========================================================================
# 2. Strain type + FASTA reading (replaces FluPredictibility / BioTools)
# ===========================================================================

"""
    Strain

One amino-acid sequence plus the header fields parsed from the FASTA.
"""
mutable struct Strain
    seq::LongSequence{AminoAcidAlphabet}
    data::Dict{String,Any}
end

"""
    parse_date(s) -> Union{Date,Missing}

Parse a date string. Tolerates `YYYY-MM-XX` (unknown day -> 1st) and
returns `missing` when only the year is known or parsing fails.
"""
function parse_date(s::AbstractString)
    try
        return Date(s)
    catch
        if !isnothing(match(r"-[0-9][0-9]*-XX$", s))
            return Date(s[1:end-3])              # drop "-XX"
        else
            return missing
        end
    end
end

"""
    parse_header(header, fields; separator='|') -> Dict
"""
function parse_header(header::AbstractString,
                      fields::AbstractVector;
                      separator::Char = '|')
    parts = split(header, separator)
    out = Dict{String,Any}()
    for (i, f) in enumerate(fields)
        i > length(parts) && break
        sf = String(f)
        sf in IGNORED_HEADER_FIELDS && continue
        out[sf] = sf in DATE_FIELDS ? parse_date(String(parts[i])) :
                                      String(parts[i])
    end
    return out
end

"""
    read_fasta_strains(path, header_fields; separator='|',
                       ignore_read_errors=true) -> Vector{Strain}
"""
function read_fasta_strains(path::AbstractString,
                            header_fields::AbstractVector;
                            separator::Char         = '|',
                            ignore_read_errors::Bool = true)
    strains = Strain[]
    ntot    = 0
    nunread = 0
    println("Reading $path ...")
    for (n, s) in FastaReader(path)
        ntot += 1
        dat  = parse_header(n, header_fields; separator = separator)
        seq  = try
            LongSequence{AminoAcidAlphabet}(s)
        catch err
            if ignore_read_errors
                nothing
            elseif occursin('X', s) || occursin('x', s)
                LongSequence{AminoAcidAlphabet}(replace(s, r"[Xx]" => "N"))
            else
                println(s); rethrow(err)
            end
        end
        if seq === nothing
            nunread += 1
        else
            push!(strains, Strain(seq, dat))
        end
    end
    println("Read $(length(strains)) strains out of $ntot. Could not read $nunread.")
    return strains
end

# ===========================================================================
# 3. Strain-level filters
# ===========================================================================

count_gaps(s::LongSequence{AminoAcidAlphabet}) =
    BioSequences.count(x -> isgap(x) || isambiguous(x), s)

passes_gap_filter(s::Strain; threshold::Real = 0.05) =
    count_gaps(s.seq) < threshold * length(s.seq)

has_date(s::Strain) = haskey(s.data, "date") && !ismissing(s.data["date"])

"""
    deduplicate_by_strain_name(strains) -> Vector{Strain}

Collapse strains that share the same `"strain"` header value, keeping
the LAST occurrence (matches the original `FluPop` constructor, which
builds `Dict(strain_name => strain)` for-comprehension: later records
silently overwrite earlier ones).

Strains without a `"strain"` field, or with an empty one, are kept as-is.
"""
function deduplicate_by_strain_name(strains::AbstractVector{Strain})
    seen = Dict{String,Strain}()
    no_name = Strain[]
    for s in strains
        name = String(get(s.data, "strain", ""))
        if isempty(name)
            push!(no_name, s)
        else
            seen[name] = s          # last write wins -- matches original
        end
    end
    out = collect(values(seen))
    append!(out, no_name)
    return out
end

"""
    strain_field(s, field; missing_label="") -> String

Return the value of `s.data[field]` as a `String`, or `missing_label`
when the field is absent. Used by `filter_by_region` and
`filter_by_country` to extract the comparison value from each strain.
"""
function strain_field(s::Strain, field::AbstractString;
                      missing_label::AbstractString = "")
    val = get(s.data, field, missing_label)
    return String(val)
end

"""
    filter_by_region(strains, regions) -> Vector{Strain}

Keep only strains whose `region` header field matches one of `regions`
(exact-match, case-sensitive). `regions` may be a single string or a
collection.

This mirrors `FluPredictibility.filter_by_region!`. The canonical
nextstrain region strings are stored in `NEXTSTRAIN_REGIONS`:
`$(["north_america", "south_america", "europe", "china", "oceania",
"southeast_asia", "japan_korea", "south_asia", "west_asia", "africa"])`.

Strains whose `region` field is missing or empty are dropped when a
filter is requested.
"""
function filter_by_region(strains::AbstractVector{Strain}, regions)
    requested = regions isa AbstractString ? Set([String(regions)]) :
                                              Set(String.(regions))

    unknown = setdiff(requested, Set(NEXTSTRAIN_REGIONS))
    if !isempty(unknown)
        @warn "Region label(s) not in NEXTSTRAIN_REGIONS: " *
              "$(collect(unknown)). Filter will still try them, but the " *
              "canonical labels are $(NEXTSTRAIN_REGIONS)."
    end

    return filter(s -> strain_field(s, "region") in requested, strains)
end

"""
    filter_by_country(strains, countries) -> Vector{Strain}

Keep only strains whose `country` header field matches one of
`countries` (exact-match, case-sensitive). `countries` may be a single
string or a collection. Mirrors
`FluPredictibility.filter_by_country!`.

Strains whose `country` field is missing or empty are dropped when a
filter is requested.
"""
function filter_by_country(strains::AbstractVector{Strain}, countries)
    requested = countries isa AbstractString ? Set([String(countries)]) :
                                                Set(String.(countries))
    return filter(s -> strain_field(s, "country") in requested, strains)
end

# ===========================================================================
# 4. Date binning
# ===========================================================================

"""
    bin_strains_by_date(strains; start, stop, binwidth, binspacing)
        -> Dict{Tuple{Date,Date}, Vector{Strain}}

Group `strains` (each with a usable `"date"` field) into bins keyed by
`(left, left+binwidth)`.

To match the original `FluPredictibility.bin_by_date!` exactly, a strain
is placed into bin `(L, R)` iff `L < date < R`  -- i.e. STRICTLY greater
than the left edge of the first bin and STRICTLY less than the right edge
of the last bin (the original drops strains dated exactly on the global
start date).
"""
function bin_strains_by_date(strains::AbstractVector{Strain};
                             start::Date,
                             stop::Date,
                             binwidth::Day,
                             binspacing::Day)
    bins = Dict{Tuple{Date,Date}, Vector{Strain}}()
    d = start + binwidth
    while d < stop                         # original: `now < stopdate`
        bins[(d - binwidth, d)] = Strain[]
        d += binspacing
    end
    isempty(bins) && return bins

    sorted_keys = sort(collect(keys(bins)))
    first_left  = sorted_keys[1][1]
    last_right  = sorted_keys[end][2]
    lefts       = first.(sorted_keys)

    for s in strains
        has_date(s) || continue
        dt = s.data["date"]::Date
        (dt <= first_left || dt >= last_right) && continue   # strict gates
        idx = searchsortedlast(lefts, dt)
        idx == 0 && continue
        key = sorted_keys[idx]
        dt < key[2] && push!(bins[key], s)
    end
    return bins
end

# ===========================================================================
# 5. AA -> integer encoding and matrix construction (with dedup)
# ===========================================================================

"""
    encode_sequence(seq) -> Vector{Int}
"""
function encode_sequence(seq)
    out = Vector{Int}(undef, length(seq))
    @inbounds for i in eachindex(seq)
        out[i] = AA_TO_INT[seq[i]]
    end
    return out
end

"""
    bins_to_matrix(bins, L; legacy_dedup=false)
        -> (data, names_per_row, regions_per_row, clades_per_row)

Convert binned strains into a deduplicated numeric matrix:

  * `data`            :: `Matrix{Int}` with columns
                         `[day, count, aa_1..aa_L]`, one row per
                         *unique* sequence on each day. `count` is the
                         number of strains that carried that sequence
                         on that day.
  * `names_per_row`   :: `Vector{Vector{String}}` parallel to the rows
                         of `data`; each entry is the list of strain
                         names (from the FASTA's `strain` header field)
                         that collapsed into that row.
  * `regions_per_row` :: same shape; lists of `region` header values.
  * `clades_per_row`  :: same shape; lists of `clade` header values.

`day` is a 1-based offset from the first non-empty bin. Strains whose
sequence length differs from `L` are silently skipped.
"""
function bins_to_matrix(bins::Dict{Tuple{Date,Date},Vector{Strain}}, L::Int;
                        legacy_dedup::Bool = false)
    isempty(bins) && error("`bins` is empty - did you forget to bin by date?")

    sorted_bins = sort(collect(keys(bins)))
    rows             = Vector{Vector{Int}}()
    names_per_row    = Vector{Vector{String}}()
    regions_per_row  = Vector{Vector{String}}()
    clades_per_row   = Vector{Vector{String}}()
    n_skip_len       = 0

    first_nonempty = findfirst(b -> !isempty(bins[b]), sorted_bins)
    if first_nonempty === nothing
        error("Every date bin between $(first(sorted_bins)[1]) and " *
              "$(last(sorted_bins)[2]) is empty, so there is nothing to build a matrix " *
              "from: either no record passed the date and gap filters, or every date " *
              "falls outside that range. The counts logged above say which filter " *
              "removed them.")
    end
    offset = first_nonempty - 1     # bin index `first_nonempty` -> day 1

    for (bin_idx, bin) in enumerate(sorted_bins)
        day = bin_idx - offset
        day < 1 && continue
        strains = bins[bin]
        isempty(strains) && continue

        # Build the bin's length-L encoded MSA preserving insertion order.
        # `annot` is a vector of per-strain attribute columns; each column
        # is a Vector{String} parallel to `encoded`.
        encoded = Vector{Vector{Int}}()
        annot   = [Vector{String}() for _ in 1:3]   # [names, regions, clades]
        for s in strains
            if length(s.seq) != L
                n_skip_len += 1
                continue
            end
            push!(encoded, encode_sequence(s.seq))
            push!(annot[1], String(get(s.data, "strain", "")))
            push!(annot[2], strain_field(s, "region"))
            push!(annot[3], strain_field(s, "clade"))
        end
        isempty(encoded) && continue

        if legacy_dedup
            # Reproduce the original `select_unique` + `eq_seq` +
            # `check_seq_equal` behaviour (sum-of-difference == 0).
            # This is mathematically incorrect (it merges sequences that
            # happen to share a row-sum) but matches the legacy output.
            row_keys, row_counts, row_annot = _legacy_select_unique(encoded, annot)
        else
            row_keys, row_counts, row_annot = _strict_select_unique(encoded, annot)
        end

        for k in eachindex(row_keys)
            row = Vector{Int}(undef, L + 2)
            row[1] = day
            row[2] = row_counts[k]
            row[3:end] .= row_keys[k]
            push!(rows, row)
            push!(names_per_row,   row_annot[1][k])
            push!(regions_per_row, row_annot[2][k])
            push!(clades_per_row,  row_annot[3][k])
        end
    end

    @info "bins_to_matrix: kept $(length(rows)) unique (day, sequence) rows; " *
          "skipped $n_skip_len strains with wrong length. " *
          "(legacy_dedup=$legacy_dedup)"

    data = Matrix{Int}(undef, length(rows), L + 2)
    @inbounds for i in eachindex(rows)
        data[i, :] .= rows[i]
    end
    return data, names_per_row, regions_per_row, clades_per_row
end

"""
    _strict_select_unique(encoded, annot)
        -> (uniq_seqs, counts, annot_per_uniq)

Correct deduplication: two rows merge iff their integer-encoded
sequences are element-wise equal. `annot` is a vector of per-strain
attribute columns (each a `Vector{String}` parallel to `encoded`); the
returned `annot_per_uniq` is the same length, where each column entry
is now a `Vector{Vector{String}}` (the per-row list of merged values).
"""
function _strict_select_unique(encoded::Vector{Vector{Int}},
                               annot::Vector{Vector{String}})
    n_cols = length(annot)
    # groups[encoded_seq] = (count, [Vector{String} per annotation column])
    groups = Dict{Vector{Int}, Tuple{Int,Vector{Vector{String}}}}()
    order  = Vector{Vector{Int}}()
    for i in eachindex(encoded)
        enc = encoded[i]
        if haskey(groups, enc)
            c, cols = groups[enc]
            for j in 1:n_cols
                push!(cols[j], annot[j][i])
            end
            groups[enc] = (c + 1, cols)
        else
            cols = [String[annot[j][i]] for j in 1:n_cols]
            groups[enc] = (1, cols)
            push!(order, enc)
        end
    end
    uniq_seqs = order
    counts    = [groups[s][1] for s in order]
    # Repack as a vector of columns, each a Vector{Vector{String}} of length |order|.
    annot_per_uniq = [[groups[s][2][j] for s in order] for j in 1:n_cols]
    return uniq_seqs, counts, annot_per_uniq
end

"""
    _legacy_select_unique(encoded, annot)
        -> (uniq_seqs, counts, annot_per_uniq)

Reproduce the original `select_unique` + `eq_seq` + `check_seq_equal`
behaviour exactly. The legacy comparison is `sum(seq1 .- seq2) == 0`,
which incorrectly treats sequences with the same total integer-encoded
sum as equal. We replicate that bug here for legacy compatibility.
"""
function _legacy_select_unique(encoded::Vector{Vector{Int}},
                               annot::Vector{Vector{String}})
    n_cols = length(annot)
    M = length(encoded)
    sums = [sum(s) for s in encoded]
    taken = falses(M)
    uniq_seqs       = Vector{Vector{Int}}()
    counts          = Int[]
    rows_annot      = [Vector{Vector{String}}() for _ in 1:n_cols]

    @inbounds for i in 1:M
        taken[i] && continue
        # `sim_sq` = all positions whose row-sum equals sums[i] (the
        # original's buggy criterion). Note this INCLUDES `i` itself.
        sim_sq = Int[]
        for j in 1:M
            if sums[j] == sums[i]
                push!(sim_sq, j)
            end
        end
        push!(uniq_seqs, encoded[i])
        push!(counts, length(sim_sq))
        for j in 1:n_cols
            push!(rows_annot[j], [annot[j][k] for k in sim_sq])
        end
        for j in sim_sq
            taken[j] = true
        end
    end
    return uniq_seqs, counts, rows_annot
end

# ===========================================================================
# 6. Frequency-trajectory machinery
# ===========================================================================

function normalise_counts(times::AbstractVector{<:Real},
                          counts::AbstractVector{<:Real})
    n   = length(counts)
    out = similar(counts, Float64)
    i   = 1
    @inbounds while i <= n
        j, s = i, 0.0
        while j <= n && times[j] == times[i]
            s += counts[j]; j += 1
        end
        if s == 0.0
            for k in i:j-1; out[k] = 0.0; end
        else
            inv_s = 1.0 / s
            for k in i:j-1; out[k] = counts[k] * inv_s; end
        end
        i = j
    end
    return out
end

function gaussian_smooth(counts_norm::AbstractVector{Float64},
                         times::AbstractVector{<:Real},
                         t::Real, tsmooth::Real)
    w    = zeros(length(counts_norm))
    tcut = 3 * tsmooth
    norm = 1 / (tsmooth * sqrt(2π))
    @inbounds for k in eachindex(counts_norm)
        if abs(times[k] - t) <= tcut
            w[k] = counts_norm[k] *
                   exp(-((times[k] - t) / tsmooth)^2 / 2) * norm
        end
    end
    return w
end

function single_site_frequencies(seqs::AbstractMatrix{<:Integer},
                                 weights::AbstractVector{Float64};
                                 q::Int = Q_DEFAULT)
    M, L = size(seqs)
    f1   = zeros(L * q)
    N    = sum(weights)
    N == 0 && return f1
    @inbounds for m in 1:M
        w = weights[m]
        w == 0 && continue
        for j in 1:L
            f1[(j - 1) * q + seqs[m, j]] += w
        end
    end
    return f1 ./ N
end

"""
    frequency_trajectory(seqs, counts, times; tstep=10.0, tsmooth=60.0,
                          q=Q_DEFAULT) -> (freq, t_grid)
"""
function frequency_trajectory(seqs::AbstractMatrix{<:Integer},
                              counts::AbstractVector{<:Real},
                              times::AbstractVector{<:Real};
                              tstep::Real   = 10.0,
                              tsmooth::Real = 60.0,
                              q::Int        = Q_DEFAULT)
    counts_norm = normalise_counts(times, Float64.(counts))
    t_grid = collect(1.0:Float64(tstep):Float64(last(unique(times))))
    L = size(seqs, 2)
    freq = zeros(L * q, length(t_grid))
    @inbounds for (i, t) in enumerate(t_grid)
        w = gaussian_smooth(counts_norm, times, t, tsmooth)
        freq[:, i] .= single_site_frequencies(seqs, w; q = q)
    end
    return freq, t_grid
end

# ===========================================================================
# 7. Ambiguous-state imputation
# ===========================================================================

"""
    impute_ambiguous(data; tstep=10.0, tsmooth=60.0, q=Q_DEFAULT)
        -> (data_imputed, n_changes)
"""
function impute_ambiguous(data::AbstractMatrix{<:Integer};
                          tstep::Real   = 10.0,
                          tsmooth::Real = 60.0,
                          q::Int        = Q_DEFAULT)
    times  = @view data[:, 1]
    counts = @view data[:, 2]
    seqs   = data[:, 3:end]

    freq, t_grid = frequency_trajectory(seqs, counts, times;
                                        tstep = tstep, tsmooth = tsmooth, q = q)

    seqs_new  = copy(seqs)
    n_changes = 0
    @inbounds for i in axes(seqs, 1)
        amb = findall(==(AMBIGUOUS_STATE), @view seqs[i, :])
        isempty(amb) && continue
        t_idx = argmin(abs.(t_grid .- times[i]))
        for k in amb
            seqs_new[i, k] = argmax(@view freq[(k - 1) * q + 1 : k * q, t_idx])
            n_changes += 1
        end
    end
    @info "impute_ambiguous: replaced $n_changes ambiguous residues."

    out = copy(data)
    out[:, 3:end] .= seqs_new
    return out, n_changes
end

# ===========================================================================
# 8. Outlier filtering (> mut_threshold mutations vs yearly reference)
# ===========================================================================

hamming_int(a::AbstractVector, b::AbstractVector) =
    (@assert length(a) == length(b); sum(a[i] != b[i] for i in eachindex(a)))

function most_common_sequence(seqs::AbstractMatrix, counts::AbstractVector)
    unique_rows = unique(eachrow(seqs))
    weights = zeros(length(unique_rows))
    for (k, r) in enumerate(unique_rows)
        weights[k] = sum(counts[i] for i in axes(seqs, 1) if seqs[i, :] == r)
    end
    return collect(unique_rows[argmax(weights)])
end

function yearly_windows(times::AbstractVector{<:Real}, delta_t::Real)
    windows = Vector{Vector{eltype(times)}}()
    n = length(times)
    i = 1
    while i <= n
        best_j, best_diff = i, Inf
        for j in i:n
            d    = times[j] - times[i]
            diff = abs(d - delta_t)
            if diff < best_diff
                best_j, best_diff = j, diff
            end
            d > delta_t && diff > best_diff && break
        end
        push!(windows, times[i:best_j])
        i = best_j + 1
    end
    return windows
end

"""
    filter_outliers(data; window_days=360.0, mut_threshold=20) -> keep_mask
"""
function filter_outliers(data::AbstractMatrix{<:Real};
                         window_days::Real   = 360.0,
                         mut_threshold::Real = 20)
    times   = @view data[:, 1]
    counts  = @view data[:, 2]
    seqs    = @view data[:, 3:end]
    u_times = unique(times)

    keep = trues(size(data, 1))
    for win in yearly_windows(Float64.(u_times), Float64(window_days))
        t_left, t_right = first(win), last(win)
        idx_left  = findfirst(==(t_left),  times)
        idx_right = findlast(==(t_right), times)

        ref_idx = findall(==(t_left), times)
        ref_seq = most_common_sequence(seqs[ref_idx, :],
                                       Float64.(counts[ref_idx]))

        for i in idx_left:idx_right
            if hamming_int(ref_seq, seqs[i, :]) > mut_threshold
                keep[i] = false
            end
        end
    end
    @info "filter_outliers: dropping $(count(!, keep)) of $(length(keep)) rows."
    return keep
end

# ===========================================================================
# 9. Date-range clipping
# ===========================================================================

"""
    clip_date_range(data, names_per_row, regions_per_row, clades_per_row;
                    start_date, stop_date, origin_date)
        -> (data_clip, names_per_row_clip, regions_per_row_clip,
            clades_per_row_clip)

Keep only rows whose `day` falls inside `[start_date, stop_date]`
measured from `origin_date`.
"""
function clip_date_range(data::AbstractMatrix{<:Integer},
                         names_per_row::AbstractVector{<:AbstractVector{<:AbstractString}},
                         regions_per_row::AbstractVector{<:AbstractVector{<:AbstractString}},
                         clades_per_row::AbstractVector{<:AbstractVector{<:AbstractString}};
                         start_date::Date,
                         stop_date::Date,
                         origin_date::Date)
    d_start = Dates.value(start_date - origin_date) + 1
    d_stop  = Dates.value(stop_date  - origin_date) + 1
    keep = (data[:, 1] .>= d_start) .& (data[:, 1] .<= d_stop)
    return (data[keep, :],
            names_per_row[keep],
            regions_per_row[keep],
            clades_per_row[keep])
end

# ===========================================================================
# 10. Clade lookup
# ===========================================================================

"""
    load_clade_map(path; isolate_col=:isolate, clade_col=:clade) -> Dict{String,String}
"""
function load_clade_map(path::AbstractString;
                        isolate_col::Symbol = :isolate,
                        clade_col::Symbol   = :clade)
    df = DataFrame(CSV.File(path))
    return Dict(String(row[isolate_col]) => String(row[clade_col])
                for row in eachrow(df))
end

"""
    majority_value(values; missing_labels=("", "unassigned", "unknown"))
        -> (label, agree)

For one row's list of per-strain string values, return the most frequent
label among the *non-missing* entries, and a boolean saying whether all
non-missing entries agreed.

A value is treated as missing if it is in `missing_labels`. If every
entry is missing, the function returns the most common ACTUAL value
that appeared (e.g. preserves `"unassigned"` rather than collapsing to
`""`).
"""
function majority_value(values::AbstractVector{<:AbstractString};
                        missing_labels = ("", "unassigned", "unknown"))
    miss_set = Set(String.(missing_labels))
    known = filter(v -> !(v in miss_set), values)
    if isempty(known)
        # No real labels at all -- preserve the actual missing token
        # that appeared (e.g. "unassigned"). Fall back to the first
        # configured missing label only if `values` itself is empty.
        isempty(values) && return String(first(missing_labels)), true
        counts = Dict{String,Int}()
        for v in values; counts[v] = get(counts, v, 0) + 1; end
        chosen, _ = first(sort(collect(counts), by = x -> -x[2]))
        return chosen, length(unique(values)) == 1
    end
    counts = Dict{String,Int}()
    for v in known; counts[v] = get(counts, v, 0) + 1; end
    chosen, _ = first(sort(collect(counts), by = x -> -x[2]))
    return chosen, length(unique(known)) == 1
end

# ===========================================================================
# 11. Orchestrator
# ===========================================================================

"""
    create_study(fasta_path; kwargs...) -> NamedTuple

Run the full pipeline. See the module docstring for the stage list.

# Keyword arguments
* `protein`        - `"HA"` or `"NA"`; selects the expected length.
* `protein_length` - expected AA length; defaults to `PROTEIN_LENGTHS[protein]`.
* `start_date`     - inclusive lower date of the study window.
* `stop_date`      - inclusive upper date of the study window.
* `bin_origin`     - date that maps to bin index 1 (default `start_date`).
* `bin_stop`       - date the binning runs up to. Default: max date in
                     the FASTA (so the imputation smoothing has access
                     to data after `stop_date`).
* `binwidth`       - `Day` width of each bin (default `Day(1)`).
* `binspacing`     - spacing between consecutive bins (default `Day(1)`).
* `header_fields`  - FASTA header field names.
* `separator`      - FASTA header separator (default `'|'`).
* `gap_threshold`  - max gap fraction (default 0.05).
* `window_days`    - outlier-filter window size in days (default 360).
* `mut_threshold`  - max mutations vs. yearly reference (default 20).
* `tstep`          - frequency-grid step in days (default 10).
* `tsmooth`        - Gaussian smoothing σ in days (default 60).
* `q`              - alphabet size (default 21).
* `deduplicate_by_name` - if `true` (default), strains sharing the same
                     `strain` header field are collapsed before binning,
                     keeping the last occurrence. This matches the
                     original `FluPredictibility.FluPop` constructor's
                     `Dict(strain_name => strain)` behaviour. Set to
                     `false` if your FASTA has been pre-cleaned.
* `legacy_dedup`   - per-bin dedup mode. `false` (default) uses strict
                     element-wise sequence equality. `true` reproduces
                     the original pipeline's `check_seq_equal` test
                     (`sum(seq1 .- seq2) == 0`), which merges sequences
                     whose integer-encoded vectors share the same
                     row-sum. The legacy mode is mathematically
                     incorrect (genuinely different sequences can
                     collide) and is provided only for backward
                     compatibility with previously generated outputs.
* `region`         - optional region filter. Pass a single label
                     (`"oceania"`), a collection
                     (`["europe", "north_america"]`), or the literal
                     string `"all"` (case-insensitive) to keep every
                     strain. Canonical nextstrain labels are in
                     `NEXTSTRAIN_REGIONS`: `north_america`,
                     `south_america`, `europe`, `china`, `oceania`,
                     `southeast_asia`, `japan_korea`, `south_asia`,
                     `west_asia`, `africa`. A non-`"all"` filter
                     requires `"region"` to appear in `header_fields`.
                     When `nothing` (default) or `"all"`, no filter is
                     applied; the `:region` column is still populated
                     in the metadata.
* `country`        - optional country filter. Same shape as `region`
                     but matched against the `country` header field.
                     Requires `"country"` in `header_fields`.
* `clade_csv`      - optional path to an `isolate,clade` CSV. When
                     provided, the CSV is treated as authoritative:
                     each strain's `clade` is set/overridden by the
                     CSV value. When omitted, the `clade` is read
                     directly from the FASTA's `clade` header field
                     (or filled in as `"unassigned"` if absent).

# Returns
A `NamedTuple` with fields:
  * `data`      :: `Matrix{Int}`     - `[day, count, aa_1..aa_L]`, one row
                                       per unique (day, sequence).
  * `frequency` :: `Matrix{Float64}` - `L*q` x `T` smoothed trajectory.
  * `t_grid`    :: `Vector{Float64}` - days at which `frequency` is sampled.
  * `metadata`  :: `DataFrame`       - one row per row of `data`, columns:
                       `:row`              (1..size(data, 1))
                       `:day`              (== data[:, 1])
                       `:count`            (== data[:, 2])
                       `:strain_names`     (joined with NAME_LIST_SEPARATOR)
                       `:n_strains`        (length of the strain list)
                       `:clade`            (majority clade among merged)
                       `:clade_agreement`  (all known clades agree?)
                       `:region`           (majority region among merged)
                       `:region_agreement` (all known regions agree?)
  * `name_index`:: `DataFrame`       - one row per individual strain, with
                       `:strain_name`, `:data_row`, `:day`, `:clade`,
                       `:region` -- inverse map (strain -> data row).
"""
function create_study(fasta_path::AbstractString;
                      protein::AbstractString = "HA",
                      protein_length::Int     = PROTEIN_LENGTHS[protein],
                      start_date::Date,
                      stop_date::Date,
                      bin_origin::Date        = start_date,
                      bin_stop::Union{Date,Nothing} = nothing,
                      binwidth::Day           = Day(1),
                      binspacing::Day         = Day(1),
                      header_fields           = ["strain", "isolate_id",
                                                 "date", "segment", "hash",
                                                 "region", "clade"],
                      separator::Char         = '|',
                      gap_threshold::Real     = 0.05,
                      window_days::Real       = 360.0,
                      mut_threshold::Real     = 20,
                      tstep::Real             = 10.0,
                      tsmooth::Real           = 60.0,
                      q::Int                  = Q_DEFAULT,
                      deduplicate_by_name::Bool = true,
                      legacy_dedup::Bool      = false,
                      region::Union{AbstractString,AbstractVector,Nothing} = nothing,
                      country::Union{AbstractString,AbstractVector,Nothing} = nothing,
                      clade_csv::Union{AbstractString,Nothing} = nothing)

    # --- (a) read FASTA ----------------------------------------------------
    strains = read_fasta_strains(fasta_path, header_fields;
                                 separator         = separator,
                                 ignore_read_errors = true)

    # --- (b) date + gap filter ---------------------------------------------
    filter!(s -> has_date(s) && passes_gap_filter(s; threshold = gap_threshold),
            strains)
    @info "After date + gap filter: $(length(strains)) strains."
    isempty(strains) && error(
        "No record passed the date and gap filter: every one of them either carries no " *
        "parsable date in its header or is more than $(100 * gap_threshold)% gaps. The " *
        "date is read from the position of \"date\" in " *
        "`header_fields=$header_fields`, so compare that order against a header line of " *
        "the FASTA.")

    # --- (b*) backfill `clade` from a CSV when the FASTA doesn't carry it.
    #          When `clade_csv` is provided it OVERRIDES any clade value in
    #          the header (the CSV is treated as authoritative -- e.g. a
    #          Nextstrain clade table). -----------------------------------
    if clade_csv !== nothing
        clade_map = load_clade_map(clade_csv)
        n_overridden   = 0    # FASTA had a clade, CSV replaced it (agree or disagree)
        n_disagreed    = 0    # FASTA had clade X, CSV said Y != X
        n_added        = 0    # FASTA had no clade, CSV provided one
        n_kept_fasta   = 0    # CSV didn't list this strain; FASTA clade kept
        n_unassigned   = 0    # neither source had a clade
        for s in strains
            name = String(get(s.data, "strain", ""))
            csv_clade = get(clade_map, name, nothing)
            fasta_clade = get(s.data, "clade", nothing)
            if csv_clade !== nothing
                if fasta_clade === nothing || fasta_clade == "" ||
                   fasta_clade == "unassigned" || fasta_clade == "unknown"
                    n_added += 1
                else
                    n_overridden += 1
                    csv_clade != fasta_clade && (n_disagreed += 1)
                end
                s.data["clade"] = csv_clade
            elseif fasta_clade === nothing
                s.data["clade"] = "unassigned"
                n_unassigned += 1
            else
                n_kept_fasta += 1
            end
        end
        @info "Clade CSV applied: " *
              "overrode $n_overridden FASTA labels ($n_disagreed disagreed), " *
              "added $n_added new labels, " *
              "kept $n_kept_fasta FASTA labels (not in CSV), " *
              "$n_unassigned strains had no clade in either source."
    end

    # --- (b') deduplicate by strain name (matches the original FluPop, which
    #          stored strains in a Dict keyed by `strain` and silently dropped
    #          all but the last occurrence of any given strain name) ---------
    if deduplicate_by_name
        n_before = length(strains)
        strains = deduplicate_by_strain_name(strains)
        @info "After dedup by strain name: $(length(strains)) strains " *
              "(dropped $(n_before - length(strains)) duplicates)."
    end

    # --- (b'') filter by region (mirrors FluPredictibility.filter_by_region!) ----
    # `region = nothing` or `region = "all"` both skip the filter entirely.
    region_is_all = region isa AbstractString && lowercase(String(region)) == "all"
    if region !== nothing && !region_is_all
        if !("region" in String.(header_fields))
            error("Cannot filter by `region`: \"region\" is not in " *
                  "`header_fields`. Update header_fields to include " *
                  "\"region\" at the correct position for your FASTA.")
        end
        n_before = length(strains)
        strains = filter_by_region(strains, region)
        @info "After region filter ($region): $(length(strains)) " *
              "strains (dropped $(n_before - length(strains)))."
        if isempty(strains)
            error("Region filter removed every strain. Check that the " *
                  "`region` field in your FASTA actually contains the " *
                  "nextstrain labels (e.g. \"oceania\", \"europe\"). " *
                  "Inspect a header line in your FASTA and verify the " *
                  "position of `region` in `header_fields=$header_fields`.")
        end
    elseif region_is_all
        @info "Region filter: \"all\" -> no filter applied."
    end

    # --- (b''') filter by country (mirrors FluPredictibility.filter_by_country!) -
    if country !== nothing
        if !("country" in String.(header_fields))
            error("Cannot filter by `country`: \"country\" is not in " *
                  "`header_fields`. Update header_fields accordingly.")
        end
        n_before = length(strains)
        strains = filter_by_country(strains, country)
        @info "After country filter ($country): $(length(strains)) " *
              "strains (dropped $(n_before - length(strains)))."
        if isempty(strains)
            error("Country filter removed every strain. Check the " *
                  "spelling and position of `country` in " *
                  "`header_fields=$header_fields`.")
        end
    end

    # Default `bin_stop` to the latest date in the data so that imputation
    # can use future data points (this matches the original pipeline, where
    # `bin_by_date!` used `stopdate = :auto`).
    if bin_stop === nothing
        max_d = maximum(s.data["date"]::Date for s in strains)
        bin_stop = max_d + Day(1)        # +1 because bins are right-open
    end

    # --- (c) bin by date ---------------------------------------------------
    bins = bin_strains_by_date(strains;
        start      = bin_origin,
        stop       = bin_stop,
        binwidth   = binwidth,
        binspacing = binspacing,
    )

    isempty(bins) && error(
        "The range $bin_origin .. $bin_stop holds no date bin, so no record can be " *
        "kept. `bin_stop` defaults to one day past the latest date in the FASTA, so an " *
        "empty range means the study window opens after the sequences end: check " *
        "`start_date` ($start_date) against the record dates.")

    # --- (d) build numeric matrix with per-bin deduplication ---------------
    data, names_per_row, regions_per_row, clades_per_row =
        bins_to_matrix(bins, protein_length; legacy_dedup = legacy_dedup)

    # Records that land in no bin are caught by the two guards above, so by here the only
    # way to end up with no row is every record having been skipped for its length.
    size(data, 1) == 0 && error(
        "The data matrix is empty because every record was skipped for its length: none " *
        "is aligned to the $protein_length residues expected for $protein. Align the " *
        "records to $protein_length columns, or pass `protein_length` " *
        "(`--protein-length` from the scripts) to match the alignment you have.")

    # --- (e) impute ambiguous BEFORE clipping so the smoothing window has
    #         access to neighbouring time points outside [start, stop] ------
    data, _ = impute_ambiguous(data; tstep = tstep, tsmooth = tsmooth, q = q)

    # --- (f) clip to study window ------------------------------------------
    data, names_per_row, regions_per_row, clades_per_row = clip_date_range(
        data, names_per_row, regions_per_row, clades_per_row;
        start_date  = start_date,
        stop_date   = stop_date,
        origin_date = bin_origin,
    )

    # --- (g) outlier filtering ---------------------------------------------
    keep             = filter_outliers(data;
                                       window_days   = window_days,
                                       mut_threshold = mut_threshold)
    data             = data[keep, :]
    names_per_row    = names_per_row[keep]
    regions_per_row  = regions_per_row[keep]
    clades_per_row   = clades_per_row[keep]

    # Clipping and outlier filtering can both empty the matrix; `frequency_trajectory`
    # below would fail on it in a way that says nothing about the cause.
    size(data, 1) == 0 && error(
        "No row is left after clipping to the study window $start_date .. $stop_date " *
        "and dropping outliers (rows more than $mut_threshold mutations from the " *
        "consensus of their $window_days-day window). The counts logged above say which " *
        "of the two removed them.")

    # --- (i) resolve majority clade and region per row ----------------------
    row_clades        = String[]
    clade_agreements  = Bool[]
    for cs in clades_per_row
        c, agree = majority_value(cs)
        push!(row_clades, c); push!(clade_agreements, agree)
    end
    n_disagree = count(!, clade_agreements)
    n_disagree > 0 && @warn "Clade disagreement on $n_disagree row(s); " *
                            "majority clade kept; see `metadata.clade_agreement`."

    row_regions       = String[]
    region_agreements = Bool[]
    for rs in regions_per_row
        r, agree = majority_value(rs)
        push!(row_regions, r); push!(region_agreements, agree)
    end

    # --- (h) final frequency trajectory ------------------------------------
    frequency, t_grid = frequency_trajectory(
        data[:, 3:end], data[:, 2], data[:, 1];
        tstep = tstep, tsmooth = tsmooth, q = q,
    )

    # --- metadata tables ---------------------------------------------------
    # `day` is a 1-based offset from `bin_origin`; convert back to a Date for
    # human-readable output.
    row_dates = [bin_origin + Day(d - 1) for d in data[:, 1]]

    metadata = DataFrame(
        row              = 1:size(data, 1),
        day              = data[:, 1],
        date             = row_dates,
        count            = data[:, 2],
        strain_names     = [join(ns, NAME_LIST_SEPARATOR) for ns in names_per_row],
        n_strains        = length.(names_per_row),
        clade            = row_clades,
        clade_agreement  = clade_agreements,
        region           = row_regions,
        region_agreement = region_agreements,
    )

    # Inverse map: one row per individual strain -> which data row it ended
    # up in. Useful for "I have a strain name, where is it in `data`?".
    name_index = DataFrame(
        strain_name = String[],
        data_row    = Int[],
        day         = Int[],
        date        = Date[],
        clade       = String[],
        region      = String[],
    )
    for (row_idx, (ns, rs, cs)) in enumerate(zip(names_per_row,
                                                  regions_per_row,
                                                  clades_per_row))
        for (n, r, c) in zip(ns, rs, cs)
            push!(name_index, (n, row_idx, data[row_idx, 1],
                               row_dates[row_idx], c, r))
        end
    end

    # --- (j) content-addressed sequence -> clade map ----------------------
    # For each unique encoded sequence in the (post-imputation) `data`
    # matrix, pick the canonical clade label. The candidate clades come
    # directly from each merged strain's FASTA header (via
    # `clades_per_row[row_idx][k]`), with first-seen-real-label wins
    # used to break ties.
    #
    # Building from `data` (rather than from the raw `Strain` list)
    # matters because `impute_ambiguous` rewrites ambiguous positions
    # in-place, so the encoded sequences in `data` differ from those
    # of the raw strains. Downstream code (e.g. the prediction pipeline
    # building `pop1` from `data`) sees the post-imputation encoding,
    # so the map keys must match that.
    sequence_to_clade = _build_sequence_clade_map(data, clades_per_row)

    # Downstream consumers (`distributions_at_t1_t2`,
    # `simulate_medoid_distribution`, etc.) are typed for
    # `Matrix{Float64}`. Internally we keep `data` as Int while binning
    # / imputing / dedup'ing (the natural type for discrete AA codes),
    # but the public-facing matrix is Float64 so it round-trips through
    # `readdlm` (which auto-promotes) without a type surprise.
    data_f64 = Matrix{Float64}(data)

    return (data              = data_f64,
            frequency         = frequency,
            t_grid            = t_grid,
            metadata          = metadata,
            name_index        = name_index,
            sequence_to_clade = sequence_to_clade)
end

"""
    _build_sequence_clade_map(data, clades_per_row) -> DataFrame

Build a per-row sequence-to-clade map. One DataFrame row per `data`
row, with columns:

  * `day`   :: `Int`     -- the day from `data[:, 1]`.
  * `seq`   :: `String`  -- comma-joined integer encoding of the
                            row's (post-imputation) amino-acid sequence.
  * `clade` :: `String`  -- the canonical FASTA clade for that row,
                            chosen from `clades_per_row[i]` by:
                              - prefer real labels over
                                `""` / `"unassigned"` / `"unknown"`;
                              - among real labels, take the first.

This map is intentionally NOT collapsed by `seq` -- the same encoded
sequence can appear on different days with different FASTA clade
labels (e.g. nomenclature drift over years). Downstream code is
expected to pick the right (day, seq) pair for its query (typically
the rows closest to the observation date).
"""
function _build_sequence_clade_map(data::AbstractMatrix{<:Integer},
                                   clades_per_row::AbstractVector{<:AbstractVector{<:AbstractString}})
    seqs   = data[:, 3:end]
    days   = data[:, 1]
    n_rows = size(seqs, 1)
    L      = size(seqs, 2)
    @assert length(clades_per_row) == n_rows

    miss = Set(("", "unassigned", "unknown"))
    out = DataFrame(day = Int[], seq = String[], clade = String[])

    for i in 1:n_rows
        seq_str = join((Int(seqs[i, j]) for j in 1:L), ',')

        row_clades = clades_per_row[i]
        real = filter(c -> !(c in miss), row_clades)
        clade = !isempty(real) ? first(real) :
                (isempty(row_clades) ? "unassigned" : first(row_clades))

        push!(out, (Int(days[i]), seq_str, clade))
    end
    return out
end

# ===========================================================================
# 12. Saving
# ===========================================================================

"""
    save_study(study, outdir; subtype, protein, region)

Write:
  * `data_<subtype>_<protein>_<region>.txt`              -- `study.data`,       `writedlm`
  * `frequency_<subtype>_<protein>_<region>.txt`         -- `study.frequency`,  `writedlm`
  * `metadata_<subtype>_<protein>_<region>.csv`          -- `study.metadata`
  * `name_index_<subtype>_<protein>_<region>.csv`        -- `study.name_index`
  * `sequence_to_clade_<subtype>_<protein>_<region>.csv` -- `study.sequence_to_clade`

`region` is a free-form filename tag; pass `"all"` when the study was
not region-filtered.
"""
function save_study(study::NamedTuple, outdir::AbstractString;
                    subtype::AbstractString,
                    protein::AbstractString,
                    region::AbstractString)
    isdir(outdir) || mkpath(outdir)
    tag = "$(subtype)_$(protein)_$(region)"
    writedlm(joinpath(outdir, "data_$(tag).txt"),       study.data)
    writedlm(joinpath(outdir, "frequency_$(tag).txt"),  study.frequency)
    CSV.write(joinpath(outdir, "metadata_$(tag).csv"),          study.metadata)
    CSV.write(joinpath(outdir, "name_index_$(tag).csv"),        study.name_index)
    CSV.write(joinpath(outdir, "sequence_to_clade_$(tag).csv"), study.sequence_to_clade)
    return nothing
end

end # module FluStudy
