# Task 1 -- MAP selection coefficients over the full study window.
#
# 1. Build the study (unique sequences + counts per day) from the FASTA of
#    `subtype` / `protein` over the study window of the subtype.
# 2. Fit the MPL selection coefficients (`MAPestimate`) using every time point in that window.
# 3. Keep only the non-gap alleles whose frequency exceeds `freq_threshold` at two
#    consecutive time points at least once (`map_selection_table`).
# 4. Write Position, Domain (HA only), Reference, Mutation and Selection_coefficient to
#    outputs/map_estimation/map_selection_<subtype>_<protein>.csv
#
# Usage (from the project root, julia/FluGEM/):
#   julia --project=. src/map_estimation/run_map_estimation.jl [--subtype H3N2|H1N1] [--protein HA|NA] [--fasta PATH]
#   julia --project=. src/map_estimation/run_map_estimation.jl [subtype] [protein]      # positional form
#
#   Defaults: H3N2 HA, data/raw/<subtype>_<protein>.fasta.
using Dates
using DataFrames
using CSV

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(PROJECT_ROOT, "src", "shared", "mpl_inference.jl"))   # MAPestimate, intToaa
include(joinpath(PROJECT_ROOT, "src", "shared", "FluStudy.jl"))
include(joinpath(PROJECT_ROOT, "src", "shared", "cli.jl"))             # parse_cli, run_name, study_window
include(joinpath(@__DIR__, "map_selection.jl"))                        # map_selection_table
using .FluStudy

opts, pos = parse_cli(ARGS)
subtype = get(opts, "subtype", length(pos) >= 1 ? pos[1] : "H3N2")
protein = get(opts, "protein", length(pos) >= 2 ? pos[2] : "HA")
start_date, stop_date = study_window(subtype)   # day 0 must match MAPestimate's time origin

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
config = (
    # --- study ---
    subtype        = subtype,
    protein        = protein,
    fasta_path     = get(opts, "fasta", joinpath(PROJECT_ROOT, "data", "raw", "$(subtype)_$(protein).fasta")),
    start_date     = start_date,
    stop_date      = stop_date,
    q              = 21,
    # --- MAP inference ---
    tstep          = 10.0,
    tsmooth        = 60.0,
    gamma          = 20.0,
    correlation    = true,
    # --- filter / output ---
    freq_threshold = 0.01,
    output_dir     = joinpath(PROJECT_ROOT, "outputs", "map_estimation"),
)

# ---------------------------------------------------------------------------
# Study + MAP estimate over the full window
# ---------------------------------------------------------------------------
@time study = create_study(config.fasta_path;
    protein    = config.protein,
    start_date = config.start_date,
    stop_date  = config.stop_date,
    tstep      = config.tstep,
    tsmooth    = config.tsmooth,
    q          = config.q,
)
println("data: $(size(study.data))")

map_result = MAPestimate(study.data, config.stop_date, config.subtype;
    tstep       = config.tstep,
    tsmooth     = config.tsmooth,
    gamma       = config.gamma,
    q           = config.q,
    correlation = config.correlation,
)

# ---------------------------------------------------------------------------
# Filter + export
# ---------------------------------------------------------------------------
df = map_selection_table(map_result; q = config.q, freq_threshold = config.freq_threshold,
                         protein = config.protein, subtype = config.subtype)

mkpath(config.output_dir)
out_path = joinpath(config.output_dir, run_name("map_selection", config.subtype, config.protein) * ".csv")
CSV.write(out_path, df)
println("Saved $(nrow(df)) mutations to $out_path")
