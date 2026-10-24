# Task 2 -- fixation-probability prediction for one prediction date.
#
# Usage (from the project root, julia/FluGEM/):
#   julia --project=. src/fixation_prediction/run_fixation_prediction.jl [options]
#
#   --subtype H3N2|H1N1          default H3N2
#   --protein HA|NA              default HA
#   --prediction-date YYYY-MM-DD date the prediction is made: the last day of data used for the fit
#   --task-id K                  SLURM array index on a 10-day grid starting 37 days after the study
#                                start (H3N2: 2009-02-07, 2009-02-17, ...; H1N1: 2009-06-07, ...), up to 2020-01-31
#   --source fasta|txt           fasta (default): build the data matrix from the FASTA with FluStudy;
#                                txt: read a precomputed data matrix
#   --fasta PATH                 default data/raw/<subtype>_<protein>.fasta
#   --txt PATH                   default data/derived/legacy/data_<subtype>_<protein>.txt
#
#   SLURM arrays may also use the positional form:
#     run_fixation_prediction.jl <task_id | YYYY-MM-DD> [subtype] [protein] [fasta|txt]
#
#   Output: outputs/fixation_prediction/<subtype>_<protein>_<source>/
#           fixation_<subtype>_<protein>_<prediction date>.csv
using Dates
using DelimitedFiles

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(@__DIR__, "FixationPrediction.jl"))
include(joinpath(PROJECT_ROOT, "src", "shared", "FluStudy.jl"))
include(joinpath(PROJECT_ROOT, "src", "shared", "cli.jl"))       # parse_cli, run_name, study_window
using .FixationPrediction

# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
opts, pos = parse_cli(ARGS)
subtype = get(opts, "subtype", length(pos) >= 2 ? pos[2] : "H3N2")
protein = get(opts, "protein", length(pos) >= 3 ? pos[3] : "HA")
source  = get(opts, "source", length(pos) >= 4 ? pos[4] : "fasta")
start_date, stop_date = study_window(subtype)   # day 0 of the data matrix, end of the study

const TASK_STEP_DAYS = 10
task_origin = start_date + Day(37)
task_end    = Date("2020-01-31")
n_tasks     = Dates.value(task_end - task_origin) ÷ TASK_STEP_DAYS + 1

function task_prediction_date(task_id::Integer)
    1 <= task_id <= n_tasks || error("task_id $task_id out of range 1..$n_tasks for $subtype")
    return task_origin + Day(TASK_STEP_DAYS * (task_id - 1))
end

first_arg = length(pos) >= 1 ? pos[1] : nothing
prediction_date = haskey(opts, "prediction-date") ? Date(opts["prediction-date"]) :
                  haskey(opts, "task-id")         ? task_prediction_date(parse(Int, opts["task-id"])) :
                  first_arg === nothing           ? error("give --prediction-date or --task-id") :
                  occursin('-', first_arg)        ? Date(first_arg) : task_prediction_date(parse(Int, first_arg))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const GAMMA   = 20.0      # L2 regularisation of the MPL fit
const TSMOOTH = 60.0      # Gaussian smoothing width (days)
const NOISE   = 1         # 0: MAP coefficients; 1: + posterior covariance; 2: + diagonal term
const M       = 1000      # forward simulations per date

# ---------------------------------------------------------------------------
# Input data matrix: rows [day, count, seq...], day 0 = start_date
# ---------------------------------------------------------------------------
data = if source == "txt"
    Float64.(readdlm(get(opts, "txt", joinpath(PROJECT_ROOT, "data", "derived", "legacy", "data_$(subtype)_$(protein).txt"))))
elseif source == "fasta"
    # Same study settings as the EMD prediction (src/emd_prediction/run_emd_prediction.jl)
    FluStudy.create_study(get(opts, "fasta", joinpath(PROJECT_ROOT, "data", "raw", "$(subtype)_$(protein).fasta"));
        protein = protein, start_date = start_date, stop_date = stop_date, tstep = 10.0, tsmooth = TSMOOTH).data
else
    error("Unknown data source \"$source\" (use fasta or txt)")
end

# ---------------------------------------------------------------------------
# Run and save
# ---------------------------------------------------------------------------
out_dir = joinpath(PROJECT_ROOT, "outputs", "fixation_prediction", "$(subtype)_$(protein)_$(source)")
out_file = joinpath(out_dir, run_name("fixation", subtype, protein, prediction_date) * ".csv")

run_simulation(data, prediction_date; start_date = start_date, stop_date = stop_date,
               gamma = GAMMA, tsmooth = TSMOOTH, noise = NOISE, M = M, out_file = out_file)
println("Saved results to: $out_file")
