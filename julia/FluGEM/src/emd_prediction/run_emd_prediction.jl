# Task 3 -- simulation-based prediction of the future sequence distribution, scored by EMD.
#
# From a prediction date t1 the causal (denoised) pipeline predicts the sequence distribution
# at t1 + horizon with GenModel (MPL selection + Wright-Fisher medoid), GenModel-Average, MLR and
# SLM, and compares them with the baselines Null, Lower Bound, Null-Emp (raw observations) and
# Null-Emp-Denoised, by EMD and consensus Hamming distance.
#
# Usage (from the project root, julia/FluGEM/):
#   julia --project=. -t auto src/emd_prediction/run_emd_prediction.jl [options]
#
#   --subtype H3N2|H1N1          default H3N2
#   --protein HA|NA              default HA
#   --prediction-date YYYY-MM-DD date the prediction is made: the last day of data used (default 2013-02-03)
#   --horizon DAYS               how far ahead to predict; target date = prediction date + horizon (default 270)
#   --initial-date YYYY-MM-DD    first day of data used to fit the selection coefficients
#                                (default: start of the study window, 2009-01-01 for H3N2, 2009-05-01 for H1N1)
#   --fasta PATH                 input FASTA (default data/raw/<subtype>_<protein>.fasta)
#   --task-id K                  SLURM array index: prediction date = 2012-01-01 + (K - 1) months, K = 1..85
#
#   SLURM arrays may also use the positional form:  run_emd_prediction.jl <task_id> [subtype] [protein]
#
#   Example: run_emd_prediction.jl --subtype H3N2 --protein HA --initial-date 2009-01-01 \
#                                  --prediction-date 2013-02-03 --horizon 270
#
#   Cache  : data/derived/study/   (FluStudy artifacts, built from the FASTA on first use)
#   Outputs: outputs/emd_prediction/predictions_<tag>.csv + side-cars (see save_pipeline_run), with
#            <tag> = <subtype>_<protein>_<prediction date>_h<horizon>_init<initial date>_gamma<γ>_tsmooth<σ>_noise<true|false>
using Dates
using DataFrames
using Base.Threads

# Thread count is fixed at process start (julia -t N or --threads=auto).
println(stderr, "Julia thread pool: ", nthreads(), " (use julia -t N ... to change)")

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(@__DIR__, "mutation_simulation.jl"))
include(joinpath(PROJECT_ROOT, "src", "shared", "FluStudy.jl"))
include(joinpath(@__DIR__, "PredictionPipeline.jl"))
include(joinpath(PROJECT_ROOT, "src", "shared", "cli.jl"))       # parse_cli, run_name, study_window
using .FluStudy
using .PredictionPipeline

# Paths in the configuration are relative to the project root, so the study cache
# manifest stays valid if the project is moved.
cd(PROJECT_ROOT)

# ===========================================================================
# Command line
# ===========================================================================
const TASK_DATE_ORIGIN = Date("2012-01-01")
const TASK_DATE_END    = Date("2019-01-01")
const N_TASKS          = 12 * (year(TASK_DATE_END) - year(TASK_DATE_ORIGIN)) +
                         (month(TASK_DATE_END) - month(TASK_DATE_ORIGIN)) + 1

"""Prediction date of SLURM array index `task_id` (first of the month, 2012-01-01 onwards)."""
function task_prediction_date(task_id::Integer)
    1 <= task_id <= N_TASKS ||
        error("task_id $task_id out of range 1..$N_TASKS ($TASK_DATE_ORIGIN .. $TASK_DATE_END).")
    return TASK_DATE_ORIGIN + Month(task_id - 1)
end

opts, pos = parse_cli(ARGS)
subtype = get(opts, "subtype", length(pos) >= 2 ? pos[2] : "H3N2")
protein = get(opts, "protein", length(pos) >= 3 ? pos[3] : "HA")
task_id = haskey(opts, "task-id") ? parse(Int, opts["task-id"]) : (length(pos) >= 1 ? parse(Int, pos[1]) : nothing)
start_date, stop_date = study_window(subtype)

prediction_date = haskey(opts, "prediction-date") ? Date(opts["prediction-date"]) :
                  task_id !== nothing             ? task_prediction_date(task_id) : Date("2013-02-03")
horizon      = parse(Int, get(opts, "horizon", "270"))
initial_date = Date(get(opts, "initial-date", string(start_date)))
fasta_path   = get(opts, "fasta", "data/raw/$(subtype)_$(protein).fasta")

horizon > 0 || error("--horizon must be a positive number of days")
start_date <= initial_date < prediction_date ||
    error("need $start_date <= initial date ($initial_date) < prediction date ($prediction_date)")
prediction_date + Day(horizon) <= stop_date ||
    error("prediction date + horizon ($(prediction_date + Day(horizon))) is after the end of the study ($stop_date)")

println(stderr, "subtype=$subtype protein=$protein initial_date=$initial_date prediction_date=$prediction_date " *
                "horizon=$horizon (target date $(prediction_date + Day(horizon))) fasta=$fasta_path")

# ===========================================================================
# Configuration
# ===========================================================================
# `save_config(config, "config.toml")` dumps this configuration to TOML;
# `config = load_config("config.toml")` loads it back.
config = PredictionConfig(
    # --- study identifiers ---
    subtype             = subtype,
    protein             = protein,
    region              = "all",
    # --- study inputs (clade labels come from the FASTA headers) ---
    fasta_path          = fasta_path,
    start_date          = start_date,
    stop_date           = stop_date,
    tstep               = 10.0,
    tsmooth             = 60.0,
    # --- filesystem ---
    study_dir           = "data/derived/study",
    output_dir          = "outputs/emd_prediction",
    # --- prediction-pipeline parameters ---
    observation_date    = prediction_date,
    initial_date        = initial_date,
    Δ                   = horizon,
    gamma               = 20.0,
    K                   = 1000,
    M                   = 100,
    K_slm               = 1000,
    correlation         = true,
    slm                 = true,
    freq_threshold      = 1e-4,
    medoid_threshold    = 1e-4,
    # --- trajectory ---
    n_snapshots_past    = 20,
    n_snapshots_models  = 10,
    clade_initial_date  = Date("2011-01-01"),   # past panel starts here;
                                                # independent of initial_date
    min_clade_freq      = 0.01,
)

# ===========================================================================
# Study: load from cache or build from FASTA
# ===========================================================================
# `load_or_create_study` reads the FluStudy artifacts from `config.study_dir`, or runs
# `FluStudy.create_study` and saves them. A manifest TOML next to the data files records
# which settings produced the cache; if the current config disagrees, it errors and asks
# you to clear the directory.
@time study = load_or_create_study(config)
data = study.data

# ===========================================================================
# Run the prediction pipeline (causal / leakage-free denoised path)
# ===========================================================================
@time pipeline_result = run_prediction_pipeline_short_ext_denoised(
    data,
    config.observation_date,
    config.Δ,
    config.subtype;
    slm              = config.slm,
    K_slm            = config.K_slm,
    correlation      = config.correlation,
    gamma            = config.gamma,
    q                = config.q,
    tstep            = config.tstep,
    tsmooth          = config.tsmooth,
    noise            = config.noise,
    N                = config.N,
    Ne               = config.Ne,
    K                = config.K,
    M                = config.M,
    freq_threshold   = config.freq_threshold,
    medoid_threshold = config.medoid_threshold,
    user_start_date  = config.initial_date,
);

# print(distance_summary_table_short_ext(pipeline_result))

# ===========================================================================
# Persist the run (predictions, h-vectors, side-car distributions)
# ===========================================================================
tag = run_name("", config.subtype, config.protein, config.observation_date;
               h = config.Δ, init = config.initial_date, gamma = config.gamma,
               tsmooth = config.tsmooth, noise = config.noise)
paths = save_pipeline_run(
    pipeline_result,
    config.output_dir;
    subtype  = config.subtype,
    protein  = config.protein,
    region   = config.region,
    obs_date = config.observation_date,
    horizon  = config.Δ,
    tag      = tag,
    config   = Dict{Symbol,Any}(
        :initial_date       => config.initial_date,
        :tstep              => config.tstep,
        :tsmooth            => config.tsmooth,
        :gamma              => config.gamma,
        :q                  => config.q,
        :N                  => config.N,
        :Ne                 => config.Ne,
        :K                  => config.K,
        :M                  => config.M,
        :K_slm              => config.K_slm,
        :n_snapshots_past   => config.n_snapshots_past,
        :n_snapshots_models => config.n_snapshots_models,
        :correlation        => config.correlation,
        :slm_enabled        => config.slm,
        :noise              => config.noise,
        :freq_threshold     => config.freq_threshold,
        :medoid_threshold   => config.medoid_threshold,
    ),
)
println("Saved ", paths.summary)
