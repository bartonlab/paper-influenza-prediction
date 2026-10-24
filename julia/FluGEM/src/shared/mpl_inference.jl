# Shared MPL inference kernel, used by all three tasks: the MAP estimation (src/map_estimation),
# the fixation prediction (src/fixation_prediction, inside its module) and the simulation-based
# EMD prediction (src/emd_prediction). Include it into the calling scope:
#
#   include(joinpath(PROJECT_ROOT, "src", "shared", "mpl_inference.jl"))

include("mpl.jl")                  # processStandard(NoCorr), Cintegrate, normalize_sc, MAPestimate
include("misc.jl")                 # closest_utime
include("aa_encoding.jl")          # intToaa, charToaa, aaToint
include("gaussian_smoothing.jl")   # normalizeCounts, smoothCounts, smoothCounts_normalized
include("Population_dynamics.jl")  # compute_f1, compute_path_f_i, fitness, smoothSpecies WF evolution
