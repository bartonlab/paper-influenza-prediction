"""
Simulation script.

Parameters:
- arg1 (String): Observation time.
- arg2 (String): Variant name.
- arg3 (String): Protein name.

"""
arg1 = ARGS[1]
arg2 = ARGS[2]
arg3 = ARGS[3]




using StatsBase, Dates, Measures, DataStructures, Random, Statistics, DelimitedFiles, DataFrames
using CSV

include("mutation_simulation.jl")
include("misc.jl")


# Set the data  directory to the parent directory of the script
wdir = dirname(@__DIR__)
cd(wdir)


println("Parameters Determination: Variant = $arg2, Protein = $arg3")

# Set data size parameters, L is Genome length and T number of bins
(L,T)=paramAlign(arg2,arg3)

directory_path = "data/$arg2/$arg3/alignments/"


println("Loading Sequence Data from $directory_path")

# Load sequence data
sVec = Matrix{Float64}[]
for i in 1:T
    pattern_sVec = r"sVec.*_" * string(i) * r"_.*\.txt"
    matching_files_sVec = filter(x -> occursin(pattern_sVec, x), readdir(directory_path))
    
    try
        push!(sVec,readdlm( "$(directory_path)/$(matching_files_sVec[1])"))
    catch e
        push!(sVec, Matrix{Float64}(undef, 0, L))
    end
end

nVec = Vector{Int64}[]
for i in 1:T
    pattern_nVec = r"nVec.*_" * string(i) * r"_.*\.txt"
    matching_files_nVec = filter(x -> occursin(pattern_nVec, x), readdir(directory_path))
    
    try
        push!(nVec, vcat(readdlm( "$(directory_path)/$(matching_files_nVec[1])")...))
    catch e
        push!(nVec, [])
    end
end    

# Initialize simulation containers
sites = []
state = []
reference = []
sigmas = []
frequency = []
gammas = []
selection_coefficient = []
av_fitness_mut = []
av_fitness_wt = []
porcent_simulation_fix = []
porcent_simulation_lost = []
mean_fixation_time = []
mean_lost_time = []
fixation = []
time = []

T0 = parse(Int64, arg1)




gamma = 20.0
sigma = 1
q = 21



println("Simulation Initialization: T0 = $T0, Sigma = $sigma, Gamma = $gamma, q = $q")

tr = init_traject(sVec, nVec, T0, sigma, gamma, q=21)

println("Trajectories Initialized...")

tr_sim = simulate_and_update_scores(tr, sVec, nVec, sigma = sigma, gamma = gamma, q = 21, noise = true)

println("Simulation Executed Successfully!")

# Collect simulation results
push!(sites, map(x -> x.i, tr_sim))
push!(state, map(x -> x.val, tr_sim))
push!(reference, map(x -> x.reference, tr_sim))
push!(sigmas, map(x -> x.sigma, tr_sim))
push!(frequency, map(x -> x.freq, tr_sim))
push!(gammas, map(x -> x.gamma, tr_sim))
push!(selection_coefficient, map(x -> x.sc, tr_sim))
push!(av_fitness_mut, map(x -> x.av_fitness_mut, tr_sim))
push!(av_fitness_wt, map(x -> x.av_fitness_wt, tr_sim))
push!(porcent_simulation_fix, map(x -> x.porcent_simulation_fix, tr_sim))
push!(porcent_simulation_lost, map(x -> x.porcent_simulation_lost, tr_sim))
push!(mean_fixation_time, map(x -> x.av_fixation_time, tr_sim))
push!(mean_lost_time, map(x -> x.av_lost_time, tr_sim))
push!(fixation, map(x -> x.fixation, tr_sim))
push!(time, map(x -> x.T0, tr_sim))

# Create DataFrame for results
df = DataFrame(
    Position = vcat(sites...),
    Nucleotide = vcat(state...),
    Reference = vcat(reference...),
    Time = vcat(time...),
    Frequency = vcat(frequency...),
    Selection_coefficient = vcat(selection_coefficient...),
    Av_fitness_mutant = vcat(av_fitness_mut...),
    Av_fitness_wildtype = vcat(av_fitness_wt...),
    Simulations_fixed = vcat(porcent_simulation_fix...),
    Simulations_lost = vcat(porcent_simulation_lost...),
    Mean_fixation_time = vcat(mean_fixation_time...),
    Mean_lost_time = vcat(mean_lost_time...),
    Fixation = vcat(fixation...),
    Gamma = vcat(gammas...),
    Sigma = vcat(sigmas...)
)

# Write results to CSV
output_filename = "data/$arg2/$arg3/results/simulation_results_T0_$(T0)_$(arg2)_$(arg3).csv"
CSV.write(output_filename, df)
println("Simulation Results Written to: $output_filename")

