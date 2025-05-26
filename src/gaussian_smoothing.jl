using Distributions
using LinearAlgebra



"""
ismember(a::Array{Float64,1}, B::Matrix{Float64})

Returns the row position(s) of the vector `a` in the matrix `B` or an empty array if `a` is not a row of `B`.

Parameters:
- `a` (Array{Float64,1}): The vector to search for in the rows of matrix B.
- `B` (Matrix{Float64}): The matrix in which to search for the vector `a`.

Returns:
- An array of integers representing the row position(s) of the vector `a` in the matrix `B`.
  If `a` is not found in any row of `B`, an empty array is returned.
"""
function ismember(a::Array{Float64,1},B::Matrix{Float64})
    return findall([a== B[i,:] for i=1:size(B,1) ] )
end   


"""
create_dict_sqs(sVec::Vector{Matrix{Float64}}, nVec::Vector{Vector{Int64}})

Creates a dictionary where keys are all unique sequences in `sVec`, and values are their counts for each time stored in `nVec`.

Parameters:
- `sVec` (Vector{Matrix{Float64}}): Vector of matrices containing sequences at each time point.
- `nVec` (Vector{Vector{Int64}}): Vector of vectors containing sequence counts at each time point.

Returns:
- A dictionary (`Dict{Vector{Float64}, Vector{Float64}}`) where keys are unique sequences, and values are arrays of counts for each time point.
"""

function create_dict_sqs(sVec::Vector{Matrix{Float64}}, nVec::Vector{Vector{Int64}})
    seqs = unique(vcat(sVec...), dims=1)
    sqs_D = Dict{Vector{Float64}, Vector{Float64}}()
    bins = length(sVec)
    
    for i in 1:size(seqs, 1)
        seq_temp = seqs[i, :]   
        freq = zeros(bins)
        
        for j in 1:bins
            imb = ismember(seq_temp, sVec[j])
            
            if isempty(imb)
                freq[j] = 0.0
            else
                imb = imb[1]
                freq[j] = nVec[j][imb]
            end
        end
        
        push!(sqs_D, seq_temp => freq) 
    end
    
    return sqs_D
end



"""
gaussian_weight(time_window::Vector{Int64}, t0::Int64, sigma::Int64)

Compute the Gaussian weight normalized for the interval defined by `time_window`.

Parameters:
- `time_window` (Vector{Int64}): Vector representing the time window.
- `t0` (Int64): Center of the Gaussian distribution.
- `sigma` (Int64): Standard deviation of the Gaussian distribution.

Returns:
- An array of Gaussian weights, normalized such that their sum is equal to 1.
"""

function gaussian_weight(time_window::Vector{Int64},t0::Int64,sigma::Int64)
    output=map(t-> exp(-(t-t0)^2/(2*sigma^2)),time_window)
    return output/sum(output)
end    


"""
gaussian_smothing(init_seqs::Dict{Vector{Float64},Vector{Float64}}, time_window::Vector{Int64}, sigma::Int64)

Return a dictionary of sequences with counts smoothed via Gaussian smoothing.

Parameters:
- `init_seqs` (Dict{Vector{Float64},Vector{Float64}}): Initial dictionary of sequences with counts.
- `time_window` (Vector{Int64}): Vector representing the time window for Gaussian smoothing.
- `sigma` (Int64): Standard deviation of the Gaussian distribution for smoothing.

Returns:
- A new dictionary with smoothed counts for each sequence based on the specified time window and sigma.
"""

function gaussian_smothing(init_seqs::Dict{Vector{Float64},Vector{Float64}},time_window::Vector{Int64},sigma::Int64)
    
    output_seq=deepcopy(init_seqs)
    for t0 in time_window
        gw=gaussian_weight(time_window,t0,sigma)
        for seqs in collect(keys(init_seqs))
          output_seq[seqs][t0]=gw'*init_seqs[seqs]
        end
    end
    
    return output_seq
end    




"""
compute_f1(seqs::Dict{Vector{Float64}, Vector{Float64}}, t::Int64, q::Int64)

Compute single site frequencies (f1) at time t from a sequences dictionary.

Parameters:
- `seqs` (Dict{Vector{Float64}, Vector{Float64}}): Dictionary of sequences with counts.
- `t` (Int64): Time point for which f1 is computed.
- `q` (Int64): Number of possible states for each site.

Returns:
- An array of single site frequencies (f1) at time t.
"""

function compute_f1(seqs::Dict{Vector{Float64},Vector{Float64}},t::Int64,q::Int64)
    Y=collect(keys(seqs))
    Y=map(x->Int64.(x),Y);
    M=size(Y,1)
    L=length(Y[1])
    N=sum(map(x->x[t],collect(values(seqs))))
    counts=map(x->x[t],collect(values(seqs)));
    f1 = zeros(Float64,L*q);
    
     for m in 1:M
        for j in 1:L
            f1[(j-1)*q+Y[m][j]] += counts[m]      
        end
     end
    
    f1 = f1./N

    return f1
    
end  


"""
compute_freqs(seqs::Dict{Vector{Float64}, Vector{Float64}}, t::Int64, q::Int64)

Compute single site frequencies (f1) and pairwise frequencies (f2) at time t from a sequences dictionary.

Parameters:
- `seqs` (Dict{Vector{Float64}, Vector{Float64}}): Dictionary of sequences with counts.
- `t` (Int64): Time point for which frequencies are computed.
- `q` (Int64): Number of possible states for each site.

Returns:
- A tuple of arrays containing f1 and f2 at time t.
"""



function compute_freqs(seqs::Dict{Vector{Float64},Vector{Float64}},t::Int64,q::Int64)
    Y=collect(keys(seqs))
    Y=map(x->Int64.(x),Y);
    M=size(Y,1)
    L=length(Y[1])
    N=sum(map(x->x[t],collect(values(seqs))))
    counts=map(x->x[t],collect(values(seqs)));
    f1 = zeros(Float64,L*q);
    f2 = zeros(Float64, L*q, L*q)

     for m in 1:M
        for j in 1:L
            f1[(j-1)*q+Y[m][j]] += counts[m]
            f2[(j-1)*q+Y[m][j], (j-1)*q+Y[m][j]] += counts[m]
            for i in (j+1):L
                f2[(i-1)*q+Y[m][i], (j-1)*q+Y[m][j]] += counts[m]
                f2[(j-1)*q+Y[m][j], (i-1)*q+Y[m][i]] += counts[m]
            end
        end
     end
    
    f1 = f1./N
    f2 = f2./N

    return (f1,f2)
    
end  


"""
compute_path_f_i(seqs::Dict{Vector{Float64}, Vector{Float64}}, q::Int64)

Compute trajectories of single site frequencies (f1) over time from a sequences dictionary.

Parameters:
- `seqs` (Dict{Vector{Float64}, Vector{Float64}}): Dictionary of sequences with counts.
- `q` (Int64): Number of possible states for each site.

Returns:
- An array representing the trajectories of f1 over time.
"""

function compute_path_f_i(seqs::Dict{Vector{Float64},Vector{Float64}},q::Int64)
    Y=collect(keys(seqs))
    L=length(Y[1])
    bins=length(collect(values(seqs))[1])
    freq_vs_time=zeros(L*q,bins)
    
    for t0 in 1:bins
        f1=compute_f1(seqs,t0,q);
        freq_vs_time[1:end,t0] .=f1
    end
    
    return freq_vs_time
end   


# Smooth methods II (along days)

function normalizeCounts(times::Vector{Float64},counts::Vector{Float64})
    d=times[1]
    counts_=[]
    for d in unique(times)
        counts_by_day=[]
        for (i,t) in enumerate(times)
            if t==d
                push!(counts_by_day,counts[i])
            end
        end    
        push!(counts_,counts_by_day)
    end     
    norm_counts=map(x->x/sum(x),counts_)
return round.(vcat(norm_counts...),digits=6)
end    

function smoothCounts(counts::Vector{Float64}, times::Vector{Float64}, t::Float64, tsmooth::Float64;useDebug=false)
    counts=normalizeCounts(times,counts)
    effectiveCounts=zeros(length(counts))
    
    tcutoff = 3 * tsmooth
    norm = 1 / (tsmooth * sqrt(2 * π))
    
    for k in 1:length(counts)
        if abs(times[k] - t) > tcutoff
            effectiveCounts[k] = 0
        else
            effectiveCounts[k] = counts[k] * exp(-((times[k] - t)/tsmooth)^2 / 2) * norm
        end
    end

    # Debug
    if useDebug
        counted = sum(effectiveCounts .> 0)
        println("time $t\t$counted contributing sequences")
    end
    
    return effectiveCounts
end






















