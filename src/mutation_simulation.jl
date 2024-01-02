include("mpl.jl")
include("gaussian_smoothing.jl")
include("Population_dynamics.jl")




"""
mutable struct mutation

Mutable structure to store information for each observed mutation.

# Fields:
- `i::Int64`: Position of the mutation.
- `val::Int64`: State of the mutation.
- `reference::Int64`: Reference state, nucleotide/aa of the wildtype sequence.
- `T0::Int64`: Date at t=t_0.
- `freq::Array{Float64, 1}`: Array containing the frequency trajectory of the mutation.
- `fixation::Symbol`: Fixation status, can be :fixed, :lost, or :poly.
- `sc::Float64`: Selection coefficient computed via MPL in the interval.
- `sigma::Int64`: Gaussian smoothing width.
- `gamma::Float64`: Regularization parameter.
- `av_fitness_mut::Float64`: Average fitness of sequences with the mutation.
- `av_fitness_wt::Float64`: Average fitness of sequences with the wildtype mutations.
- `porcent_simulation_fix::Float64`: Percentage of simulations that reach a fixed state.
- `porcent_simulation_lost::Float64`: Percentage of simulations that reach a lost state.
- `av_fixation_time::Float64`: Average fixation time in simulations that reach a fixed state.
- `av_lost_time::Float64`: Average lost time in simulations that reach a lost state.
"""


mutable struct mutation
    i::Int64  
    val::Int64 
    reference::Int64
    T0::Int64 
    freq::Array{Float64,1} 
    fixation::Symbol 
    sc::Float64 
    sigma::Int64 
    gamma::Float64 
    av_fitness_mut::Float64 
    av_fitness_wt::Float64 
    porcent_simulation_fix::Float64 
    porcent_simulation_lost::Float64 
    av_fixation_time::Float64 
    av_lost_time::Float64
end




"""
function init_traject(svec::Vector{Matrix{Float64}},nvec::Vector{Vector{Int64}},T0::Int64,sigma::Int64,gamma::Float64;q=5)

Initialize mutation trajectories based on observed sequences.

Parameters:
- svec: Vector{Matrix{Float64}} - observed variants
- nvec: Vector{Vector{Int64}} - corresponding frequencies
- T0::Int64 - last observation time
- sigma::Int64 - Gaussian smoothing width
- gamma::Float64 - regularization parameter
- q::Int64 - alphabet size (default: 5)

Returns:
- Vector{mutation} - mutation trajectories

The function initializes a vector of mutation structs for each position at `T0` from variant sequences in `sVec` and their frequencies in `nVec`. The `sigma` parameter is the Gaussian smoothing width, and `gamma` is the regularization parameter. The default alphabet size is 5.

"""

function init_traject(svec::Vector{Matrix{Float64}},nvec::Vector{Vector{Int64}},T0::Int64,sigma::Int64,gamma::Float64;q=5)
	
    # Genome length
    L=size(svec[1],2) #genome length
    
    #Create an array of times for the bins considered (not removed)
    min_seq_per_bin=1
    Tf=length(svec)
    bins_time=Int.(1:1:Tf); 

    # Compute full frequency trajectory for each mutation
	full_path_freq=zeros(q*L,Tf)  
	if sigma==0   
        full_path_freq .=compute_path_f_i(nvec,svec,q,L)
    else
        # Gaussian smoothing full path
        # a) Count the frequency of each sequence along the full interval [1:Tf] and create a dict{seq,freq}
	    dict_=create_dict_sqs(svec,nvec)
        # b) Get the same dictionary but with Gaussian smoothed frequencies
	    dict_sm=gaussian_smothing(dict_,bins_time,sigma);
        # c) From sequences frequency stored in dictionary dict_sm, compute single site frequencies smoothed.
    	full_path_freq .=compute_path_f_i(dict_sm,q);
	end

    # Compute consensus sequence at time T=1 and use it as a reference seq.
	fr_t0=full_path_freq[:,1]#
	f_mat=zeros(L,q)
	for i in 1:L
        for j in 1:q
      	 f_mat[i,j]= fr_t0[(i-1)*q+j] 
        end
	end    
	ref_s=map(x->findmax(f_mat[x,:])[2],1:L)

    
    # Compute the frequency trajectories from the initial time to T0
    observed_path_freq=zeros(q*L,T0)
    if sigma==0
        observed_path_freq .=compute_path_f_i(nvec[1:T0],svec[1:T0],q,L)
    else
	# Gaussian smoothing available data
    # a) Count the frequency of each sequence along the learn time windows [1:T0] and create a dict{seq,freq}
        dict_T0=create_dict_sqs(sVec[1:T0],nVec[1:T0]);
    # b) Get the same dictionary but with Gaussian smoothed frequencies
        dict_T0_sm=gaussian_smothing(dict_T0,bins_time[1:T0],sigma);
    # c) From sequences frequency stored in dictionary `dict_T0_sm`, compute single site frequencies smoothed.
        observed_path_freq .=compute_path_f_i(dict_T0_sm,q);
   	end
  
    # Compute selection coefficient via MPL
    h=zeros(q*L)
    mu=0.0 # Mutation rate

    if sigma==0
        h=processStandard(nvec[1:T0],svec[1:T0],Float64.(bins_time[1:T0]),mu,L,q,gamma);
        h=normalize_sc(h,Float64.(ref_s),q,L);
    else
        h=processStandard(dict_T0_sm,Float64.(bins_time[1:T0]),mu,L,q,gamma);     
        h=normalize_sc(h,Float64.(ref_s),q,L); 
    end   

    # Create mutations vector
    out=Array{mutation,1}(undef, 0)
    
    # Fixation and lost thresholds
    lost_thr = 0.05
    fix_threshold=0.95
         
    for pos in 1:L
        for state in 1:q
            mut_vs_time=select_traj(observed_path_freq,pos,state,q) 
            sc=h[(pos-1)*q+state]
            fixation=:poly
            
            for time in T0:1:Tf-1
                if full_path_freq[(pos-1)*q+state,time] <= lost_thr && full_path_freq[(pos-1)*q+state,time+1] <= lost_thr 
                    fixation=:lost
                    break
                elseif full_path_freq[(pos-1)*q+state,time]>=fix_threshold && full_path_freq[(pos-1)*q+state,time+1]>=fix_threshold  
                    fixation=:fixed
                    break
                end
            end
                   
            
            # Average fitness of sequences with mutations and wildtype
            seqs_mut,seqs_wt=select_seq_with_mut(svec[T0],pos,state)
            r=nvec[T0].*fitness(h,svec[T0],q)
            p=r/sum(r)
            p_mut=sum(p[seqs_mut])
            p_wt=sum(p[seqs_wt])
            
            push!(out, mutation(pos,state,ref_s[pos],T0,mut_vs_time,fixation,sc,sigma,gamma,p_mut,p_wt,0.0,0.0,0.0,0.0))
        end
    end
        
  return out  
        
end    



"""
function find_active_positions(trajs::Vector{mutation},q::Int64)


Find positions of active mutations (0.05 < f0 < 0.95) at time T0.

# Parameters:
- `trajs::Vector{mutation}`: Vector of mutation trajectories.
- `q::Int64`: Alphabet size.

# Returns:
- `Vector{Int64}`: Vector of positions for active mutations at T0.

This function identifies and returns the positions of mutations that are active (with a frequency between 0.05 and 0.95) at the observation time T0.

"""


function find_active_positions(trajs::Vector{mutation},q::Int64)
   
    a1=findall(x->(0.05<x.freq[end]<0.95)&&x.val!=x.reference,trajs);
    return map(x->(x.i-1)*q+x.val,trajs[a1])
end    


"""
function get_submatrix(CC::Matrix{Float64},mutations::Vector{Int64})

Extract a submatrix from the input matrix based on specified mutation positions.

# Parameters:
- `CC::Matrix{Float64}`: Input matrix.
- `mutations::Vector{Int64}`: Vector of mutation positions.

# Returns:
- `Matrix{Float64}`: Submatrix extracted from `CC` corresponding to the specified mutation positions.

This function takes an input matrix `CC` and a vector of mutation positions `mutations`. It returns a submatrix extracted from `CC` by selecting rows and columns based on the provided mutation positions.

"""

function get_submatrix(CC::Matrix{Float64},mutations::Vector{Int64})
        L=length(mutations)
        subC=zeros(L,L)
       
        for i in 1:L
            irow=mutations[i]
            subC[i,:] .= CC[irow,mutations]
        end

        return subC          
    end        


"""
function GaussianVector(sqrtC::Array{Float64,2}, mu::Array{Float64,1})

Generate a realization of a Gaussian vector with a specified correlation matrix `sqrtC` and mean vector `mu`.

# Parameters:
- `sqrtC::Array{Float64, 2}`: Square root of the correlation matrix.
- `mu::Array{Float64, 1}`: Mean vector.

# Returns:
- `Array{Float64, 1}`: Realization of the Gaussian vector.

This function generates a random realization of a Gaussian vector using the specified square root of the correlation matrix `sqrtC` and mean vector `mu`. The generated vector is obtained by multiplying a standard Gaussian vector by `sqrtC` and adding the mean vector `mu`.
"""


function GaussianVector(sqrtC::Array{Float64,2}, mu::Array{Float64,1})
    qL = size(sqrtC)[1]
    Y = randn(Float64, qL)
    X = sqrtC*Y + mu # C should be a positive definite matrix
    return X
end


"""
function simulate_and_update_scores(trajs::Vector{mutation},L::Int64,sVec::Vector{Matrix{Float64}},nVec::Vector{Vector{Int64}};q=5,noise=false)

Simulate forward trajectories and update mutation scores.

# Parameters:
- `trajs::Vector{mutation}`: Vector of mutation structs representing trajectories.
- `sVec::Vector{Matrix{Float64}}`: Vector of matrices representing sequences alignments at each time.
- `nVec::Vector{Vector{Int64}}`: Vector of vectors representing corresponding variant frequencies at each time.
- `q::Int64`: Alphabet size (default: 5).
- `noise::Bool`: Consider selection coefficients noise (default: false).

# Returns:
- `Vector{mutation}`: Updated vector of mutation structs with simulation scores.
"""


function simulate_and_update_scores(trajs::Vector{mutation},sVec::Vector{Matrix{Float64}},nVec::Vector{Vector{Int64}};sigma=1,gamma=20.0,q=5,noise=false)
    
    # Print parameters definition
    println("Simulation Parameters:")
    println("sigma = $sigma")
    println("gamma = $gamma")
    println("q = $q")
    println("noise = $noise")


    # Genome length
    L=size(sVec[1],2) 

    # Fixation and lost thresholds
    fix_threshold=0.95
    lost_threshold=0.05

    # Number of forward sampling trajectories and size of new sampled populations
    M=1000   
    N=1000 

    # Observation time
    T0=trajs[1].T0

    # Selection coefficients at T0
    h=map(x->x.sc,trajs);
    # Mutation rate
    mu=0.0
    # Time step and frequency vector for each time
    time_step=1
    f1 = zeros(Float64,L*q)
    

    # Store for each mutation the time where a mutation gets fixed or lost
    fixMat=zeros(q*L,M) 
    lostMat=zeros(q*L,M) 



    # If noise=true, compute the error bar for selection coefficients of observed mutations
    if noise
     
        # Compute the integrated covariance matrix "Cint" at t=T0 
        Tf=length(sVec);
        bins_time=Int.(1:1:Tf);   
        dict_T0=create_dict_sqs(sVec[1:T0],nVec[1:T0]);
        dict_T0_sm=gaussian_smothing(dict_T0,bins_time[1:T0],sigma);
        Cint=Cintegrate(dict_T0_sm,Float64.(bins_time[1:T0]),L,q,gamma)
        iCCint=inv(Cint) 

        # Mutations found active at t=T0
        mut_pos=find_active_positions(trajs,q)

        # Extract submatrix in "Cint" for active mutations
        isC=get_submatrix(iCCint,mut_pos)
        isC=(1/T0)*((isC+isC')/2)    
        sq_isC=sqrt(isC);# # Square root of the covariance matrix for multivariate Gaussian sampling
        L_sC=size(sq_isC,1)

    end    

    
    h_0=deepcopy(h)

    # Simulations
    println("\nSimulating...")
    for i in 1:M      
        println("\n Trajectory $(i)...")
        # If noise=true, add a Gaussian perturbation to selection coefficients for mutations found active at t=T0
       if noise 
	       h_0[mut_pos]=h[mut_pos] .+ GaussianVector(sq_isC,zeros(L_sC));
       end    
        # Initial population at time T0
	   (i_pop,i_sVec,i_nVec)=initial_population(sVec[T0],nVec[T0],q,h_0);##initial population at time T0
        
        t=1

        # Evolve population with Wright-Fisher dynamics
        while  length(i_nVec[1][i_nVec[1].>0])!==1 # Condition of collapsing population into one sequence
            
            (i_pop,i_sVec,i_nVec)= evolve_pop_WF(time_step,N,q,L,mu,h_0,i_pop,i_sVec,i_nVec;record=1)   
            
            # Compute mutations frequency in the new population
            freq_=compute_f1(i_sVec,i_nVec,q;pc=0.0)
            i_sVec=[i_sVec]
            i_nVec=[i_nVec]

             # Check possible fixation positions
            fix_pos=findall(!iszero, freq_.>=fix_threshold)
            if !isempty(fix_pos)
                for k in fix_pos
                    if fixMat[k,i]==0.0
                        fixMat[k,i]=t
                    end 
                end    
            end    

            # Check possible lost positions
            lost_pos=findall(!iszero, freq_.<=lost_threshold)
            if !isempty(lost_pos)
                for k in lost_pos
                    if lostMat[k,i]==0.0
                            lostMat[k,i]=t
                    end 
                end    
            end    
       
            
            t+=1
        end
    end    
    
    # Update mutations with simulation scores
    traj_=Vector{mutation}(undef,0)

    for (id, ft) in enumerate(trajs)
        index=(ft.i-1)*q+ft.val
        ft_ = deepcopy(ft)
        ft_.porcent_simulation_fix=sum(fixMat[index,:].>0)/M
        ft_.porcent_simulation_lost=sum(lostMat[index,:].>0)/M
        ft_.av_fixation_time=mean(fixMat[index,fixMat[index,:].>0])
        ft_.av_lost_time=mean(lostMat[index,lostMat[index,:].>0])
        push!(traj_, ft_)
    end  
       
    return traj_
end  

