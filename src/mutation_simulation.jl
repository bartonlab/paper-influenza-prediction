include("mpl.jl")
include("misc.jl")
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
- `emp_frequency::Float64`: mutation frequency at observation time.
- `time_active::Float64`: time mutation has been active.
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
    emp_frequency::Float64
    time_active::Float64
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

function init_traject(svec::Vector{Matrix{Float64}},nvec::Vector{Vector{Int64}},T0::Int64,gamma::Float64,sigma::Int64;mu=0.0,q=5,correlation=true)
	
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
        dict_T0=create_dict_sqs(svec[1:T0],nvec[1:T0]);
    # b) Get the same dictionary but with Gaussian smoothed frequencies
        dict_T0_sm=gaussian_smothing(dict_T0,bins_time[1:T0],sigma);
    # c) From sequences frequency stored in dictionary `dict_T0_sm`, compute single site frequencies smoothed.
        observed_path_freq .=compute_path_f_i(dict_T0_sm,q);
   	end
  
    # Compute selection coefficient via MPL
    h=zeros(q*L)

    if sigma==0
        h=processStandard(nvec[1:T0],svec[1:T0],Float64.(bins_time[1:T0]),mu,L,q,gamma);
    else
        if correlation
            h=processStandard(dict_T0_sm,Float64.(bins_time[1:T0]),mu,L,q,gamma);     
         else
            h=processStandardNoCorr(dict_T0_sm,Float64.(bins_time[1:T0]),mu,L,q,gamma);     
         end   
    end   



    h=normalize_sc(h,Float64.(ref_s),q,L); 

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
        
  return out,h  
        
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


function init_traject(svec::Vector{Matrix{Float64}},nvec::Vector{Vector{Int64}},T0::Int64,sigma::Int64,gamma::Float64; mu=0.0,q=5,correlation=true)
    
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

    seqs_at_T0 = Matrix{Float64}(undef, 0, 0)
    scounts_at_T0 = Float64[]
    
    # Compute the frequency trajectories from the initial time to T0
    observed_path_freq=zeros(q*L,T0)
    if sigma==0
        observed_path_freq .=compute_path_f_i(nvec[1:T0],svec[1:T0],q,L)
    else
    # Gaussian smoothing available data
    # a) Count the frequency of each sequence along the learn time windows [1:T0] and create a dict{seq,freq}
        dict_T0=create_dict_sqs(svec[1:T0],nvec[1:T0]);
    # b) Get the same dictionary but with Gaussian smoothed freqquencies
        dict_T0_sm=gaussian_smothing(dict_T0,bins_time[1:T0],sigma);
    # c) From sequences frequency stored in dictionary `dict_T0_sm`, compute single site frequencies smoothed.
        observed_path_freq .=compute_path_f_i(dict_T0_sm,q);
    
    # Compute scounts_at_T0 and seqs_at_T0 dynamically
        scounts_at_T0=last.(collect(values(dict_T0_sm)))
        scounts_at_T0 ./= (sum(scounts_at_T0))
        Y=collect(keys(dict_T0_sm))
        seqs_at_T0=zeros(size(Y,1),length(Y[1]));
        for i in 1:size(Y,1)
            seqs_at_T0[i,:] .= Y[i]
        end
    end


    
    # Compute selection coefficient via MPL
    h=zeros(q*L)

   
  if sigma==0
        h=processStandard(nvec[1:T0],svec[1:T0],Float64.(bins_time[1:T0]),mu,L,q,gamma);
    else
        if correlation
            h=processStandard(dict_T0_sm,Float64.(bins_time[1:T0]),mu,L,q,gamma);     
         else
            h=processStandardNoCorr(dict_T0_sm,Float64.(bins_time[1:T0]),mu,L,q,gamma);     
         end   
    end   

    h=normalize_sc(h,Float64.(ref_s),q,L); 

    # Create mutations vector
    out=Array{mutation,1}(undef, 0)
    
    # Fixation and lost thresholds
    lost_thr = 0.05
    fix_threshold=0.95
    
    # compute empirical frequency at T0
    femp=observed_path_freq[:,end]
    
    for pos in 1:L
        for state in 1:q
            mut_vs_time=select_traj(observed_path_freq,pos,state,q)
            times_measure=Float64.(1:1:T0)
            t_active=obs_time(mut_vs_time,times_measure)
            sc=h[(pos-1)*q+state]
            fixation=:poly
            f0=femp[(pos-1)*q+state]# empirical frecuency

            for time in T0:1:Tf-1
                if full_path_freq[(pos-1)*q+state,time] <= lost_thr && full_path_freq[(pos-1)*q+state,time+1] <= lost_thr 
                    fixation=:lost
                    break
                elseif full_path_freq[(pos-1)*q+state,time]>=fix_threshold && full_path_freq[(pos-1)*q+state,time+1]>=fix_threshold  
                    fixation=:fixed
                    break
                end
            end
                   
            p_mut=0.0
            p_wt=0.0
            # Average fitness of sequences with mutations and wildtype
            if sigma==0
                seqs_mut,seqs_wt=select_seq_with_mut(svec[T0],pos,state)
                r=nvec[T0].*fitness(h,svec[T0],q)
                p=r/sum(r)
                p_mut=sum(p[seqs_mut])
                p_wt=sum(p[seqs_wt])

            else
                seqs_mut,seqs_wt=select_seq_with_mut(seqs_at_T0,pos,state)
                r=scounts_at_T0.*fitness(h,seqs_at_T0,q)    
                p=r/sum(r)
                p_mut=sum(p[seqs_mut])
                p_wt=sum(p[seqs_wt])
            end    
            
            

            push!(out,mutation(pos,state,ref_s[pos],T0,mut_vs_time,f0,t_active,fixation,sc,sigma,gamma,p_mut,p_wt,0.0,0.0,0.0,0.0))

            
        end
    end
        
  return out  
        
end    




"""
function init_traject(data::Matrix{Float64},T0::Int64,gamma::Float64;tstep=10.0,tsmooth=15.0,q=21)

Initialize mutation trajectories based on observed sequences using smoothing method II (smoothing sequences over time).

Parameters:
- data::Matrix{Float64} - observed sequences configurations and its counts for each observation time
- T0::Int64 - last observation time
- tsmooth::Float64 - Width of Gaussian kernel, Default: 15.0.
 -tstep:: Float64, Default: 10.0. Time step when smoothing sequences over time (see below).
- gamma::Float64 - regularization parameter
- q::Int64 - alphabet size (default: 5)

Returns:
- Vector{mutation} - mutation trajectories

The function initializes a vector of 'mutation' structs for each position at `T0` from 'data'. The `sigma` parameter is the Gaussian smoothing width, and `gamma` is the regularization parameter. The default alphabet size is 5.

"""


function init_traject(data::Matrix{Float64},T0::Int64,gamma::Float64;tstep=10.0,tsmooth=15.0,mu=0.0,q=21,correlation=true)
   
    # full data
    times=data[1:end,1] #observation times
    counts=data[1:end,2] #sequences count
    seqs=data[1:end,3:end];# sequences configuration
    #unique observation times.
    u_times=unique(times);
    # Genome length
    L=size(seqs,2) 
    # Smooth and compute full frequency trajectory for each mutation
    effectiveCounts_full=[]
    
    for t_ in 1.0:tstep:last(u_times)
        effectiveCounts_=smoothCounts(counts,times,t_,tsmooth);
        push!(effectiveCounts_full,effectiveCounts_)    
    end    

    full_path_freq=compute_path_f_i(seqs,effectiveCounts_full,q);
    Tf=size(full_path_freq,2)     


    # Compute the frequency trajectories from the initial time to u_times[T0]
    #Smooth counts 
    t_c=u_times[T0];# current time given T0

    (position_in_time_T1,position_in_time_T0)=find_previous_time_bin(times,T0)

    #position_in_time_T0=last(findall(x->x==t_c,times))#last position in times where t<=t_c 


    effectiveCounts_list=[]
    times_measure=Float64[]
    for t_ in 1.0:tstep:t_c
        push!(times_measure,t_)
        effectiveCounts_=smoothCounts(counts[1:position_in_time_T0],times[1:position_in_time_T0],t_,tsmooth);
        push!(effectiveCounts_list,effectiveCounts_)    
    end    

    seqs_observed=seqs[1:position_in_time_T0,:]# sequences considered
    observed_path_freq=compute_path_f_i(seqs_observed,effectiveCounts_list,q);


    # compute empirical frequency at t_c
    count_current_bin=counts[position_in_time_T1:position_in_time_T0]
    seqs_current_bin=seqs[position_in_time_T1:position_in_time_T0,:];
    femp=compute_f1(seqs_current_bin,count_current_bin);


    # Compute consensus sequence at time T=1 and use it as a reference seq.
    fr_t0=full_path_freq[:,1]#
    f_mat=zeros(L,q)
    for i in 1:L
        for j in 1:q
         f_mat[i,j]= fr_t0[(i-1)*q+j] 
        end
    end    
    ref_s=map(x->findmax(f_mat[x,:])[2],1:L)
    

 # Compute selection coefficient via MPL
    h=zeros(q*L)
    if correlation
         h= processStandard(seqs_observed,effectiveCounts_list,times_measure,mu,L,q,gamma);
        else
        h=processStandardNoCorr(seqs_observed,effectiveCounts_list,times_measure,mu,L,q,gamma)
    end





    # normalize selection coefficient for reference seq
    h=normalize_sc(h,Float64.(ref_s),q,L); 

    # Create mutations vector
    out=Array{mutation,1}(undef, 0)
    
    # Fixation and lost thresholds
    lost_thr = 0.05
    fix_threshold=0.95
    Ti=length(times_measure)#number of bins in the training/observed dataset

    for pos in 1:L
        for state in 1:q
            mut_vs_time=select_traj(observed_path_freq,pos,state,q) 
            t_active=obs_time(mut_vs_time,times_measure)
            sc=h[(pos-1)*q+state]
            fixation=:poly

            f0=femp[(pos-1)*q+state]# empirical frecuency
            
            for time in Ti:1:Tf-1
                if full_path_freq[(pos-1)*q+state,time] <= lost_thr && full_path_freq[(pos-1)*q+state,time+1] <= lost_thr 
                    fixation=:lost
                    break
                elseif full_path_freq[(pos-1)*q+state,time]>=fix_threshold && full_path_freq[(pos-1)*q+state,time+1]>=fix_threshold  
                    fixation=:fixed
                    break
                end
            end
                   
            
            # Average fitness of sequences with mutations and wildtype
            ncounts=last(effectiveCounts_list);
            counts_=ncounts[ncounts.>0.0]
            sequences=seqs_observed[ncounts.>0.0,:];
            
            seqs_mut,seqs_wt=select_seq_with_mut(sequences,pos,state)
            r=counts_.*fitness(h,sequences,q)
            p=r/sum(r)
            p_mut=sum(p[seqs_mut])
            p_wt=sum(p[seqs_wt])
            
            push!(out, mutation(pos,state,ref_s[pos],t_c,mut_vs_time,f0,t_active,fixation,sc,tsmooth,gamma,p_mut,p_wt,0.0,0.0,0.0,0.0))
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
function find_observed_positions(trajs::Vector{mutation},q::Int64)

Find mutations observed  (0.05 < f0 < 0.95) for t<=T0.

# Parameters:
- `trajs::Vector{mutation}`: Vector of mutation trajectories.
- `q::Int64`: Alphabet size.

# Returns:
- `Vector{Int64}`: Vector of positions for observed mutations t<=T0.

This function identifies and returns the positions of mutations that reached in some time before T0  frequency between 0.05 and 0.95.

"""


function find_observed_positions(trajs::Vector{mutation},q::Int64)
    mut_list=Int64[]
    for t in 1:length(trajs)
        x=trajs[t]
        for i in 1:length(x.freq)
            if 0.05<x.freq[i]<0.95&&x.val!=x.reference
                push!(mut_list,(x.i-1)*q+x.val)
                break
            end
        end
           
    end    

    return mut_list
end  


"""
function unique_sequences(seqs::Matrix{Float64},counts::Vector{Float64})


# Parameters:
- `seqs::Matrix{Float64}`: sequence configurations.
- `counts::Vector{Float64}`: sequence counts.

# Returns:
- (Matrix{Float64},Vector{Float64}): Unique sequences in 'seqs'  and their corresponding effective counts normalized

"""

function unique_sequences(seqs::Matrix{Float64},counts::Vector{Float64})
    
    sVec=[]
    for i in 1:size(seqs,1)
        push!(sVec,seqs[i,:])
    end    
    
    norm_counts=counts./sum(counts)

    
    newSeqs=[]
    newCounts=Float64[]
    for j in 1:length(sVec)
                unique = true
                for k in 1:length(newSeqs)
                    if sVec[j] == newSeqs[k]
                        unique = false
                        newCounts[k]+=norm_counts[j]
                        break
                    end
                end
                
                if unique
                    push!(newSeqs,sVec[j])
                    push!(newCounts,norm_counts[j])
                end  
    end


    newSeqs=Float64.(hcat(newSeqs...)')

    return newSeqs,newCounts
end   




"""
function obs_time(mut_vs_t::Vector{Float64},times_::Vector{Float64})


# Parameters:
- `mut_vs_t::Vector{Float64}`: mutation frequencies
- `times_`: observed times.

# Returns:
- how long the mutation has been observed since the last time it became fixed or lost

"""

function obs_time(mut_vs_t::Vector{Float64},times_::Vector{Float64})
    if length(mut_vs_t)!=length(times_)
        return println("Error:frequency trajectory and times must have the same length")
    end
    
    # initialize times
    t_fix=0.0 
    t_lost=0.0 
    t_o=last(times_)#current time
    
    # find positions in "mut_vs_t" where the mutations is fixed or lost
    fix_t=findall(x->(x>0.95),mut_vs_t)
    lost_t=findall(x->(0.05>x),mut_vs_t)

    
    if isempty(lost_t) # never found lost
        t_lost=1.0
    else
        t_lost=last(lost_t) # last time this mutation reach lost threshold
    end

    if isempty(fix_t) # never found fix
        t_fix=1.0
    else
        t_fix=last(fix_t) # last time this mutation reach fixation threshold
    end

    # take the most recent time where the mutation is either fixed or lost
    if t_lost>=t_fix
        return t_o-times_[Int64(t_lost)]
    else
        return t_o-times_[Int64(t_fix)]
    end    
    
end    




function find_previous_time_bin(times::Vector{Float64},T0::Int64)
    # select unique times
    unique_times=unique(times);
    # get the real observation time correspondeing to the bin T0
    t_c=unique_times[T0]

    # found last position in times list corresponding to "t_c"
    position_in_time_T0=last(findall(x->x==t_c,times))

    # get the list of times corresponding to the observation time window
    obs_times=times[1:position_in_time_T0]

    # get the position x of the bin where times[x]+30=position_in_time_T0
    diff=(t_c.-obs_times)   
    
    if isempty(sort(diff[(diff.-30).>=0.0]))
        return 1,position_in_time_T0
    else    
        t=first(sort(diff[(diff.-30).>=0.0]))  
        position_in_time_T1=first(findall(x->x==t,diff))   
        # return (T1,T0) where times[T1]-times[T0]=1 months
        return position_in_time_T1,position_in_time_T0 
    end
    
end

# this function find in T0_list the index T1 where unique_times[T1]-unique_times[T0]=360 days
function find_next_year(times::Vector{Float64},T0::Int64,T0_list::Vector{Float64})
    # select unique times
    unique_times=unique(times);
    # get the real observation time correspondeing to the bin T0
    t_c=unique_times[T0]

    diff_=(unique_times[Int64.(T0_list)].-t_c)   
    if isempty(sort(diff_[(diff_.-360).>=0.0]))
        println("Error: There is not data 1y after T0")
        return 0
    else    
        
        t=first(sort(diff_[(diff_.-360).>=0.0]))  
        position_in_time_T1=first(findall(x->x==t,diff_))   
        return Int64(T0_list[position_in_time_T1])
    end
    
end

# this function find in T0_list the index T1 where unique_times[T1]-unique_times[T0]=time_dist days
function find_next(times::Vector{Float64},time_dist::Int64,T0::Int64,T0_list::Vector{Float64})
    # select unique times
    unique_times=unique(times);
    # get the real observation time correspondeing to the bin T0
    t_c=unique_times[T0]

    diff_=(unique_times[Int64.(T0_list)].-t_c)   
    if isempty(sort(diff_[(diff_.-time_dist).>=0.0]))
        println("Error: There is not data 1y after T0")
        return 0
    else    
        
        t=first(sort(diff_[(diff_.-time_dist).>=0.0]))  
        position_in_time_T1=first(findall(x->x==t,diff_))   
        return Int64(T0_list[position_in_time_T1])
    end
    
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





function DiagNoiseMat(trajs::Vector{mutation},σ0::Float64,τ::Float64)
    active_trajs=findall(x->(0.05<x.freq[end]<0.95)&&x.val!=x.reference,trajs);
    times=map(x->x.time_active,trajs[active_trajs])./τ
    return σ0 .* diagm(exp.(-times))
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

function simulate_and_update_scores(trajs::Vector{mutation},sVec::Vector{Matrix{Float64}},nVec::Vector{Vector{Int64}};q=5,noise=false,gamma_noise=10000.0,M=1000)


    sigma=trajs[1].sigma
    gamma=trajs[1].gamma

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

    # size of new sampled populations
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

    Tf=length(sVec);
    bins_time=Int.(1:1:Tf);   
    dict_T0=create_dict_sqs(sVec[1:T0],nVec[1:T0]);
    dict_T0_sm=gaussian_smothing(dict_T0,bins_time[1:T0],sigma);

    # obtain smoothed counts and sequences at T0
    scounts_at_T0=last.(collect(values(dict_T0_sm)))
    scounts_at_T0=scounts_at_T0./(sum(scounts_at_T0))
    Y=collect(keys(dict_T0_sm))
    seqs_at_T0=zeros(size(Y,1),length(Y[1]));
    for i in 1:size(Y,1)
        seqs_at_T0[i,:].= Y[i]
    end    


    # If noise=true, compute the error bar for selection coefficients of observed mutations
    if noise
     
        # Compute the integrated covariance matrix "Cint" at t=T0 
     
        Cint_=Cintegrate(dict_T0_sm,Float64.(bins_time[1:T0]),L,q,gamma_noise)
        
        # Mutations found active at t=T0
        mut_pos=find_active_positions(trajs,q)
        sC=get_submatrix(Cint_,mut_pos)
        isC=inv(sC)
        isC=((isC+isC')/2)

        # Extract submatrix in "Cint" for active mutations
        #isC=(1/T0)*((isC+isC')/2)    

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

         (i_pop,i_sVec,i_nVec)=initial_population(seqs_at_T0,scounts_at_T0,q,h_0);
        
        t=1

        # Evolve population with Wright-Fisher dynamics
        while  length(i_nVec[1][i_nVec[1].>0])!==1 # Condition of collapsing population into one sequence

            (i_pop,i_sVec,i_nVec)= evolve_pop_WF(time_step,N,q,L,h_0,i_pop,i_sVec,i_nVec;record=1)   

            # Compute mutations frequency in the new population
            freq_=compute_f1(i_sVec,i_nVec,q=q)
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





function simulate_and_update_scores_v2(trajs::Vector{mutation},sVec::Vector{Matrix{Float64}},nVec::Vector{Vector{Int64}};q=5,noise=false,gamma_noise=10000.0,M=1000,mu=0.0)


    sigma=trajs[1].sigma
    gamma=trajs[1].gamma

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

    # size of new sampled populations
    N=1000 

    # Observation time
    T0=trajs[1].T0

    # Selection coefficients at T0
    h=map(x->x.sc,trajs);
    # Mutation rate
    
    # Time step and frequency vector for each time
    time_step=1
    f1 = zeros(Float64,L*q)
    
    # Store for each mutation the time where a mutation gets fixed or lost
    fixMat=zeros(q*L,M) 
    lostMat=zeros(q*L,M) 

    Tf=length(sVec);
    bins_time=Int.(1:1:Tf);   

    global seqs_at_T0
    global scounts_at_T0
    global dict_T0_sm
    if sigma!=0
        dict_T0=create_dict_sqs(sVec[1:T0],nVec[1:T0]);
        dict_T0_sm=gaussian_smothing(dict_T0,bins_time[1:T0],sigma);

        # obtain smoothed counts and sequences at T0
        scounts_at_T0=last.(collect(values(dict_T0_sm)))
        scounts_at_T0=scounts_at_T0./(sum(scounts_at_T0))
        Y=collect(keys(dict_T0_sm))
        seqs_at_T0=zeros(size(Y,1),length(Y[1]));
        for i in 1:size(Y,1)
            seqs_at_T0[i,:].= Y[i]
        end  

     else    
        seqs_at_T0 = sVec[T0]
        scounts_at_T0 = nVec[T0]
    end     



    # If noise=true, compute the error bar for selection coefficients of observed mutations
    if noise
     
        # Compute the integrated covariance matrix "Cint" at t=T0 
        if sigma!=0
            Cint_=Cintegrate(dict_T0_sm,Float64.(bins_time[1:T0]),L,q,gamma_noise)
        else
             Cint_=Cintegrate(sVec[1:T0],nVec[1:T0],Float64.(bins_time[1:T0]),L,q,gamma_noise)

        end            
        # Mutations found active at t=T0
        mut_pos=find_active_positions(trajs,q)
        sC=get_submatrix(Cint_,mut_pos)
        isC=inv(sC)
        isC=((isC+isC')/2)

        # Rescale as function of the observation time
       # isC=(1/T0)*((isC+isC')/2)    

        sq_isC=sqrt(isC);# # Square root of the covariance matrix for multivariate Gaussian sampling
        L_sC=size(sq_isC,1)
        
    end    

    
    h_0=deepcopy(h)

    # Simulations
    #println("\nSimulating...")
    for i in 1:M      
        #println("\n Trajectory $(i)...")
        # If noise=true, add a Gaussian perturbation to selection coefficients for mutations found active at t=T0
       if noise 
           h_0[mut_pos]=h[mut_pos] .+ GaussianVector(sq_isC,zeros(L_sC));
       end    
        # Initial population at time T0

         (i_pop,i_sVec,i_nVec)=initial_population(seqs_at_T0,scounts_at_T0,q,h_0);
        
        t=1

        # Evolve population with Wright-Fisher dynamics
        while  length(i_nVec[1][i_nVec[1].>0])!==1 # Condition of collapsing population into one sequence

            #(i_pop,i_sVec,i_nVec)= evolve_pop_WF(time_step,N,q,L,h_0,i_pop,i_sVec,i_nVec;record=1)   
                r = [s.n * s.f for s in i_pop]
                r = abs.(r)
                p = r / sum(r) # probability of selection for each species (sequence)
                
                if sigma==0
                    n = rand(Multinomial(N, p))    # selected number of each species counts    
                    else 
                    n = rand(Multinomial(N, p))./N # selected number of each species counts
                end 

                for i in 1:length(i_pop)
                    i_pop[i].n = n[i] # set new number of each species counts
                end
                    # Update measurements
                i_nVec=[[s.n  for s in i_pop]]    
            t+=1
        end
        # Compute mutations frequency in the new population
            freq_=compute_f1(i_sVec[1],Float64.(i_nVec[1]),q=q)
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





















"""
function simulate_and_update_scores(trajs::Vector{mutation},data::Matrix{Float64};gamma=20.0,q=21,noise=0,gamma_noise=10000,σ0=0.01,τ=100.0,tstep=10,M=1000)

Simulate forward trajectories and update mutation scores.

# Parameters:
- `trajs::Vector{mutation}`: Vector of mutation structs representing trajectories.
- data::Matrix{Float64} - observed sequences configurations and its counts for each observation time
- `q::Int64`: Alphabet size (default: 5).
- `noise::Int64`: Consider selection coefficients noise if noise=1 (default: noise=0).
- `gamma_noise::Int64`: noise tunning parameter.
 -tstep:: Int64, Default: 10. Time step when smoothing sequences over time.
# Returns:
- `Vector{mutation}`: Updated vector of mutation structs with simulation scores.
"""



function simulate_and_update_scores(trajs::Vector{mutation},data::Matrix{Float64};gamma=20.0,q=21,noise=0,gamma_noise=10000,σ0=0.01,τ=100.0,tstep=10,M=1000)
    
    # Print parameters definition
    println("Simulation Parameters:")
    println("gamma = $gamma")
    println("q = $q")
    println("noise = $noise")
     # full data
    times=data[1:end,1] #observation times
    counts=data[1:end,2] #sequences count
    seqs=data[1:end,3:end];# sequence configurations
    #unique observation times.
    u_times=unique(times);
    # Genome length
    L=size(seqs,2) 

    # Fixation and lost thresholds
    fix_threshold=0.95
    lost_threshold=0.05

    # Number of forward sampling trajectories and size of new sampled populations
      
    N=1000 

    # Observation time
    t_c=trajs[1].T0
    
    
    #Smooth counts for observation considered
    position_in_time_T0=last(findall(x->x==t_c,times))#last position in "times" where t=t_c 
    effectiveCounts_list=[]
    times_measure=Float64[]
    tsmooth=Float64(trajs[1].sigma)
    for t_ in 1.0:tstep:t_c
        push!(times_measure,t_)
        effectiveCounts_=smoothCounts(counts[1:position_in_time_T0],times[1:position_in_time_T0],t_,tsmooth);
        push!(effectiveCounts_list,effectiveCounts_)    
    end    
    seqs_observed=seqs[1:position_in_time_T0,:]# sequences considered

    
    # Selection coefficients at T0
    h=map(x->x.sc,trajs);
    # Mutation rate
    mu=0.0
    # Time step and frequency vector for each time
    time_step=1
    # Store for each mutation the time where a mutation gets fixed or lost
    fixMat=zeros(q*L,M) 
    lostMat=zeros(q*L,M) 
    
    # If noise!-0, compute the error bar for selection coefficients of observed mutations
    if noise!=0
        # Compute the integrated covariance matrix "Cint" at t=T0 
        Cint_=Cintegrate(seqs_observed,effectiveCounts_list,times_measure,L,q,gamma_noise)
       # iCCint=inv(Cint_) 
        # Mutations found active at t=T0
        mut_pos=find_active_positions(trajs,q)
        # Extract submatrix in "Cint" for active mutations
        sC=get_submatrix(Cint_,mut_pos)
        isC=inv(sC)
        isC=((isC+isC')/2)
        # isC=get_submatrix(iCCint,mut_pos)
       # isC=((isC+isC')/2)    
        if noise==2
            isC .= isC .+ DiagNoiseMat(trajs,σ0,τ)
        end

        sq_isC=sqrt(isC);# # Square root of the covariance matrix for multivariate Gaussian sampling
        L_sC=size(sq_isC,1)
    end    
    
    h_0=deepcopy(h)
    # Simulations
    println("\nSimulating...")
    for i in 1:M      
        println("\n Trajectory $(i)...")
        # If noise=true, add a Gaussian perturbation to selection coefficients for mutations found active at t=T0
       if noise!=0 
           h_0[mut_pos]=h[mut_pos] .+ GaussianVector(sq_isC,zeros(L_sC));
       end    
        # Initial population at time T0

        ncounts=last(effectiveCounts_list);
        counts_=ncounts[ncounts.>0.0]
        sequences=seqs_observed[ncounts.>0.0,:];

        (u_sequences,u_counts)=unique_sequences(sequences,counts_)
        
       (i_pop,i_sVec,i_nVec)=initial_population(u_sequences,u_counts,q,h_0);
    
        t=1
        # Evolve population with Wright-Fisher dynamics
        while  length(i_nVec[1][i_nVec[1].>0])!==1 # Condition of collapsing population into one sequence
            
            (i_pop,i_sVec,i_nVec)= evolve_pop_WF(time_step,N,q,L,h_0,i_pop,i_sVec,i_nVec;record=1)   
            
            # Compute mutations frequency in the new population
            freq_=compute_f1(i_sVec,i_nVec,q=q)
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


