using Distributions
using LinearAlgebra


"""

function comp_freqs(Y::Array{Float64,2},counts::Array{Int64,1},q::Int64;pc=0.0)

Compute allele frequencies and pairwise allele frequencies for given sequences and counts.

# Arguments
- `Y::Array{Float64,2}`: Matrix representing the observed sequences with each row as a sequence.
- `counts::Array{Int64,1}`: Vector of counts corresponding to each sequence.
- `q::Int64`: Number of states (e.g., nucleotides:q=5 or amino acids:q=21).
- `pc::Float64`: Pseudocount, a small value to avoid division by zero. Defaults to 0.0.

# Returns
- Tuple containing:
  - `f1::Array{Float64,1}`: Array of allele frequencies.
  - `f2::Array{Float64,2}`: Matrix of pairwise allele frequencies.
"""

function comp_freqs(Y::Array{Float64,2},counts::Array{Int64,1},q::Int64;pc=0.0)
    Y=Int64.(Y)
    (M,L) = size(Y)
     N=sum(counts)
     f1 = zeros(Float64,L*q)
     f2 = zeros(Float64, L*q, L*q)

    for m in 1:M
        for j in 1:L
            f1[(j-1)*q+Y[m,j]] += counts[m]
            f2[(j-1)*q+Y[m,j], (j-1)*q+Y[m,j]] += counts[m]
            for i in (j+1):L
                f2[(i-1)*q+Y[m,i], (j-1)*q+Y[m,j]] += counts[m]
                f2[(j-1)*q+Y[m,j], (i-1)*q+Y[m,i]] += counts[m]
            end
        end
    end

    f2 = f2./N
    f1 = f1./N


        if pc != 0.
        f1 = (1-pc)*f1 .+ pc/q
        f2 = (1-pc)*f2 .+ pc/q/q
        for i in 1:size(Y,2)
            f2[(i-1)*q .+ (1:q), (i-1)*q .+ (1:q)] .= 0
            for a in 1:q
                f2[(i-1)*q+a, (i-1)*q+a] = f1[(i-1)*q+a]
            end
        end
    end

    return (f1,f2)
end

#  for smooth counts
function comp_freqs(Y::Array{Float64,2},counts::Array{Float64,1},q;pc=0.0)
    Y=Int64.(Y)
    (M,L) = size(Y)
     N=sum(counts)
     f1 = zeros(Float64,L*q)
     f2 = zeros(Float64, L*q, L*q)

    for m in 1:M
        if counts[m] > 0
            for j in 1:L
                f1[(j-1)*q+Y[m,j]] += counts[m]
                f2[(j-1)*q+Y[m,j], (j-1)*q+Y[m,j]] += counts[m]
                for i in (j+1):L
                    f2[(i-1)*q+Y[m,i], (j-1)*q+Y[m,j]] += counts[m]
                    f2[(j-1)*q+Y[m,j], (i-1)*q+Y[m,i]] += counts[m]
                end
            end
        end
    end

    f2 = f2./N
    f1 = f1./N


        if pc != 0.
        f1 = (1-pc)*f1 .+ pc/q
        f2 = (1-pc)*f2 .+ pc/q/q
        for i in 1:size(Y,2)
            f2[(i-1)*q .+ (1:q), (i-1)*q .+ (1:q)] .= 0
            for a in 1:q
                f2[(i-1)*q+a, (i-1)*q+a] = f1[(i-1)*q+a]
            end
        end
    end

    return (f1,f2)
end
"""

function compute_f1(Y::Array{Float64,2},counts::Array{Int64,1},q::Int64;pc=0.0)

Compute single site frequencies (`f1`) at a given time `t` for the observed sequences.

# Arguments
- `Y::Array{Float64,2}`: Matrix representing the observed sequences with each row as a sequence.
- `counts:: Vector of counts corresponding to each sequence.
- `q::Int64`: Number of states (e.g., nucleotides or amino acids).
- `pc::Float64`: Pseudocount, a small value to avoid division by zero. Defaults to 0.0.

# Returns
- `f1::Array{Float64,1}`: Array of single site frequencies.
"""

function compute_f1(Y::Array{Float64,2},counts::Array{Int64,1},q::Int64;pc=0.0)
    Y=Int64.(Y)
    (M,L) = size(Y)
    N=sum(counts)
    f1 = zeros(Float64,L*q)
    
    for m in 1:M
        for j in 1:L
            f1[(j-1)*q+Y[m,j]] += counts[m]      
        end
     end
    
    f1 = f1./N

    return f1  
end  



# Here the counts are frequencies
function compute_f1(Y::Array{Float64,2},counts::Array{Float64,1};q=21)
    Y=Int64.(Y)
    (M,L) = size(Y)
    f1 = zeros(Float64,L*q)
    N=sum(counts)
    for m in 1:M
        if counts[m] > 0  # only perform the calculation if counts are nonzero
            for j in 1:L
                f1[(j-1)*q+Y[m,j]] += counts[m]      
            end    
         end
    end
    f1 = f1./N

    return f1  
end 


"""

function compute_path_f_i(nVec::Vector{Vector{Int64}},sVec::Vector{Matrix{Float64}},q::Int64,L::Int64)

Compute frequency trajectories over time for each mutation.

# Arguments
- `nVec::Vector{Vector{Int64}}`: Vector of vectors representing population counts at different time points.
- `sVec::Vector{Matrix{Float64}}`: Vector of matrices representing sequences alignment at different time points.
- `q::Int64`: Number of states.
- `L::Int64`: Genome length.

# Returns
- `freq_vs_time::Matrix{Float64}`: Matrix containing frequency trajectories over time for each mutation.
"""

function compute_path_f_i(nVec::Vector{Vector{Int64}},sVec::Vector{Matrix{Float64}},q::Int64,L::Int64)
  interval=length(sVec)
  pseudoc=0.0 
  freq_vs_time=zeros(L*q,interval)
  for i in 1:interval
    counts=Int.(nVec[i]);
    (f1,f2)=comp_freqs(sVec[i],counts,q,pc=pseudoc)
    freq_vs_time[1:end,i] .=f1
  end
  
  return freq_vs_time
end   


function compute_path_f_i(seqs::Matrix{Float64},counts::Vector{Any},q::Int64)
    (M,L)=size(seqs)
    bins=length(counts)
    freq_vs_time=zeros(L*q,bins)
    
    for t0 in 1:bins
        f1=compute_f1(seqs,counts[t0],q=q);
        freq_vs_time[1:end,t0] .=f1
    end
    
    return freq_vs_time
end




"""
function select_traj(freq_vs_time::Matrix{Float64},pos::Int64,state::Int64,q::Int64)

Select the trajectory of a specific state at a given position over time.

# Arguments
- `freq_vs_time::Matrix{Float64}`: Matrix containing frequency trajectories over time.
- `pos::Int64`: Position of the mutation.
- `state::Int64`: State of the mutation.
- `q::Int64`: Number of states.

# Returns
- `traj::Array{Float64,1}`: Array representing the frequency trajectory of the specified mutation.
"""


function select_traj(freq_vs_time::Matrix{Float64},pos::Int64,state::Int64,q::Int64)
  if state >q
    return println("error state must be in [1:q] range")
  end     
  
  return freq_vs_time[(pos-1)*q+state,1:end]
end  



"""
select_seq_with_mut(data::Array{Float64, 2}, pos::Int64, state::Int64)

Select sequences with a specific state at a given position.

Parameters:
- `data` (Array{Float64, 2}): Matrix of sequences.
- `pos` (Int64): Position to check for the specified state.
- `state` (Int64): The state to check for.

Returns:
- Two arrays containing rows index of data matrix with and without the specified state at the given position.
"""
  
function select_seq_with_mut(data::Array{Float64, 2}, pos::Int64, state::Int64)
    seqs_with_mut = []
    seqs_no_mut = []

    for i in 1:size(data, 1)
        if data[i, pos] == state
            push!(seqs_with_mut, i)
        else
            push!(seqs_no_mut, i)
        end
    end

    return seqs_with_mut, seqs_no_mut
end



"""

function init_select_coeff(nB::Int64,nD::Int64,fB::Float64,fD::Float64,q::Int64,L::Int64,ref_seq::Array{Float64,1})

Initialize artificial selection coefficients.

# Arguments
- `nB::Int64`: Number of beneficial mutations.
- `nD::Int64`: Number of deleterious mutations.
- `fB::Float64`: Selection coefficient for beneficial mutations.
- `fD::Float64`: Selection coefficient for deleterious mutations.
- `q::Int64`: Number of states.
- `L::Int64`: Genome length.
- `ref_seq::Array{Float64,1}`: Reference sequence.

# Returns
- `h::Array{Float64,1}`: Vector of selection coefficients.

"""

function init_select_coeff(nB::Int64,nD::Int64,fB::Float64,fD::Float64,q::Int64,L::Int64,ref_seq::Array{Float64,1})
  # Vector of selection coefficients     
  h = zeros(q*L) 
     
  σh=0.2
    for i in 1:L
      if i<nB     
        #pos=rand(1:q-1)     
        h[(i-1)*q .+ (1:q)] .= fB#rand(Normal(fB,σh),q)      
      end   
      if i>L-nD 
        #pos=rand(1:q-1)     
        h[(i-1)*q .+ (1:q)] .=fD#rand(Normal(fD,σh),q)      
      end 

        h[(i-1)*q+Int.(ref_seq[i])]=0
    end 
    
    
    return  h  
  end    


"""
function fitness(field::Vector{Float64},seq::Array{Float64,1},q::Int64)

Compute fitness for a single sequence.

# Arguments
- `field::Vector{Float64}`: Selection coefficients.
- `seq::Array{Float64,1}`: Single sequence.
- `q::Int64`: Number of states.

# Returns
- `Float64`: Fitness value.
"""
 
function fitness(field::Vector{Float64},seq::Array{Float64,1},q::Int64)
  h = 1  
  for i in 1:length(seq)
    h += field[(i-1)*q+Int.(seq[i])]
  end
  
  return h
end    


"""
function fitness(selection::Vector{Float64},mat_seq::Matrix{Float64},q::Int64)

Compute fitness for each sequence in `mat_seq`.

# Arguments
- `selection::Vector{Float64}`: Selection coefficients.
- `mat_seq::Matrix{Float64}`: Matrix of sequences.
- `q::Int64`: Number of states.

# Returns
- `Array{Float64,1}`: Array of fitness values for each sequence.

"""

function fitness(selection::Vector{Float64},mat_seq::Matrix{Float64},q::Int64)
  num_seq=size(mat_seq,1)
  fitness_seq=zeros(num_seq)    
    
  for i in 1:num_seq
    fitness_seq[i]=fitness(selection,mat_seq[i,1:end],q)
  end    
  
  return fitness_seq
end    

        
function fitness_pop_clade(fields::Vector{Float64},seq::Matrix{Float64},counts_::Vector{Float64},clades_sequences::Matrix{Float64};q=21)
    positive_counts= counts_[counts_.>0.0]
    sequences=seq[counts_.>0.0,:]

    #select clade sequences
    A_rows =  eachrow(sequences)
    B_rows = Set(eachrow(clades_sequences))  # Use a Set for efficient lookup
    indices = findall(row -> row in B_rows, A_rows);
    if !isempty(indices)
        sequences=sequences[indices,:]
        positive_counts=positive_counts[indices]
    else
        return 0.0
    end
    
    

    tot_seqs=sum(positive_counts)

    tot_fitness=0.0
    for i in 1:length(positive_counts)
        tot_fitness+=fitness(fields,sequences[i,:],q)*positive_counts[i]
    end

    return tot_fitness/tot_seqs
        
end    


function pop_clade_frequency(seq::Matrix{Float64},counts_::Vector{Float64},clades_sequences::Matrix{Float64};q=21)
    positive_counts= counts_[counts_.>0.0]
    sequences=seq[counts_.>0.0,:]
        #select clade sequences
    A_rows =  eachrow(sequences)
    B_rows = Set(eachrow(clades_sequences))  # Use a Set for efficient lookup
    indices = findall(row -> row in B_rows, A_rows);
    positive_counts=positive_counts./sum(positive_counts)
    if !isempty(indices)
           return sum(positive_counts[indices])
    else
            return 0.0
    end
        
end    













#######Dynamics#############

"""
mutable struct Species


Mutable struct representing a species

# Fields:
- `n::Int64`: Number of members in the species population.
- `s::Array{Float64, 1}`: Sequence identifier, which is an array of floating-point numbers representing the genetic sequence of the species.
- `f::Float64`: Fitness of the species, indicating its ability to survive and reproduce in the given environment.
"""

mutable struct Species
    n::Int64 
    s::Array{Float64,1} 
    f::Float64  
end


"""
mutable struct smoothSpecies


Mutable struct representing a species with counts and sequences after smooth method

# Fields:
- `n::Float64`: Frequency of members in the species population.
- `s::Array{Float64, 1}`: Sequence identifier, which is an array of floating-point numbers representing the genetic sequence of the species.
- `f::Float64`: Fitness of the species, indicating its ability to survive and reproduce in the given environment.
"""

##Pop dynamics for effective counts
mutable struct smoothSpecies
    n::Float64 
    s::Array{Float64,1} 
    f::Float64  
end


"""
function init_Species(number_of_members::Int64,q::Int64,L::Int64,field::Array{Float64,1};seq=[])

Initialize a species with the given parameters.

# Arguments:
- `number_of_members::Int64`: Number of members in the species, representing the population size.
- `q::Int64`: Number of possible states for each element in the genetic sequence.
- `L::Int64`: Length of the genetic sequence.
- `field::Array{Float64,1}`: Array representing the selection coefficients for each state.
- `seq::Array{Float64,1}` (optional): Initial genetic sequence for the species. If not provided, a random sequence is generated.

# Returns:
- An instance of the `Species` struct.
"""

function init_Species(number_of_members::Int64,q::Int64,L::Int64,field::Array{Float64,1};seq=[])
    if isempty(seq)
      rand_seq=rand(1.0:q,L)  
      return Species(number_of_members,rand_seq,fitness(field,rand_seq,q))
    else
      return Species(number_of_members,seq,fitness(field,seq,q))
    end    
end 

"""
function init_Species(effectiveCount::Float64,q::Int64,L::Int64,field::Array{Float64,1};seq=[])

Initialize a species with the given parameters.

# Arguments:
- `effectiveCount::Float64`: Frequency of members in the species, representing the population size.
- `q::Int64`: Number of possible states for each element in the genetic sequence.
- `L::Int64`: Length of the genetic sequence.
- `field::Array{Float64,1}`: Array representing the selection coefficients for each state.
- `seq::Array{Float64,1}` (optional): Initial genetic sequence for the species. If not provided, a random sequence is generated.

# Returns:
- An instance of the `smoothSpecies` struct.
"""

function init_Species(effectiveCount::Float64,q::Int64,L::Int64,field::Array{Float64,1};seq=[])
    if isempty(seq)
      rand_seq=rand(1.0:q,L)  
      return smoothSpecies(effectiveCount,rand_seq,fitness(field,rand_seq,q))
    else
      return smoothSpecies(effectiveCount,seq,fitness(field,seq,q))
    end    
end 




"""
function clone(s::Species)

Clone a species, creating a new instance with a single member.

# Arguments:
- `s::Species`: The species to be cloned.

# Returns:
- A new copy of s  with a single member.
"""

function clone(s::Species)
    output=deepcopy(s)
    output.n=1
    return output  
end    
    



"""
function mutate(s::Species,mu::Float64,h::Array{Float64,1},q::Int64,L::Int64)

Mutate a species by introducing new mutations at different sites and return self + new species.

# Arguments:
- `s::Species`: The species to be mutated.
- `mu::Float64`: Mutation rate.
- `h::Array{Float64,1}`: selection coefficients values for each allele.
- `q::Int64`: Number of states.
- `L::Int64`: Length of the sequence.

# Returns:
- An array of mutated species with new sequences.

"""

function mutate(s::Species,mu::Float64,h::Array{Float64,1},q::Int64,L::Int64)
  # """ Mutate and return self + new sequences. """
        
  newSpecies = []
            
  if s.n>0
    nMut    = rand(Binomial(s.n,mu*L)) # get number of individuals that mutate
    s.n -= nMut # subtract number mutated from size of current clone
        
    # process mutations
    site = rand(1:L,nMut) # choose mutation sites at random
    for i in site
      new_s = clone(s) # create a new copy 
      current_mut = new_s.s[i]
      new_mut=rand(1.0:q)
      while new_mut==current_mut
        new_mut=rand(1.0:q)
      end
      
      new_s.s[i] =new_mut # mutate the randomly-selected site
      new_s.f = fitness(h,new_s.s,q)# update fitness
      push!(newSpecies,new_s)
    end
  end   
    
  if (s.n>0)
    #return self + new sequences  
    push!(newSpecies,s)
  end
        
  return newSpecies
    
end   


"""
function initial_population(sVec_i::Matrix{Float64},nVec_i::Array{Int64,1},q::Int64,h::Array{Float64,1})

Initialize a population of species from variants sequences and counts.

# Arguments:
- `sVec_i::Matrix{Float64}`: Matrix of initial sequences for each species.
- `nVec_i::Array{Int64,1}`: Array of initial counts for each species.
- `q::Int64`: Number of states.
- `h::Array{Float64,1}`: Selection coefficients for each allele.

# Returns:
- A population of species, an array of sequence matrices, and an array of count vectors.

"""

function initial_population(sVec_i::Matrix{Float64},nVec_i::Array{Int64,1},q::Int64,h::Array{Float64,1})
  pop = Species[]
  (M,L)=size(sVec_i)
  for i in 1:M     
    push!(pop,init_Species(nVec_i[i],q,L,h,seq=sVec_i[i,:])) # current population        
  end               
  return pop, [sVec_i], [nVec_i] 
 end 



"""
function initial_population(sVec_i::Matrix{Float64},effectiveCounts::Array{Float64,1},q::Int64,h::Array{Float64,1})

Initialize a population of species from variants sequences and counts.

# Arguments:
- `sVec_i::Matrix{Float64}`: Matrix of initial sequences for each species.
- `effectiveCounts::Array{Float64,1}`: Array of initial counts for each species normalized.
- `q::Int64`: Number of states.
- `h::Array{Float64,1}`: Selection coefficients for each allele.

# Returns:
- A population of species, an array of sequence matrices, and an array of count vectors.

"""



function initial_population(sVec_i::Matrix{Float64},effectiveCounts::Array{Float64,1},q::Int64,h::Array{Float64,1})
  pop = smoothSpecies[]
  (M,L)=size(sVec_i)
  for i in 1:M     
    push!(pop,init_Species(effectiveCounts[i],q,L,h,seq=sVec_i[i,:])) # current population        
  end               
  return pop, [sVec_i], [effectiveCounts] 
 end 




"""
function initial_population(referece_seq::Vector{Float64},h::Array{Float64,1},N::Int64,q::Int64,L::Int64)
  

Initialize a population of species from one sequence.

# Arguments:
- `referece_seq`:  initial sequence.
- `h::Array{Float64,1}`: Selection coefficients for each allele.
- `N::Int64`: Total population size.
- `q::Int64`: Number of states.

# Returns:
- A population of species, an array of sequence matrices, and an array of count vectors.

"""


function initial_population(referece_seq::Vector{Float64},h::Array{Float64,1},N::Int64,q::Int64)
    
    L=length(referece_seq)
    # set fitness of reference sequence to f=1    
    for i in 1:L
      h[(i-1)*q+Int.(referece_seq[i])]=0
    end 


    pop, sVec, nVec = Species[], Array{Float64,2}[], Vector{Int64}[]
     
      temp_sVec = zeros(1,L)
      temp_nVec = Int64[]
        
      push!(pop,init_Species(N,q,L,h,seq=referece_seq))
      temp_sVec[1,:] .= referece_seq
      push!(temp_nVec,N)
      push!(sVec,temp_sVec)
      push!(nVec,temp_nVec)  
 
    return pop, sVec, nVec  
 end    




"""
function initial_population(randomize::Int64,h::Array{Float64,1},N::Int64,q::Int64,L::Int64)

Generate a initial population of species with random sequences, fitness, and counts.

# Arguments
- `randomize::Int64`: Number of clusters or randomization factor for initial sequences.
- `h::Array{Float64,1}`: selection coefficients for each allele.
- `N::Int64`: Total population size.
- `q::Int64`: Number of states.
- `L::Int64`: Length of the sequence.
  
# Returns
- `pop::Vector{Species}`: Vector of Species representing the initial population.
- `sVec::Vector{Array{Float64,2}}`: Array of sequences at each time point (initially).
- `nVec::Vector{Vector{Int64}}`: Array of sequence counts at each time point (initially).

# Details
- If `randomize > 1`, the population is divided into `randomize` clusters, each contributing a unique initial sequence.
- Each cluster's size is proportional to the total population, ensuring that the sum of the cluster sizes equals `N`.
- If `randomize <= 1`, a single initial sequence is used for the entire population.
"""

 function initial_population(randomize::Int64,h::Array{Float64,1},N::Int64,q::Int64,L::Int64)
  
    pop, sVec, nVec = Species[], Array{Float64,2}[], Vector{Int64}[]
     
    if randomize>1
      if randomize>N
        randomize = N
      end
      n_seqs    = Int64(floor(N/randomize))
      temp_sVec = zeros(randomize,L)
      temp_nVec = Int64[]
      for i in 1:randomize
        temp_seq =rand(1.0:q,L) 
        if i==randomize
          n_seqs = N - sum(temp_nVec)
        end
        push!(pop,init_Species(n_seqs,q,L,h,seq=temp_seq))
        temp_sVec[i,:] .= temp_seq
        push!(temp_nVec,n_seqs)
      end
      
      push!(sVec,temp_sVec)
      push!(nVec,temp_nVec)  
           
    else
      temp_seq=rand(1.0:q,L) 
      push!(pop, init_Species(N,q,L,h,seq=temp_seq[1:end])) # current population
      push!(sVec,temp_seq)    # array of sequences at each time point
      push!(nVec,N*ones(Int64,1)) # array of sequence counts at each time point
    end

    return pop, sVec, nVec  
 end    

"""
function evolve_pop_WF(T::Int64, N::Int64, q::Int64, L::Int64, mu::Float64, h::Vector{Float64}, i_pop::Array{Species,1}, i_sVec::Vector{Matrix{Float64}},i_nVec::Vector{Vector{Int64}}; record=1, ReturnLast=true)

Evolve a population through a Wright-Fisher model for a specified number of generations.

# Arguments
- `T::Int64`: Number of generations to simulate.
- `N::Int64`: Total population size.
- `q::Int64`: Number of states of alleles.
- `L::Int64`: Length of the sequence.
- `mu::Float64`: Mutation rate per sequence per generation.
- `h::Vector{Float64}`: Selection coefficients for each allele.
- `i_pop::Array{Species,1}`: Initial population.
- `i_sVec::Vector{Matrix{Float64}}`: Initial array of sequences at each time point.
- `i_nVec::Vector{Vector{Int64}}`: Initial array of sequence counts at each time point.
- `record::Int64`: Record data every `record` generations (default is 1).
- `ReturnLast::Bool`: Return data for the last generation only (default is true).

# Returns
- `pop::Array{Species,1}`: Final population after evolution.
- `nVec::Vector{Vector{Int64}}`: Array of sequence counts at each time point.
- `sVec::Vector{Matrix{Float64}}`: Array of sequences at each time point.

# Details
- The Wright-Fisher model simulates genetic drift and mutation in a population.
- Fitness values influence the probability of selection for each allele.
"""

function evolve_pop_WF(T::Int64, N::Int64, q::Int64, L::Int64, mu::Float64, h::Vector{Float64},
                        i_pop::Array{Species,1}, i_sVec::Vector{Matrix{Float64}},
                        i_nVec::Vector{Vector{Int64}}; record=1, ReturnLast=true)
    pop = deepcopy(i_pop)
    sVec = Array{Float64,2}[]
    nVec = Vector{Int64}[]
    push!(sVec, i_sVec[1])
    push!(nVec, i_nVec[1])

    tStart = 1       # start generation
    tEnd   = T      # end generation  
    
    mutations_present = mu != 0.0
    newPop = mutations_present ? Species[] : nothing

    for t in tStart:tEnd    
        r = [s.n * s.f for s in pop]
        r = abs.(r)
        p = r / sum(r) # probability of selection for each species (sequence)
        n = rand(Multinomial(N, p)) # selected number of each species       
        # Update population size and mutate
        
        for i in 1:length(pop)
            pop[i].n = n[i] # set new number of each species
            
           if mutations_present
              # include mutations, then add mutants to the population array
              p = mutate(pop[i], mu, h, q, L)
            
              for j in 1:length(p)
                unique = true
                for k in 1:length(newPop)
                    if p[j].s == newPop[k].s
                        unique = false
                        newPop[k].n += p[j].n
                        break
                    end
                end
                
                if unique
                    push!(newPop, p[j])
                end  
              end
            end      
        end
        
       if mutations_present
            pop = deepcopy(newPop)
            newPop = Species[]  # reset newPop for the next iteration
        end
        
        # Update measurements
   
        if t % record == 0
            ad_s = zeros(length(pop), L)
            i = 1
            for s in pop
                ad_s[i, :] = s.s
                i += 1  
            end 
            push!(nVec, [s.n  for s in pop])  
            push!(sVec, ad_s)      
        end
        # End and output total time
    
    end

    if ReturnLast
        return pop, sVec[end], nVec[end]
    else
        return pop, sVec, nVec
    end  
end

"""
function evolve_pop_WF(T::Int64, N::Int64, q::Int64, L::Int64,h::Vector{Float64}, i_pop::Array{smoothSpecies,1}, i_sVec::Vector{Matrix{Float64}},i_nVec::Vector{Vector{Float64}}; record=1, ReturnLast=true)

Evolve a population through a Wright-Fisher model for a specified number of generations.

# Arguments
- `T::Int64`: Number of generations to simulate.
- `N::Int64`: Total population size.
- `q::Int64`: Number of states of alleles.
- `L::Int64`: Length of the sequence.
- `h::Vector{Float64}`: Selection coefficients for each allele.
- `i_pop::Array{smoothSpecies,1}`: Initial population.
- `i_sVec::Vector{Matrix{Float64}}`: Initial array of sequences at each time point.
- `i_nVec::Vector{Vector{Float64}}`: Initial array of sequence frequencies at each time point.
- `record::Int64`: Record data every `record` generations (default is 1).
- `ReturnLast::Bool`: Return data for the last generation only (default is true).

# Returns
- `pop::Array{Species,1}`: Final population after evolution.
- `nVec::Vector{Vector{Int64}}`: Array of sequence counts at each time point.
- `sVec::Vector{Matrix{Float64}}`: Array of sequences at each time point.

# Details
- The Wright-Fisher model simulates genetic drift and mutation in a population.
- Fitness values influence the probability of selection for each allele.
"""


 function evolve_pop_WF(T::Int64, N::Int64, q::Int64, L::Int64, h::Vector{Float64},
                        i_pop::Array{smoothSpecies,1}, i_sVec::Vector{Matrix{Float64}},
                        i_nVec::Vector{Vector{Float64}}; record=1, ReturnLast=true)
    
    pop = deepcopy(i_pop)
    sVec = Array{Float64,2}[]
    nVec = Vector{Float64}[]
    push!(sVec, i_sVec[1])
    push!(nVec, i_nVec[1])

    tStart = 1       # start generation
    tEnd   = T      # end generation  
    
    for t in tStart:tEnd    
        r = [s.n * s.f for s in pop]
        r = abs.(r)
        p = r / sum(r) # probability of selection for each species (sequence)
        n = rand(Multinomial(N, p))./N # selected number of each species counts

        # Update population size and mutate
        
        for i in 1:length(pop)
            pop[i].n = n[i] # set new number of each species counts
           
        end
        
        # Update measurements
   
        if t % record == 0
            ad_s = zeros(length(pop), L)
            i = 1
            for s in pop
                ad_s[i, :] = s.s
                i += 1  
            end 
            push!(nVec, [s.n  for s in pop])  
            push!(sVec, ad_s)      
        end
        # End and output total time
    
    end

    if ReturnLast
        return pop, sVec[end], nVec[end]
    else
        return pop, sVec, nVec
    end  
end





"""
evolve_pop_NB(T::Int64, N::Int64, q::Int64, L::Int64, mu::Float64, h::Vector{Float64}, k::Float64, R::Float64, 
              max_pop_size::Int64, i_pop::Array{Species,1}, i_sVec::Vector{Matrix{Float64}}, i_nVec::Vector{Vector{Int64}};
              record=1)
    
    Simulates the evolution of a population over discrete time intervals using a Negative Binomial distribution.
    
    Parameters:
    - T (Int64): Total number of generations to simulate.
    - N (Int64): Initial population size.
    - q (Int64): Number of possible states for each allele.
    - L (Int64): Length of the sequence.
    - mu (Float64): Mutation rate.
    - h (Vector{Float64}): Vector of selection coefficients for each allele state.
    - k (Float64): Dispersion parameter for the Negative Binomial distribution.
    - R (Float64): Effective reproductive number.
    - max_pop_size (Int64): Maximum allowed population size. If the population exceeds this size, 
                           R is adjusted to control growth.
    - i_pop (Array{Species,1}): Initial population.
    - i_sVec (Vector{Matrix{Float64}}): Initial sequence configurations at each time point.
    - i_nVec (Vector{Vector{Int64}}): Initial sequence counts at each time point.
    - record (Int64): Frequency of recording population state (default is 1, records every generation).
    
    Returns a tuple containing:
    - pop (Array{Species,1}): Final population after the simulation.
    - sVec (Vector{Matrix{Float64}}): Array of sequence configurations at each recorded time point.
    - nVec (Vector{Vector{Int64}}): Array of sequence counts at each recorded time point.
"""



 function evolve_pop_NB(T::Int64, N::Int64, q::Int64, L::Int64, mu::Float64, h::Vector{Float64}, k::Float64, R::Float64, 
                       max_pop_size::Int64, i_pop::Array{Species,1}, i_sVec::Vector{Matrix{Float64}}, 
                       i_nVec::Vector{Vector{Int64}}; record=1)
    pop = deepcopy(i_pop)
    sVec = Array{Float64,2}[]
    nVec = Vector{Int64}[]
    push!(sVec, i_sVec[1])
    push!(nVec, i_nVec[1])

    tStart = 1       # start generation
    tEnd = T         # end generation
    total_population = sum([s.n for s in pop])

    for t in tStart:tEnd
        #println("time=$(t)")
        #println("R=$(R)")
        #println("total_population=$(total_population)")

        average_fitness = sum([s.f * s.n for s in pop]) / sum([s.n for s in pop])

        if total_population >= max_pop_size
            R = (1 / average_fitness) * (max_pop_size / total_population)
        end
        #println("R=$(R)")

        r = [s.n * k for s in pop]
        r[r .== 0.0] .= 0.00001
        R_a = [R * s.f for s in pop]
        #println("Ra-max=$(findmax(R_a)[1])")
        
        p = k ./(k .+ R_a)

        n = rand.(map(NegativeBinomial, r, p))
        total_population = sum(n)

        while total_population < 1000  # ensures that the population doesn't die off by resampling if there are too few infected
            n = rand.(map(NegativeBinomial, r, p))  # selected number of each species
            total_population = sum(n)
        end

        # Update population size and mutate
        newPop = Species[]
        for i in 1:length(pop)
            pop[i].n = n[i]  # set new number of each species
            # include mutations, then add mutants to the population array
            p = mutate(pop[i], mu, h, q, L)
            for j in 1:length(p)
                unique = true
                for k in 1:1:length(newPop)
                    if p[j].s == newPop[k].s
                        unique = false
                        newPop[k].n += p[j].n
                        break
                    end
                end

                if unique
                    push!(newPop, p[j])
                end
            end
        end
        pop = newPop

        # Update measurements
        if t % record == 0
            ad_s = zeros(length(pop), L)
            i = 1
            for s in pop
                ad_s[i, :] = s.s
                i += 1
            end
            push!(nVec, [s.n for s in pop])
            push!(sVec, ad_s)
        end
    end
    return pop, sVec, nVec
end





