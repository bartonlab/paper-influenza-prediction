
 """
 function paramAlign(variant::String, protein::String)

    Determine the parameters (length and time) for sequence alignment based on the virus variant and protein.

    Parameters:
        - variant: String, the virus variant (e.g., "H3N2" or "H1N1pdm").
        - protein: String, the target protein (e.g., "HA" or "NA").

    Returns:
        - L: Int, the length of the protein sequence.
        - T: Int, the time duration for the sequence alignment.

    Raises:
        - Error: If the variant or protein name is not recognized.

    Example:
        L, T = paramAlign("H3N2", "HA")
    """


function paramAlign(variant::String, protein::String)
   
    if variant == "H3N2" && protein == "HA"
        L = 566
        T = 122
        return L, T
    elseif variant == "H3N2" && protein == "NA"
        L = 469
        T = 121
        return L, T
    elseif variant == "H1N1pdm" && protein == "HA"
        L = 566
        T = 133
        return L, T
    elseif variant == "H1N1pdm" && protein == "NA"
        L = 470
        T = 133
        return L, T
    else
        println("error: wrong definition of variant or protein name ")
        return 0,0
    end
end



"""
observation_time(variant::String, protein::String, year::String)

Determine the observation time in months based on the virus variant, target protein, and the year of observation.

# Arguments
- `variant`: String, the virus variant (e.g., "H3N2" or "H1N1pdm").
- `protein`: String, the target protein (e.g., "HA" or "NA").
- `year`: String, the year of observation.

# Returns
- Tuple: A tuple containing the observation time in months. If the variant, protein, or year is not recognized, the function prints an error message and returns (0, 0).
"""
function observation_time(variant::String, protein::String,year::String)
    if variant == "H3N2" && protein == "HA"
        year_to_month_dict = Dict("2009" => 0, "2011" => 24, "2013" => 45, "2015" => 63);
        return year_to_month_dict[year]
    elseif variant == "H3N2" && protein == "NA"
         year_to_month_dict = Dict("2009" => 0, "2011" => 24, "2013" => 48, "2015" => 72);
        return year_to_month_dict[year]
    elseif variant == "H1N1pdm" && protein == "HA"
        year_to_month_dict = Dict("2009" => 0, "2011" => 20, "2013" => 43, "2015" => 72);
        return year_to_month_dict[year]
    elseif variant == "H1N1pdm" && protein == "NA"
         year_to_month_dict = Dict("2009" => 0, "2011" => 20, "2013" => 43, "2015" => 72);
        return year_to_month_dict[year]
    else
        println("error: wrong definition of variant or protein name ")
        return 0,0
    end
end



### To compute enrichment for aa sequences



"""
Map selection positions to genome positions.

# Arguments
- `pos::Int64`: The selection position to be mapped.
- `q::Int64`: The number of sequences per time point.
- `L::Int64`: The sequence length.

# Returns
- `Int64`: The corresponding genome position for the selection position.
"""
function map_sc_pos_to_genome_pos(pos::Int64, q::Int64, L::Int64)
    expanded_vec = collect(1:q*L)
    expanded_vec = reshape(expanded_vec, (q, L))
    for i in 1:L
        if pos in expanded_vec[:, i]
            return i
        end
    end
    println("Error: position was not found")
end

"""
function find_common_elements(list1::Vector{Int64}, list2::Vector{Int64})

Find common elements between two lists.

# Arguments
- `list1::Vector{Int64}`: The first list.
- `list2::Vector{Int64}`: The second list.

# Returns
- `Vector{Int64}`: A list containing common elements found in both input lists.
"""

function find_common_elements(list1::Vector{Int64}, list2::Vector{Int64})
    output = Int64[]
    for i in 1:length(list1)
        if list1[i] in list2
            push!(output, list1[i])
        end
    end
    return output

end

"""
function find_elements_not_in_list(list1::Vector{Int64}, list2::Vector{Int64})

Find elements in the first list that are not present in the second list.

# Arguments
- `list1::Vector{Int64}`: The first list.
- `list2::Vector{Int64}`: The second list.

# Returns
- `Vector{Int64}`: A list containing elements from `list1` that are not present in `list2`.
"""

# Function to find elements in list1 that are not in list2
function find_elements_not_in_list(list1::Vector{Int64}, list2::Vector{Int64})
    output = Int64[]
    for i in 1:length(list1)
        if !(list1[i] in list2)
            push!(output, list1[i])
        end
    end
    return output
end

"""
Select the top mutations based on selection coefficients.

# Arguments
- `selec_coeff::Vector{Float64}`: Vector of selection coefficients.
- `percent::Float64`: Percentage of top mutations to select.

# Returns
- `Vector{Int64}`: Indices of the selected top mutations.
"""

function select_top_mutations(selec_coeff::Vector{Float64}, percent::Float64)
   
    select_topXpcent = Int64.(round(percent * length(selec_coeff)))
    top_mut = sort(selec_coeff, rev=true)[1:select_topXpcent]

    top_positions = []
      for i in 1:length(top_mut)
        push!(top_positions, findall(x -> x == top_mut[i], selec_coeff))
      end

    return unique(vcat(top_positions...))

end

"""
function filter_top_mutations(fields::Vector{Float64}, percent::Float64, full_path_freq_::Matrix{Float64}, q::Int64, L::Int64)

Filter top mutations that have never been seen with f>0.1 in two consecutive bins or correspond to a gap.

# Arguments
- `fields::Vector{Float64}`: Vector of selection coefficients.
- `percent::Float64`: Percentage of top mutations to select.
- `full_path_freq_::Matrix{Float64}`: Matrix of full path frequencies.
- `q::Int64`: Number of sequences per lineage.
- `L::Int64`: Length of the genome.

# Returns
- `Vector{Int64}`: Filtered indices of top mutations.
"""

function filter_top_mutations(fields::Vector{Float64}, percent::Float64, full_path_freq_::Matrix{Float64}, q::Int64, L::Int64)
    top_sc = select_top_mutations(fields, percent)

    top_sc_filt = Int64[]
    for i in top_sc
        for t in 1:size(full_path_freq_, 2) - 1
            if full_path_freq_[i, t] > 0.1 && full_path_freq_[i, t + 1] > 0.1
                push!(top_sc_filt, i)
            end
        end
    end
    top_sc_filt = unique(top_sc_filt)

    # Remove gaps positions
    gaps_positions = [q * (i - 1) + 1 for i in 1:L]

    top_sc_filt = find_elements_not_in_list(top_sc_filt, gaps_positions)

    return top_sc_filt
end




"""
function compute_enrichment_aa(top_sc::Vector{Int64}, epitope_pos::Vector{Float64}, q::Int64, L::Int64)

Compute enrichment for amino acid sequences.

# Arguments
- `top_sc::Vector{Int64}`: Positions of top mutations based on selection coefficients.
- `epitope_pos::Vector{Float64}`: Positions in the epitope list.
- `q::Int64`: Number of variants.
- `L::Int64`: Length of the genome sequence.

# Returns
- `rr1::Float64`: Fraction of top mutations in the epitope.
- `rr2::Float64`: Fraction of epitope positions in the genome.
- `enrichment::Float64`: Enrichment score (rr1 / rr2).
"""
function compute_enrichment_aa(top_sc::Vector{Int64}, epitope_pos::Vector{Float64}, q::Int64, L::Int64)
    # Transforming positions from q*L vector s_i(a) to genome sequence positions 1:L
    Nsel = map_sc_pos_to_genome_pos.(top_sc, q, L)
    Nsel = unique(Nsel)

    # Select positions in epitope list
    nsel = find_common_elements(Nsel, Int.(epitope_pos))

    # Output score
    rr1 = length(nsel) / length(Nsel)
    rr2 = length(epitope_pos) / L

    return rr1, rr2, rr1 / rr2
end


#######

###smhot by days utils
##########
function export_data_by_days(sVec::Vector{Matrix{Float64}},nVec::Vector{Vector{Int64}})
    data=[]
    L=size(sVec[1],2)
    for i in 1:size(sVec,1)
        if !(isempty(sVec[i]))
            msa=sVec[i]
            count=nVec[i]
            for s in 1:size(msa,1)
                seq=zeros(L+2)
                seq[1]=i
                seq[2]= Float64(count[s])
                seq[3:end].=msa[s,:]
                push!(data,seq)
            end       
        end    
    end    

    M=length(data)
    output=zeros(M,L+2)
    for i in 1:M
       output[i,:]=data[i]
    end    

    #shift time so t1=1
    p=0
    for t in 1:length(sVec)
        if isempty(sVec[t])
           p+=1
        else
          break  
        end
    end    
    
    output[:,1]=output[:,1] .- p

    return Int64.(output)
end 





function export_cpp_input_data(sVec::Vector{Matrix{Float64}},nVec::Vector{Vector{Int64}})
    data=[]
    L=size(sVec[1],2)
    for i in 1:size(sVec,1)
        if !(isempty(sVec[i]))
            msa=sVec[i]
            count=nVec[i]
            for s in 1:size(msa,1)
                seq=zeros(L+2)
                seq[1]=i
                seq[2]= Float64(count[s])
                seq[3:end].=msa[s,:] .- 1.0 # transform to 0...q-1 alphabet
                push!(data,seq)
            end       
        end    
    end    

    M=length(data)
    output=zeros(M,L+2)
    for i in 1:M
       output[i,:]=data[i]
    end    

   #shift time so t1=1
    p=0
    for t in 1:length(sVec)
        if isempty(sVec[t])
           p+=1
        else
          break  
        end
    end    
    
    output[:,1]=output[:,1] .- p


    return Int64.(output)
end    


function df_to_latex(df)
    ncols = size(df, 2)
    header = join(["c" for _ in 1:ncols], " | ")
    latex_code = "\\begin{tabular}{" * header * "}\n\\hline\n"
    latex_code *= join(names(df), " & ") * " \\\\\n\\hline\n"
    for row in eachrow(df)
        latex_code *= join(row, " & ") * " \\\\\n\\hline\n"
    end
    latex_code *= "\\end{tabular}"
    return latex_code
end

    
function Remove_ambigous_state(data_::Matrix{Float64};tstep=10.0,tsmooth=60.0,q=21)
    #import original data
    times=data_[1:end,1] #observation times
    counts=data_[1:end,2] #sequences count
    seqs=data_[1:end,3:end];# sequences configuration
    #unique observation times.
    u_times=unique(times);
    # Genome length
    L=size(seqs,2) ;

    println("1. original data imported!")
 
    # Smooth and compute full frequency trajectory for each mutation
    effectiveCounts_full=[]
    times_measure=Float64[]
    for t_ in 1.0:tstep:last(u_times)
        push!(times_measure,t_)
        effectiveCounts_=smoothCounts(counts,times,t_,tsmooth);
        push!(effectiveCounts_full,effectiveCounts_)    
    end    
    full_path_freq=compute_path_f_i(seqs,effectiveCounts_full,q);
    Tf=size(full_path_freq,2) ;    
    println("2. frequency computed!")

    
    seqs_changed=deepcopy(seqs)
    for i in 1:length(times)
        time_0=times[i]
        obj=seqs[i,:]
        pos=findall(x->x==1.0,obj)
        closest_time = argmin(abs.(times_measure .- time_0))
       
        if !isempty(pos)
            println("sequence $(i)")
            println("real time $(time_0)")
    
            println("effective time  $(closest_time)")
            for k in pos
                println("position $(k)")
                old_state=seqs_changed[i,k]
                println("old state $(old_state)")
    
               (freq_max,new_st)= findmax(full_path_freq[[(k-1)*q+a for a in 1:q],closest_time])
                println("new state $(new_st) and max_freq=$(freq_max)")
    
                seqs_changed[i,k]=new_st
            end    
        end    
    end    

    new_data=deepcopy(data_)
    new_data[1:end,3:end].=seqs_changed

    return new_data,seqs_changed
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
