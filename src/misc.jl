
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



