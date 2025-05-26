using StatsBase, Dates, Measures, DataStructures, Random, Statistics, DelimitedFiles, DataFrames
using CSV

using BioSequences
#using FluPredictibility, FluPredictibility.BioTools, FluPredictibility.TreeTools
#Flu = FluPredictibility


const intToaa= Dict(1=>AA_Gap,2=>AA_A, 3=>AA_C, 4=>AA_D, 5=>AA_E, 6=>AA_F, 7=>AA_G, 8=>AA_H, 9=>AA_I, 10=>AA_K, 11=>AA_L, 12=>AA_M, 13=>AA_N, 14=>AA_P, 15=>AA_Q, 16=>AA_R, 17=>AA_S, 18=>AA_T, 19=>AA_V, 20=>AA_W, 21=>AA_Y);

const charToaa= Dict('-'=>AA_Gap,'A'=>AA_A, 'C'=>AA_C, 'D'=>AA_D, 'E'=>AA_E, 'F'=>AA_F, 'G'=>AA_G, 'H'=>AA_H, 'I'=>AA_I, 'K'=>AA_K, 'L'=>AA_L, 'M'=>AA_M, 'N'=>AA_N, 'P'=>AA_P, 'Q'=>AA_Q, 'R'=>AA_R, 'S'=>AA_S, 'T'=>AA_T, 'V'=>AA_V, 'W'=>AA_W, 'Y'=>AA_Y);
const aaToint = Dict(AA_Term=>1,AA_B=>1,AA_J=>1,AA_X=>1,AA_Z=>1,AA_Gap => 1,AA_A=>2,AA_C=>3, AA_D=>4, AA_E=>5, AA_F=>6, AA_G=>7, AA_H=>8, AA_I=>9, AA_K=>10, AA_L=>11, AA_M=>12, AA_N=>13, AA_P=>14, AA_Q=>15, AA_R=>16, AA_S=>17, AA_T=>18, AA_V=>19, AA_W=>20, AA_Y=>21)


mutable struct Pop 
    MSA::Matrix{Float64}
    weigts::Vector{Float64}
    time::Int64
end


function transform_seq(seq::Any,D::Dict{Int,AminoAcid})
    q=length(seq)
    let_seq=Vector{AminoAcid}(undef,q)
    for i in 1:q
       let_seq[i]=D[seq[i]]
    end    
  return let_seq
end 


function transform_aa_seq(aa_seq_::Vector{AminoAcid})
   seq = deepcopy(aa"")
    for i in 1:length(aa_seq_)
        push!(seq,aa_seq_[i])
    end  
   return seq
end

function transform_seq(seq::Any,D::Dict{Char,AminoAcid})
    q=length(seq)
    let_seq=Vector{AminoAcid}(undef,q)
    for i in 1:q
       let_seq[i]=D[seq[i]]
    end    
  return let_seq
end 


function transform_seq(seq::Any,D::Dict{AminoAcid,Int64})
    q=length(seq)
    let_seq=Array{Int64}(undef,q)
    for i in 1:q
       let_seq[i]=D[seq[i]]
    end    
  return let_seq
end 


function sequence_distribution(msa::Dict{String, Float64})
    d = Dict{BioSequence, Float64}()
    all_seqs=collect(keys(msa))
    all_values=collect(values(msa))
    for i in 1:length(all_values)
        
        aa_seq=transform_aa_seq(transform_seq(all_seqs[i],charToaa))
        d[aa_seq] = msa[all_seqs[i]]
    end
    return d
end


function sequence_distribution(msa::Matrix{Float64},weigts::Vector{Float64})
	d = Dict{BioSequence, Float64}()
    (m,l)=size(msa)
	for i in 1:m
        aa_seq=transform_aa_seq(transform_seq(Int.(msa[i,:]),intToaa))
		d[aa_seq] = weigts[i]
	end
	return d
end

function expand_msa(msa::Matrix{Float64},count::Vector{Int64})
    (M,L)=size(msa)
    resize=sum(count)
    seqs=zeros(resize,L)
    pos=1
    for i in 1:length(count)
        for j in 1:count[i]
           seqs[pos,:] .= msa[i,:]
            pos+=1
        end
    end    
    return seqs
end    

function find_closest(A::AbstractArray{T}, b::T) where {T<:Real}
    if length(A) <= 1
        return firstindex(A)
    end

    i = searchsortedfirst(A, b)

    if i == firstindex(A)
        return i
    elseif i > lastindex(A)
        return lastindex(A)
    else
        prev_dist = b - A[i-1]
        next_dist = A[i] - b

        if prev_dist < next_dist
            return i - 1
        else
            return i
        end
    end
end

function get_bioseq_array_from_msa(msa::Matrix{Float64})
    (M,L)=size(msa)
    #dd=Vector{BioSequence}(undef,M)
	dd = BioSequence[]
	for i in 1:M
        aa_seq=transform_aa_seq(transform_seq(Int.(msa[i,:]),intToaa))
		push!(dd,aa_seq)
	end
	return dd
end

function lower_bound_dist(d0::Dict{BioSequence, Float64},dtarget::Dict{BioSequence, Float64})
    P=deepcopy(d0)
    Q=dtarget
    f=BioTools.hamming
    # Building the distance matrix
	d = Array{Float64, 2}(undef, length(P), length(Q))
	for (i,(x,p)) in enumerate(P)
		for (j,(y,q)) in enumerate(Q)
			d[i,j] = f(x,y)
		end
	end

    for key in keys(P)
        P[key] = 0.0
    end
    
    (n1,n2)=size(d)

    for i in 1:n2
        (dis,pos)=findmin(d[:,i])
        s1=collect(keys(P))[pos,:][1]
        P[s1]+=collect(values(dtarget))[i]   
    end  

    return P
      
end    






function select_distribution_subset(msa::Dict{BioSequence, Float64},thr::Float64)
    d = Dict{BioSequence, Float64}()
    all_seqs=collect(keys(msa))
    all_values=collect(values(msa))
    
    for i in 1:length(all_values)
        if  msa[all_seqs[i]]>=thr
            d[all_seqs[i]] = msa[all_seqs[i]]
        end    
    end
    
    #normalize
    norm=sum(collect(values(d)))
    new_seqs=collect(keys(d))
    
    for i in 1:length(new_seqs)
         d[new_seqs[i]] =  d[new_seqs[i]]/norm
    end
    
    
    return d
end

function top_indices(arr::Array{Float64},n::Int64) 
    # Get the indices of the array sorted by descending value
    sorted_indices = sortperm(arr, rev=true)
    
    # Return the first 10 indices
    return sorted_indices[1:min(n, length(sorted_indices))]
end


function select_distribution_subset(msa::Dict{BioSequence, Float64},seqs_number::Int64)
    
    d = Dict{BioSequence, Float64}()
    all_seqs=collect(keys(msa))
    all_values=collect(values(msa))
    if length(all_seqs)>seqs_number
        indx=top_indices(all_values,seqs_number)
        selected_sqs=all_seqs[indx]
        norm=sum(all_values[indx])
        for i in 1:length(indx)
            d[selected_sqs[i]] = msa[selected_sqs[i]]/norm 
        end
    #normalize
    
        return d
    else
        
     return msa

    end    

    
end


function consensus_sequence(data::Dict{BioSequence, Float64};q=21)   
    M=length(data)
    sqs=collect(keys(data))
    L=length(sqs[1])

    msa=zeros(M,L)
    freq_=zeros(M)
    for i in 1:M
        msa[i,:] .= transform_seq(sqs[i],aaToint)
        freq_[i] =data[sqs[i]]
    end
    f_mat=reshape(compute_f1(msa,freq_),(q,L))
    ref_s=map(x->findmax(f_mat[:,x])[2],1:L);
    aa_s=transform_seq(ref_s,intToaa)
    bioseq=transform_aa_seq(aa_s)
    return bioseq
end        



