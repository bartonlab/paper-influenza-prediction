import JuMP
import COSMO, MathOptInterface
import FluPredictibility.BioTools
using BioSequences: BioSequence
using Ipopt


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

    d[new_seqs[end]]+=1.0 - sum(collect(values(d)))

    return d
end


function EMD_Ipopt(P,Q)
	# Building the distance matrix
	f=BioTools.hamming

	d = Array{Float64, 2}(undef, length(P), length(Q))
	for (i,(x,p)) in enumerate(P)
		for (j,(y,q)) in enumerate(Q)
			d[i,j] = f(x,y)
		end
	end
	# Building the JuMP model
	model = JuMP.Model(Ipopt.Optimizer)
	JuMP.set_silent(model)
	JuMP.@variable(model, w[i=1:length(P), j=1:length(Q)] >= 0)
	for (i,(x,p)) in enumerate(P)
		JuMP.@constraint(model, sum(w[i,j] for j in 1:length(Q)) ==  p)
	end
	for (j,(y,q)) in enumerate(Q)
		JuMP.@constraint(model, sum(w[i,j] for i in 1:length(P)) ==  q)
	end
	JuMP.@objective(model, Min, sum(sum(w.*d)))
	JuMP.optimize!(model)
	# Looking at the solution
	if JuMP.termination_status(model) != MathOptInterface.OPTIMAL
		@warn "Termination status of optimization: $(JuMP.termination_status(model))"
	end
	flow = JuMP.value.(w)
	emd = JuMP.objective_value(model)

	return (emd=emd, flow=flow, status=JuMP.termination_status(model))
end