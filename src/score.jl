using DataFrames
using ROC  # Assumes ROC.roc and AUC are available


"""
parse_string(str::String)

Parse a comma-separated string into an array of Float64 values.

# Arguments
- `str`: String, the input comma-separated string.

# Returns
- Array{Float64}: An array of Float64 values parsed from the input string.
"""


function parse_string(str::String)
   return parse.(Float64, split(chop(str; head=1, tail=1), ','))
end    


"""
unique_condition(position::Vector{Int64}, nucleotide::Vector{Int64})

Check for unique conditions based on position and nucleotide pairs.

# Arguments
- `position`: Vector{Int64}, the positions.
- `nucleotide`: Vector{Int64}, the nucleotides.

# Returns
- Vector{Bool}: A boolean vector indicating unique conditions.
"""

function unique_condition(position::Vector{Int64},nucleotide::Vector{Int64})
    
    position_list=Tuple{Int64, Int64}[]
    N=length(position)
    output=zeros(Bool,N)
    for k in 1:N
        if isempty(findall(x->x==(position[k],nucleotide[k]),position_list))
                output[k]=true
                push!(position_list,(position[k],nucleotide[k]))
        end
    end
    
    return output
end

"""
decreasing_condition(freq::Vector{Float64})

Check if the frequencies are in decreasing order.

# Arguments
- `freq`: Vector{Float64}, the frequency vector.

# Returns
- Bool: `true` if frequencies are in decreasing order, `false` otherwise.
"""

function decreasing_condition(freq::Vector{Float64})
    return !isempty(findall(!iszero,freq.>last(freq)))
            
end 


"""
lost_condition(freq::Vector{Float64}; lost_thr = 0.05)

Check if a mutation has been lost based on the frequency vector.

# Arguments
- `freq`: Vector{Float64}, the frequency vector representing the mutation frequencies over time.
- `lost_thr`: Float64, the threshold below which a mutation is considered lost. Default is 0.05.

# Returns
- Bool: `true` if the mutation has been lost, `false` otherwise.
"""



function lost_condition(freq::Vector{Float64};lost_thr = 0.05,)

    for i in 2: length(freq)
        if  freq[i-1] <= lost_thr && freq[i] <= lost_thr # Was lost
            return true
        end
    end
     
    return false
end 

"""
roc_score(result::Matrix{Float64})

Calculate the Receiver Operating Characteristic (ROC) curve metrics including True Positive Rate (TPR), False Positive Rate (FPR), and Positive Predictive Value (PPV).

# Arguments
- `result`: Matrix{Float64}, a matrix containing the model predictions and true labels. The last column should represent the true labels (1 for positive, 0 for negative).
# Returns
- TPR_list: Vector{Float64}, a vector of True Positive Rates for different decision thresholds.
- FPR_list: Vector{Float64}, a vector of False Positive Rates for different decision thresholds.
- PPV: Vector{Float64}, a vector of Positive Predictive Values for different decision thresholds.
"""

function roc_score(result::Matrix{Float64})
    L=size(result,1)
    dt_range=1:1:L #0.1:0.1:1.0
    TPR_list=zeros(length(dt_range))
    FPR_list=zeros(length(dt_range))
    ppv=zeros(length(dt_range))

    for k in 1:length(dt_range)
        
       decision_th=dt_range[k]
        #dt=Int(floor((1-decision_th)*L))
        #dt=L-decision_th
        dt=decision_th
        pp=result[1:dt,:]#predicted positives
    
        tp=sum(pp[:,end].==1) #true positives
        fp=sum(pp[:,end].==0) #false positives
    
        pn=result[dt+1:end,:]#predicted negatives
    
        fn=sum(pn[:,end].==1) #false negatives
        tn=sum(pn[:,end].==0) #true negatives

        TPR=tp/(tp+fn) #true positive rate
        FPR=fp/(fp+tn) #false positive rate
        
        ppv[k]=tp/(tp+fp)
        TPR_list[k]=TPR
        FPR_list[k]=FPR
        
    end    
    return TPR_list,FPR_list,ppv
    
end   

  """
score_fitness(Df::DataFrame)

Calculate a fitness score for each row in the DataFrame based on the difference between Av_fitness_mutant and Av_fitness_wildtype columns. Sorts the DataFrame in descending order based on the fitness score.

# Arguments
- `Df`: DataFrame, a DataFrame containing columns 'Av_fitness_mutant', 'Av_fitness_wildtype', and 'Fixation'.

# Returns
- `order_traj`: Matrix{Float64}, a matrix containing the original row indices, fitness scores, and binary values indicating fixation status sorted in descending order based on fitness scores.
"""

function score_fitness(Df::DataFrame)
    
    order_traj=zeros(size(Df,1),3)

    for k in 1:size(Df,1)  
        order_traj[k,1]=k
        order_traj[k,2]= Df.Av_fitness_mutant[k]-Df.Av_fitness_wildtype[k]
        order_traj[k,3]= Df.Fixation[k]=="fixed" ? 1 : 0  
    end  
    
    order_traj=reverse(sortslices(order_traj,dims=1,by=x->x[2]),dims=1)

    return  order_traj
    
end    

"""
score_simulations(Df::DataFrame)

Calculate a simulation score for each row in the DataFrame based on the difference between Simulations_fixed and Simulations_lost columns. Sorts the DataFrame in descending order based on the simulation score.

# Arguments
- `Df`: DataFrame, a DataFrame containing columns 'Simulations_fixed', 'Simulations_lost', and 'Fixation'.

# Returns
- `order_traj`: Matrix{Float64}, a matrix containing the original row indices, simulation scores, and binary values indicating fixation status sorted in descending order based on simulation scores.
"""

function score_simulations(Df::DataFrame)
    
    order_traj=zeros(size(Df,1),3)

    for k in 1:size(Df,1)  
        order_traj[k,1]=k
        order_traj[k,2]= Df.Simulations_fixed[k]-Df.Simulations_lost[k]
        order_traj[k,3]= Df.Fixation[k]=="fixed" ? 1 : 0  
    end  
    
    order_traj=reverse(sortslices(order_traj,dims=1,by=x->x[2]),dims=1)

    return  order_traj
    
end    



function score_simulations_V2(Df::DataFrame)
    
    order_traj=zeros(size(Df,1),3)

    for k in 1:size(Df,1)  
        order_traj[k,1]=k
        order_traj[k,2]= Df.Simulations_fixed[k]-Df.Simulations_lost[k]
        order_traj[k,3]= Df.Fixation[k]==:fixed ? 1 : 0  
    end  
    
    order_traj=reverse(sortslices(order_traj,dims=1,by=x->x[2]),dims=1)

    return  order_traj
    
end    

"""
score_simulations(Df::DataFrame)

Calculate a simulation score for each row in the DataFrame based on the difference between Simulations_fixed and Simulations_lost columns. Sorts the DataFrame in descending order based on the simulation score.

# Arguments
- `Df`: DataFrame, a DataFrame containing columns 'Simulations_fixed', 'Simulations_lost', and 'Fixation'.

# Returns
- `order_traj`: Matrix{Float64}, a matrix containing the original row indices, simulation scores, and binary values indicating fixation status sorted in descending order based on simulation scores.
"""

function score_selection_coeff(Df::DataFrame)
    
    order_traj=zeros(size(Df,1),3)

    for k in 1:size(Df,1)  
        order_traj[k,1]=k
        order_traj[k,2]= Df.Selection_coefficient[k]
        order_traj[k,3]= Df.Fixation[k]=="fixed" ? 1 : 0  
    end  
    
    order_traj=reverse(sortslices(order_traj,dims=1,by=x->x[2]),dims=1)

    return  order_traj
    
end 


"""
score_fixation_time(Df::DataFrame)

Calculate a fixation time score for each row in the DataFrame based on the 'Mean_fixation_time' column. Sorts the DataFrame in descending order based on the fixation time scores.

# Arguments
- `Df`: DataFrame, a DataFrame containing columns 'Mean_fixation_time' and 'Fixation'.

# Returns
- `order_traj`: Matrix{Float64}, a matrix containing the original row indices, fixation time scores, and binary values indicating fixation status sorted in descending order based on fixation time scores.
"""   
function score_fixation_time(Df::DataFrame)
    
    order_traj=zeros(size(Df,1),3)
  
    i_fixation_time=zeros(size(Df,1))
    
    for i in 1:size(Df,1)
        if !isnan(Df.Mean_fixation_time[i])
            i_fixation_time[i]=1/Df.Mean_fixation_time[i]
        end 
 
    end    
    
    
    
    for k in 1:size(Df,1)  
        order_traj[k,1]=k
        order_traj[k,2]= i_fixation_time[k]#-i_lost_time[k]
        order_traj[k,3]= Df.Fixation[k]=="fixed" ? 1 : 0  
    end  
    
    order_traj=reverse(sortslices(order_traj,dims=1,by=x->x[2]),dims=1)

    return  order_traj
    
end    

"""
score_combined(Df::DataFrame)

Calculate a combined score for each row in the DataFrame based on multiple columns. Sorts the DataFrame in descending order based on the combined scores.

# Arguments
- `Df`: DataFrame, a DataFrame containing columns 'Simulations_fixed', 'Mean_fixation_time', 'Av_fitness_mutant', 'Selection_coefficient', and 'Fixation'.

# Returns
- `order_traj`: Matrix{Float64}, a matrix containing the original row indices and binary values indicating fixation status sorted in descending order based on the combined scores.

"""
function score_combined(Df::DataFrame)
    order_traj=zeros(size(Df,1),2)

    DF_sort=sort(Df, [order(:Simulations_fixed,rev=true),order(:Mean_fixation_time),order(:Av_fitness_mutant,rev=true),order(:Selection_coefficient,rev=true)])

    
    for k in 1:size(DF_sort,1)
        order_traj[k]=k
        order_traj[k,2]= DF_sort.Fixation[k]=="fixed" ? 1 : 0    
    end      
    
    return order_traj
end   

"""
score_frequency(Df::DataFrame)

Calculate a score for each row in the DataFrame based on the 'Frequency' column. Sorts the DataFrame in descending order based on the calculated scores.

# Arguments
- `Df`: DataFrame, a DataFrame containing a 'Frequency' column.

# Returns
- `order_traj`: Matrix{Float64}, a matrix containing the original row indices, frequency values, and binary values indicating fixation status sorted in descending order based on the frequency values.
"""

function score_frequency(Df::DataFrame)
    
    order_traj=zeros(size(Df,1),3)
    frq=last.(parse_string.(Df.Frequency))
    for k in 1:size(Df,1)  
        order_traj[k,1]=k
        order_traj[k,2]=frq[k]
        order_traj[k,3]= Df.Fixation[k]=="fixed" ? 1 : 0  
    end  
    
    order_traj=reverse(sortslices(order_traj,dims=1,by=x->x[2]),dims=1)

    return  order_traj
    
end   




function score_emp_frequency(Df::DataFrame)
    
    order_traj=zeros(size(Df,1),3)
    frq=Df.Empirical_frequency
    for k in 1:size(Df,1)  
        order_traj[k,1]=k
        order_traj[k,2]=frq[k]
        order_traj[k,3]= Df.Fixation[k]=="fixed" ? 1 : 0  
    end  
    
    order_traj=reverse(sortslices(order_traj,dims=1,by=x->x[2]),dims=1)

    return  order_traj
    
end  

function score_emp_frequency_V2(Df::DataFrame)
    
    order_traj=zeros(size(Df,1),3)
    frq=Df.Empirical_frequency
    for k in 1:size(Df,1)  
        order_traj[k,1]=k
        order_traj[k,2]=frq[k]
        order_traj[k,3]= Df.Fixation[k]==:fixed ? 1 : 0  
    end  
    
    order_traj=reverse(sortslices(order_traj,dims=1,by=x->x[2]),dims=1)

    return  order_traj
    
end  





"""
likelihood_model(df::DataFrame)

Calculate the likelihood of a given model based on a DataFrame containing information about fixation events and mutant fitness.

# Arguments
- `df`: DataFrame, a DataFrame containing information about fixation events and mutant fitness. Should include columns like 'Fixation' and 'Av_fitness_mutant'.

# Returns
- `L_model`: Float64, the likelihood of the model based on the provided DataFrame.
"""
function likelihood_model(df::DataFrame)
    L_model=1.0
    for i in 1:size(df,1)
        if df.Fixation[i]=="fixed"
            L_model*=df.Av_fitness_mutant[i]+0.0001
        else
            L_model*=(1-df.Av_fitness_mutant[i])+0.0001
        end
    end
    return L_model
end   

"""
likelihood_model_sim(df::DataFrame)

Calculate the likelihood  based on a DataFrame containing information about simulated fixation events.

# Arguments
- `df`: DataFrame, a DataFrame containing information about simulated fixation events. Should include columns like 'Fixation' and 'Simulations_fixed'.

# Returns
- `L_model`: Float64, the likelihood of the simulation model based on the provided DataFrame.
""" 

function likelihood_model_sim(df::DataFrame)
    L_model=1.0
    for i in 1:size(df,1)
        if df.Fixation[i]=="fixed"
            L_model*=df.Simulations_fixed[i]+0.0001
        else
            L_model*=(1-df.Simulations_fixed[i])+0.0001
        end
    end
    return L_model
end  

"""
likelihood_null(df::DataFrame)

Calculate the likelihood of a null model based on a DataFrame containing information about fixation events and mutant frequencies.

# Arguments
- `df`: DataFrame, a DataFrame containing information about fixation events and mutant frequencies. Should include columns like 'Fixation' and 'Frequency'.

# Returns
- `L_null`: Float64, the likelihood of the null model based on the provided DataFrame.
"""


function likelihood_null(df::DataFrame)
    L_null=1.0
    for i in 1:size(df,1)
        if df.Fixation[i]=="fixed"
            L_null*=last.(parse_string.(df.Frequency))[i]+0.0001
        else
            L_null*=(1-last.(parse_string.(df.Frequency))[i])+0.0001
        end
    end
    
    return L_null
end    



function compute_fixation_metrics(dft_::DataFrame; q::Int = 21, percentage::Float64 = 0.5)
    
    percentage_1=0.5
    percentage_2=0.25

    delta = 0.05
    gamma = 20.0
    sigma = 60

    AUC_vs_f0_3_0 = Float64[]
    AUC_vs_f0_1_0_null = Float64[]

    fix_prob_score_top = Float64[]
    fix_prob_score_bottom = Float64[]
    fix_prob_score_top_25 = Float64[]
    fix_prob_score_bottom_25 = Float64[]
    fix_prob_frequency = Float64[]

    number_of_fix = Int[]
    number_of_lost = Int[]

    true_positives_times_0 = Float64[]
    false_positives_times_0 = Float64[]
    true_negatives_times_0 = Float64[]
    false_negatives_times_0 = Float64[]

    for f0 in 0.1:0.1:0.9
        freqs = last.(parse_string.(dft_.Frequency))
        df = dft_[(freqs .>= f0 - delta) .& (freqs .<= f0 + delta), :]

        df = df[(df.Fixation .== "lost") .| (df.Fixation .== "fixed"), :]

        push!(true_positives_times_0, mean(df[(df.Simulations_fixed .> 0.5) .& (df.Fixation .== "fixed"), :].Time))
        push!(false_positives_times_0, mean(df[(df.Simulations_fixed .< 0.5) .& (df.Fixation .== "fixed"), :].Time))
        push!(true_negatives_times_0, mean(df[(df.Simulations_fixed .< 0.5) .& (df.Fixation .== "lost"), :].Time))
        push!(false_negatives_times_0, mean(df[(df.Simulations_fixed .> 0.5) .& (df.Fixation .== "lost"), :].Time))

        number_of_tr = size(df, 1)
        push!(number_of_fix, size(df[df.Fixation .== "fixed", :], 1))
        push!(number_of_lost, size(df[df.Fixation .== "lost", :], 1))

        predictions_top_50 = Int(round(percentage_1 * number_of_tr; digits = 0))
        predictions_top_25 = Int(round(percentage_2 * number_of_tr; digits = 0))

        sc_3 = score_simulations(df)
        sc_null = score_frequency(df)

        roc_3 = ROC.roc(sc_3[:, 2], sc_3[:, 3], true)
        roc_null = ROC.roc(sc_null[:, 2], sc_null[:, 3], true)

        push!(AUC_vs_f0_3_0, AUC(roc_3))
        push!(AUC_vs_f0_1_0_null, AUC(roc_null))

        push!(fix_prob_score_top, sum(sc_3[1:predictions_top_50, 3]) / predictions_top_50)
        push!(fix_prob_score_bottom, sum(sc_3[predictions_top_50+1:end, 3]) / (number_of_tr - predictions_top_50))
        push!(fix_prob_frequency, sum(sc_3[:, 3]) / number_of_tr)

        push!(fix_prob_score_top_25, sum(sc_3[1:predictions_top_25, 3]) / predictions_top_25)
        push!(fix_prob_score_bottom_25, sum(sc_3[end-predictions_top_25:end, 3]) / (predictions_top_25))
    end

    return (
        AUC_vs_f0_3_0,
        AUC_vs_f0_1_0_null,
        fix_prob_score_top,
        fix_prob_score_bottom,
        fix_prob_score_top_25,
        fix_prob_score_bottom_25,
        fix_prob_frequency
    )
end

function make_auc_fix_plots(auc_sim, auc_null, fix_top_50, fix_bottom_50, fix_top_25, fix_bottom_25,fix_all, label)
    # For plots
    base = 9
    lfs = base # Legend font size
    alfs = base # axis label font size
    tfs = base # title font size
    atfs = base # axis ticks font size
    rl = 0.065 # position of the "A" or "B" label in % 
    ms = 6 # marker size
    ls = 6 # line size
    annfs = 34
    bar_width=0.09
    lw=2
    font2=font("Arial",base)
    ##########
    frequencies=0.1:0.1:0.9
    auc = Plots.bar(frequencies, auc_sim, color=RGB(0.3, 0.45, 0.8), alpha=1.0, label="prediction",bar_width=bar_width)
    Plots.bar!(frequencies, auc_null, color=:lightgray, alpha=1.0, label="null model",bar_width=bar_width)
    plot!(auc, ylim=[0.0, 1.01], frame=false, grid=false, size=(250,450),
          tickfontsize=tfs, legend=(0.35,0.98), legendfontsize=lfs,
          foreground_color_legend=nothing, background_color_legend=nothing,
          title="", xlabel="frequency", ylabel="Fix/loss AUROC",font =font2)

    fix = Plots.plot(frequencies, frequencies, line=:dash, color=:black, label="")
    Plots.plot!(frequencies, fix_top_25, color=RGB(0.3, 0.7, 1.0), label="top 25%",linewidth=lw)

    Plots.plot!(frequencies, fix_top_50, color=RGB(0.3, 0.45, 0.8), label="top 50%",linewidth=lw)
    Plots.plot!(frequencies, fix_bottom_25, color=RGB(0.95, 0.4, 0.4), label="bottom 25%",linewidth=lw)

    Plots.plot!(frequencies, fix_bottom_50, color=RGB(0.6, 0.05, 0.05), label="bottom 50%",linewidth=lw)
    Plots.plot!(frequencies, fix_all, color=:lightgray, label="all mutations",linewidth=lw)
    plot!(fix, xlim=[0.0, 1.01], ylim=[-0.01, 1.01], frame=false, grid=false, size=(250,450),
          tickfontsize=tfs, legend=(0.12,0.98), legendfontsize=lfs,
          foreground_color_legend=nothing, background_color_legend=nothing,
          title="", xlabel="frequency", ylabel="Fixation probability",font =font2)


    return auc, fix
end
    