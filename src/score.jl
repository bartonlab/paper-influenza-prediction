

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

    