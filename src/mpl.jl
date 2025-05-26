

#computation of expression (12) from MPL paper.
#MPL

#To increase robustness and avoid unnatural 
#covariance and flux terms, we assumed that the true underlying allele frequency 
#trajectories were piecewise linear and replaced the sums over time in equation (12) 
#with integrals

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

"""
function interpFreq(p1_0::Array{Float64,1},p2_0::Array{Float64,2},p1_1::Array{Float64,1},p2_1::Array{Float64,2},x::Float64) 

Interpolate single and pairwise allele frequencies between two points, weighted by (1-x) and x, respectively.

# Arguments
- `p1_0::Array{Float64,1}`: Single allele frequencies at time t.
- `p2_0::Array{Float64,2}`: Pairwise allele frequencies at time t.
- `p1_1::Array{Float64,1}`: Single allele frequencies at time t+1.
- `p2_1::Array{Float64,2}`: Pairwise allele frequencies at time t+1.
- `x::Float64`: Interpolation parameter.

# Returns
- Tuple of interpolated single and pairwise allele frequencies.
"""

function interpFreq(p1_0::Array{Float64,1},p2_0::Array{Float64,2},p1_1::Array{Float64,1},p2_1::Array{Float64,2},x::Float64) 
  return (1 - x) .* p1_0 .+ (x .*p1_1), (1 - x) .* p2_0 .+ (x .*p2_1)
end



"""
function updateCovariance(dg::Float64, p1::Array{Float64,1},p2::Array{Float64,2})

Update the summed covariance matrix.

# Arguments
- `dg::Float64`: Time interval.
- `p1::Array{Float64,1}`: Single allele frequencies at time t.
- `p2::Array{Float64,2}`: Pairwise allele frequencies at time t.

# Returns
- Updated covariance matrix.
"""

function updateCovariance(dg::Float64, p1::Array{Float64,1},p2::Array{Float64,2})
   return dg .* (p2 .- p1*p1')
end 


"""
function updateCovarianceIntegrate(dg::Float64,p1_0::Array{Float64,1},p2_0::Array{Float64,2},p1_1::Array{Float64,1},p2_1::Array{Float64,2}) 

Update the summed covariance matrix integrated.

# Arguments
- `dg::Float64`: Time interval.
- `p1_0::Array{Float64,1}`: Single allele frequencies at time t.
- `p2_0::Array{Float64,2}`: Pairwise allele frequencies at time t.
- `p1_1::Array{Float64,1}`: Single allele frequencies at time t+1.
- `p2_1::Array{Float64,2}`: Pairwise allele frequencies at time t+1.

# Returns
- Updated covariance matrix.
"""

function updateCovarianceIntegrate(dg::Float64,p1_0::Array{Float64,1},p2_0::Array{Float64,2},p1_1::Array{Float64,1},p2_1::Array{Float64,2}) 
 return -dg .* ((2 .* p1_0 * p1_0') .+ (2 .* p1_1 * p1_1') .+ (p1_0 * p1_1') .+ (p1_1 * p1_0')) ./ 6 .+  dg .* 0.5 .* (p2_0 .+ p2_1);
end


"""
function updateMu(dg::Float64,mu::Float64,p1::Array{Float64,1}) 

Update the summed mutation vector (flux out minus flux in).

# Arguments
- `dg::Float64`: Time interval.
- `mu::Float64`: Mutation rate.
- `p1::Array{Float64,1}`: Single allele frequencies at time t.

# Returns
- Updated mutation vector.
"""
function updateMu(dg::Float64,mu::Float64,p1::Array{Float64,1}) 
   return -mu*dg .* (p1 .- (1 .- p1))
end

"""
function updateMuIntegrate(dg::Float64,mu::Float64,p1_0::Array{Float64,1},p1_1::Array{Float64,1}) 

Update the summed mutation vector integrated.

# Arguments
- `dg::Float64`: Time interval.
- `mu::Float64`: Mutation rate.
- `p1_0::Array{Float64,1}`: Single allele frequencies at time t.
- `p1_1::Array{Float64,1}`: Single allele frequencies at time t+1.

# Returns
- Updated mutation vector.
"""


function updateMuIntegrate(dg::Float64,mu::Float64,p1_0::Array{Float64,1},p1_1::Array{Float64,1}) 
   return -mu*dg .* (1 .- (p1_0 .+ p1_1))
end




"""
function processStandard(nVec::Vector{Vector{Int64}},sVec::Vector{Matrix{Float64}},times::Array{Float64,1},mu::Float64,L::Int64,q::Int64,gamma::Float64)

Compute selection coefficients from sequences (time series) using MPL method.

# Arguments
- `nVec::Vector{Vector{Int64}}`: Vector of sequence counts.
- `sVec::Vector{Matrix{Float64}}`: Vector of variants alignments.
- `times::Array{Float64,1}`: Array of time points.
- `mu::Float64`: Mutation rate.
- `L::Int64`: Genome length.
- `q::Int64`: Number of states.
- `gamma::Float64`: Regularization parameter.

# Returns
- Inferred selection coefficients.

"""

function processStandard(nVec::Vector{Vector{Int64}},sVec::Vector{Matrix{Float64}},times::Array{Float64,1},mu::Float64,L::Int64,q::Int64,gamma::Float64)
  
  #Set initial allele frequency and covariance then loop
     (lastp1,lastp2)=comp_freqs(sVec[1],Int.(nVec[1]),q)
      dx=zeros(q*L)
      dx = dx .- lastp1
     
      totalCov=zeros(q*L,q*L)
      totalMu=zeros(q*L)
      p1=zeros(q*L)
      p2=zeros(q*L,q*L)

    # Iterate through sets of sequences collected at all time points,
    # computing allele frequencies, interpolating, and
    # adding contributions to selection coefficient estimates 
    for  k in 2:length(times)
      (p1,p2)=comp_freqs(sVec[k],Int.(nVec[k]),q)
      totalCov .+= updateCovarianceIntegrate(times[k]-times[k-1],lastp1,lastp2,p1,p2) 
      totalMu .+= updateMuIntegrate(times[k] - times[k-1],mu,lastp1,p1)
        
      if k== length(times)
        dx .+= p1
      else
        lastp1 .= p1
        lastp2 .= p2;    
      end          
    end
    
  # Gather dx and totalMu terms
    dx .+= totalMu
    
    
  return inv(totalCov .+ gamma*Matrix(I,q*L , q*L))*dx
    
end  








"""
function Cintegrate(seqs::Dict{Vector{Float64},Vector{Float64}},times::Array{Float64,1},L::Int64,q::Int64,gamma::Float64)

Compute the integrated covariance matrix from smoothed sequences in the dictionary `seqs`.

# Arguments
- `seqs::Dict{Vector{Float64}, Vector{Float64}}`: Dictionary of smoothed sequences. For each sequence correspond a vector of its frequency along time.
- `times::Array{Float64,1}`: Array of time points.
- `L::Int64`: Genome length.
- `q::Int64`: Number of states.
- `gamma::Float64`: Regularization parameter.

# Returns
- Integrated covariance matrix.
"""

function Cintegrate(seqs::Dict{Vector{Float64},Vector{Float64}},times::Array{Float64,1},L::Int64,q::Int64,gamma::Float64)
  
  #Set initial allele frequency and covariance then loop
  (lastp1,lastp2)=compute_freqs(seqs,Int64(times[1]),q);
  dx=zeros(q*L)
  dx = dx .- lastp1
     
  totalCov=zeros(q*L,q*L)
  #totalMu=zeros(q*L)
  p1=zeros(q*L)
  p2=zeros(q*L,q*L) 


  # Iterate through sets of sequences collected at all time points,
  # computing allele frequencies, interpolating, and
  # adding contributions to selection coefficient estimates 
  for  k in 2:length(times)
    (p1,p2)=compute_freqs(seqs,Int64(times[k]),q)
    totalCov .+= updateCovarianceIntegrate(times[k]-times[k-1],lastp1,lastp2,p1,p2) 
    if k== length(times)
      dx .+= p1
    else
      lastp1 .= p1
      lastp2 .= p2;    
    end          
  end
 
  return totalCov .+ gamma*Matrix(I,q*L , q*L)
    
end





"""
function Cintegrate(seqs::Matrix{Float64},counts::Vector{Any},times::Array{Float64,1},L::Int64,q::Int64,gamma::Float64)

Compute the integrated covariance matrix from smoothed sequences in seqs::Matrix{Float64},counts::Vector{Any}.

# Arguments
- `seqs::Matrix{Float64}`: sequence configurations.
- `counts::Vector{Any}`: sequence counts.
- `times::Array{Float64,1}`: Array of time points.
- `L::Int64`: Genome length.
- `q::Int64`: Number of states.
- `gamma::Float64`: Regularization parameter.

# Returns
- Integrated covariance matrix.
"""



function Cintegrate(seqs::Matrix{Float64},counts::Vector{Any},times::Array{Float64,1},L::Int64,q::Int64,gamma::Float64)
  
  #Set initial allele frequency and covariance then loop
    (lastp1,lastp2)=comp_freqs(seqs,counts[1],q)
    dx=zeros(q*L)
    dx = dx .- lastp1
     
  totalCov=zeros(q*L,q*L)
  p1=zeros(q*L)
  p2=zeros(q*L,q*L) 

  # Iterate through sets of sequences collected at all time points,
  # computing allele frequencies, interpolating, and
  # adding contributions to selection coefficient estimates 
  for  k in 2:length(times)
    (p1,p2)=comp_freqs(seqs,counts[k],q)    
    totalCov .+= updateCovarianceIntegrate(times[k]-times[k-1],lastp1,lastp2,p1,p2) 
    if k== length(times)
      dx .+= p1
    else
      lastp1 .= p1
      lastp2 .= p2;    
    end          
  end
 
  return totalCov .+ gamma*Matrix(I,q*L , q*L)
   
end  



"""
function processStandard(seqs::Dict{Vector{Float64},Vector{Float64}},times::Array{Float64,1},mu::Float64,L::Int64,q::Int64,gamma::Float64)

Compute selection coefficients from  dictionary `seqs` using MPL method.

# Arguments
- `seqs::Dict{Vector{Float64}, Vector{Float64}}`: Dictionary containing smoothed sequences. For each sequence correspond a vector of its frequency along time.
- `times::Array{Float64, 1}`: Array of time points.
- `mu::Float64`: Mutation rate.
- `L::Int64`: Genome length.
- `q::Int64`: Number of states.
- `gamma::Float64`: Regularization parameter.

# Returns
- Inferred selection coefficients.
"""
function processStandard(seqs::Dict{Vector{Float64},Vector{Float64}},times::Array{Float64,1},mu::Float64,L::Int64,q::Int64,gamma::Float64)
  
    #Set initial allele frequency and covariance then loop
  (lastp1,lastp2)=compute_freqs(seqs,Int64(times[1]),q);
  dx=zeros(q*L)
  dx = dx .- lastp1
     
  totalCov=zeros(q*L,q*L)
  totalMu=zeros(q*L)
  p1=zeros(q*L)
  p2=zeros(q*L,q*L)


  # Iterate through sets of sequences collected at all time points,
  # computing allele frequencies, interpolating, and
  # adding contributions to selection coefficient estimates 
  for  k in 2:length(times)
    (p1,p2)=compute_freqs(seqs,Int64(times[k]),q)
    totalCov .+= updateCovarianceIntegrate(times[k]-times[k-1],lastp1,lastp2,p1,p2) 
    totalMu .+= updateMuIntegrate(times[k] - times[k-1],mu,lastp1,p1)
        
    if k== length(times)
      dx .+= p1
    else
      lastp1 .= p1
      lastp2 .= p2;    
    end     
        
  end
    
  # Gather dx and totalMu terms
  dx .+= totalMu
    
    
  return inv(totalCov .+ gamma*Matrix(I,q*L , q*L))*dx
    
end  



"""
function processStandard(seqs::Matrix{Float64},counts::Vector{Any},times::Array{Float64,1},mu::Float64,L::Int64,q::Int64,gamma::Float64)

Compute selection coefficients from  dictionary `seqs` using MPL method.

# Arguments
- `seqs::Matrix{Float64}`: sequence configurations.
- `counts::Vector{Any}`: sequence counts.
- `times::Array{Float64, 1}`: Array of time points.
- `mu::Float64`: Mutation rate.
- `L::Int64`: Genome length.
- `q::Int64`: Number of states.
- `gamma::Float64`: Regularization parameter.

# Returns
- Inferred selection coefficients.
"""

function processStandard(seqs::Matrix{Float64},counts::Vector{Any},times::Array{Float64,1},mu::Float64,L::Int64,q::Int64,gamma::Float64)
  
  #Set initial allele frequency and covariance then loop
     (lastp1,lastp2)=comp_freqs(seqs,counts[1],q)
      dx=zeros(q*L)
      dx = dx .- lastp1
     
      totalCov=zeros(q*L,q*L)
      totalMu=zeros(q*L)
      p1=zeros(q*L)
      p2=zeros(q*L,q*L)

    # Iterate through sets of sequences collected at all time points,
    # computing allele frequencies, interpolating, and
    # adding contributions to selection coefficient estimates 
    for  k in 2:length(times)
      (p1,p2)=comp_freqs(seqs,counts[k],q)
      totalCov .+= updateCovarianceIntegrate(times[k]-times[k-1],lastp1,lastp2,p1,p2) 
      totalMu .+= updateMuIntegrate(times[k] - times[k-1],mu,lastp1,p1)
        
      if k== length(times)
        dx .+= p1
      else
        lastp1 .= p1
        lastp2 .= p2;    
      end          
    end
    
  # Gather dx and totalMu terms
    dx .+= totalMu
    
    
  return inv(totalCov .+ gamma*Matrix(I,q*L , q*L))*dx
    
end  

"""
function diagonal_matrix(A::Matrix{Float64})

Compute selection coefficients from  dictionary `seqs` using MPL method.

# Arguments
- `A::Matrix{Float64}`: target matrix

# Returns
-  Matrix A with zero offdiagonal elemetns.
"""


function diagonal_matrix(A::Matrix{Float64})
    n, m = size(A)
    B = zeros(n, m) # Initialize a matrix of zeros with the same size as A
    for i in 1:min(n, m)
        B[i, i] = A[i, i] # Copy diagonal elements from A to B
    end
    return B
end




"""
function processStandardNoCorr(seqs::Matrix{Float64},counts::Vector{Any},times::Array{Float64,1},mu::Float64,L::Int64,q::Int64,gamma::Float64)

Compute selection coefficients from  dictionary `seqs` using SL method.

# Arguments
- `seqs::Matrix{Float64}`: sequence configurations.
- `counts::Vector{Any}`: sequence counts.
- `times::Array{Float64, 1}`: Array of time points.
- `mu::Float64`: Mutation rate.
- `L::Int64`: Genome length.
- `q::Int64`: Number of states.
- `gamma::Float64`: Regularization parameter.

# Returns
- Inferred selection coefficients.
"""

function processStandardNoCorr(seqs::Matrix{Float64},counts::Vector{Any},times::Array{Float64,1},mu::Float64,L::Int64,q::Int64,gamma::Float64)

  #Set initial allele frequency and covariance then loop
     (lastp1,lastp2)=comp_freqs(seqs,counts[1],q)
      dx=zeros(q*L)
      dx = dx .- lastp1

      totalCov=zeros(q*L,q*L)
      totalMu=zeros(q*L)
      p1=zeros(q*L)
      p2=zeros(q*L,q*L)

    # Iterate through sets of sequences collected at all time points,
    # computing allele frequencies, interpolating, and
    # adding contributions to selection coefficient estimates 
    for  k in 2:length(times)
      (p1,p2)=comp_freqs(seqs,counts[k],q)
      totalCov .+= updateCovarianceIntegrate(times[k]-times[k-1],lastp1,lastp2,p1,p2)
      totalMu .+= updateMuIntegrate(times[k] - times[k-1],mu,lastp1,p1)

      if k== length(times)
        dx .+= p1
      else
        lastp1 .= p1
        lastp2 .= p2;
      end
    end

  # Gather dx and totalMu terms
    dx .+= totalMu

    totalCov=diagonal_matrix(totalCov)

  return inv(totalCov .+ gamma*Matrix(I,q*L , q*L))*dx

end

function processStandardNoCorr(seqs::Dict{Vector{Float64},Vector{Float64}},times::Array{Float64,1},mu::Float64,L::Int64,q::Int64,gamma::Float64)
  
    #Set initial allele frequency and covariance then loop
  (lastp1,lastp2)=compute_freqs(seqs,Int64(times[1]),q);
  dx=zeros(q*L)
  dx = dx .- lastp1
     
  totalCov=zeros(q*L,q*L)
  totalMu=zeros(q*L)
  p1=zeros(q*L)
  p2=zeros(q*L,q*L)


  # Iterate through sets of sequences collected at all time points,
  # computing allele frequencies, interpolating, and
  # adding contributions to selection coefficient estimates 
  for  k in 2:length(times)
    (p1,p2)=compute_freqs(seqs,Int64(times[k]),q)
    totalCov .+= updateCovarianceIntegrate(times[k]-times[k-1],lastp1,lastp2,p1,p2) 
    totalMu .+= updateMuIntegrate(times[k] - times[k-1],mu,lastp1,p1)
        
    if k== length(times)
      dx .+= p1
    else
      lastp1 .= p1
      lastp2 .= p2;    
    end     
        
  end
    
  # Gather dx and totalMu terms
  dx .+= totalMu
    
        totalCov=diagonal_matrix(totalCov)

  return inv(totalCov .+ gamma*Matrix(I,q*L , q*L))*dx
    
end 

"""
function normalize_sc(h::Array{Float64,1},ref_seq::Array{Float64,1},q::Int64,L::Int64)

Normalize the inferred selection coefficients such that the wild-type (WT) allele has a selection coefficient of zero.

# Arguments
- `selec_coeff::Array{Float64,1}`: Array of inferred selection coefficients.
- `q::Int64`: Number of states.
- `L::Int64`: Genome length.

# Returns
- Normalized selection coefficients
"""

function normalize_sc(h::Array{Float64,1},ref_seq::Array{Float64,1},q::Int64,L::Int64)
    selec_coeff=deepcopy(h)
    for i in 1:L
        for j in 1:q
      selec_coeff[(i-1)*q+j]-=h[(i-1)*q+Int.(ref_seq[i])]      
        end
    end     
    return selec_coeff
end 



