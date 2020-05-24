module AttractorNetworksBase
using LinearAlgebra
using Distributions, Random
# numerical defaults for the constructors
const default_values = Dict(
    :muw=>3.0,
    :Wmeanf => [ 2.5  -1.3
                 2.4   -1.  ],
    :Wmeanf_alt => [ 1.08  -0.8
                       0.8   -2.25 ],
    :external_current => 7.0,
    :tau_ei => (0.02,0.01),
    :gainalpha => 0.02 )


abstract type GainFunction end
Base.Broadcast.broadcastable(g::GainFunction)=Ref(g)
Base.copy(g::GainFunction) = g
struct GFQuad{T} <: GainFunction where T
    α::T
end
struct GFId <:GainFunction
end
struct GFSRelu{T} <:GainFunction where T
end

g!(x_dest::AbstractVector,x,gf::GainFunction) = broadcast!(gf, x_dest,x)
dg!(x_dest::AbstractVector,x,gf::GainFunction) = broadcast!(_x->dg(_x,gf), x_dest,x)
ddg!(x_dest::AbstractVector,x,gf::GainFunction) = broadcast!(_x->ddg(_x,gf), x_dest,x)
ig!(x_dest::AbstractVector,x,gf::GainFunction) = broadcast!(_x->ig(_x,gf), x_dest,x)

(gf::GFQuad)(x) =  gf.α * ( log(1. + exp(x)) )^2
ig(y,gf::GFQuad) = log( exp( sqrt(y/gf.α) ) - 1.0 )

@inline function dg(x,gf::GFQuad)
    ex= exp(x)
    return gf.α * 2.0ex*log(1. + ex)/(1. + ex)
end
@inline function ddg(x,gf::GFQuad)
    ex= exp(x)
    oex= 1. + ex
    return 2.0*gf.α*ex*(log(oex)+ex) / (oex*oex)
end

(gf::GFId)(x) = x
ig(y,gf::GFId) = y
dg(x,gf::GFId) = 1.0
ddg(x,gf::GFId) = 0.0

"""
        populations(ne::I,ni::I) where I<:Integer -> (pe::BitArray,pi::BitArray)

For `ne` excitatory an `ni` inhibitory neurons, returns the positions of each in the form of BitArrays. The convention is that E neurons come first.  The indexes of the connections in `W` *from* poulation `pa` *to* population `pb` can then be selected as: `view(W,pb,pa)`
"""
function populations(ne::I,ni::I) where I<:Integer
    ntot = ne+ni
    pe = BitArray(falses(ntot))
    pi = copy(pe)
    pe[1:ne] .= true
    pi[ne+1:end] .= true
    return (pe,pi)
end
"""
        populations(ne::I,ni::I, ne_aux::I,ni_aux::I) where I<:Integer
            -> ((pe,pi,pe_pri,pe_aux,pi_pri,pi_aux)

Like `populations(ne,ni)`, but now E and I populations can have a subset of auxiliary neurons, with `ne = ne_pri+ne_aux`. Returns the indexing for each population in the form of BitArrays. The convetion is that excitatory neurons go first, and principal neurons go first.
"""
function populations(ne::I,ni::I, ne_aux::I,ni_aux::I) where I<:Integer
    ne_pri = ne - ne_aux
    ni_pri = ni - ni_aux
    @assert (ne_pri >= 0 ) && (ni_pri >= 0)
    ntot = ne+ni
    pe,pi = populations(ne,ni)
    pe_pri,pe_aux,pi_pri,pi_aux = (BitArray(falses(ntot)) for _ in 1:4)
    pe_pri[1:ne_pri] .= true
    pe_aux[ne_pri+1:ne_pri+ne_aux] .= true
    pi_pri[ne+1:ne+ni_pri] .= true
    pi_aux[ne+ni_pri+1:end] .= true
    return (pe,pi,pe_pri,pe_aux,pi_pri,pi_aux)
end

"""
        diagtozero!(M::AbstractMatrix{T}) where T
Replaces the matrix diagonal with zeros
"""
function diagtozero!(M::AbstractMatrix{T}) where T
    ms = minimum(size(M))
    for i in 1:ms
        @inbounds M[i,i] = zero(T)
    end
    return nothing
end

"""
        norm_sum_rows!(mat)
Rescales the matrix by row so that the sum of each row is 1.0
"""
function norm_sum_rows!(mat)
    normf=abs.(inv.(sum(mat,dims=2)))
    broadcast!(*,mat,mat,normf)
    return nothing
end

 # two populations means Dale!
function make_wmat(ne::I,ni::I,wmeanf::M ; noautapses=true) where
            {I<: Integer,M <: AbstractMatrix}
    nw = size(wmeanf,1)
    @assert (nw == 2) || (nw==ne+ni)
    if nw == ne+ni #nothing to do
        return wmeanf
    end
    # else, expand the 2D
    Wμ = wmeanf
    d = Exponential(default_values[:muw])
    W=rand(d,ne+ni,ne+ni)
    noautapses && diagtozero!(W)
    # now normalize by block
    pe,pi = populations(ne,ni)
    let wee =  view(W,pe,pe)
        norm_sum_rows!(wee)
        wee .*= Wμ[1,1]
    end
    let wii =  view(W,pi,pi)
        norm_sum_rows!(wii)
        wii .*= Wμ[2,2]
    end
    let wei =  view(W,pe,pi) # inhibitory to excitatory
        norm_sum_rows!(wei)
        wei .*= Wμ[1,2]
    end
    let wie =  view(W,pi,pe) # excitatory to inhibitory
        norm_sum_rows!(wie)
        wie .*= Wμ[2,1]
    end
    return W
end

struct RecurrentNetwork{M,G,V}
    weights::M
    gain_function::G
    membrane_taus::V
    external_input::V
end

Base.copy(ntw::RecurrentNetwork) = RecurrentNetwork( (copy(getfield(ntw,n))
                for n in fieldnames(RecurrentNetwork) )...)

n_neurons(rn::RecurrentNetwork) = size(rn.weights,1)


function RecurrentNetwork(ne::I,ni::I;
        gfun::Union{Nothing,G}=nothing,
        taus::Union{Nothing,V}=nothing,
        external_current::Union{Nothing,V}=nothing,
        W::Union{Nothing,M}=nothing,
        noautapses = true) where
                {I<:Integer, G<:GainFunction,M<:AbstractMatrix,V<:AbstractVector}

   W = if isnothing(W)
       make_wmat(ne,ni,default_values[:Wmeanf];noautapses=noautapses)
   # if full D , just use it
   elseif size(W,1)==ne+ni
       W
   # if 2D use as mean field
   elseif (size(W,1) == 2 ) && (size(W,2) == 2 )
       make_wmat(ne,ni,W;noautapses=noautapses)
   else
       @warn "something might be wrong with the weight matrix"
       W
   end
   # now the taus
   taus =
       if isnothing(taus)
           τe,τi =default_values[:tau_ei]
           vcat(fill(τe,ne),fill(τi,ni))
       elseif length(taus)==2
           τe,τi =taus
           vcat(fill(τe,ne),fill(τi,ni))
       else
           taus
    end
    # currents
    h  = something(external_current,
            fill(default_values[:external_current],ne+ni) )
    # gain
    gfun = something(gfun, GFQuad(default_values[:gainalpha]))
   return RecurrentNetwork(W,gfun,taus,h)
end

function velocity!(v_out,u,gu,rn::RecurrentNetwork)
    copy!(v_out, rn.external_input)
    v_out .-= u  #  v_out <-  - u +  h
    LinearAlgebra.BLAS.gemv!('N',1.0,rn.weights,gu,1.0,v_out) # W*g(v) - v + h
    return v_out ./= rn.membrane_taus # ( W*g(v) - v + h) / taus
end
velocity(u,rn) = velocity!(similar(u),u,rn.gain_function.(u), rn)



# Jacobian stuff ! Compute and derivatives!
# let's define (and test) the derivatives of the jacobian here!

struct JGradPars{M,V}
    weights::M
    u::M
    inv_taus::V
    ddgu_alloc::V
    function JGradPars(ntw::RecurrentNetwork)
        w = similar(ntw.weights)
        u = similar(ntw.weights)
        v = similar(ntw.membrane_taus)
        ddgu_alloc = similar(v)
        new{typeof(w),typeof(v)}(w,u,inv.(ntw.membrane_taus),ddgu_alloc)
    end
end

function _jacobian!(J,gradpars::Union{Nothing,JGradPars},
            u,dgu,ntw::RecurrentNetwork)
    broadcast!(*,J, ntw.weights, transpose(dgu)) # multiply columnwise
    @simd for i in 1:size(J,1)
        @inbounds J[i,i] -= 1.0 #subtract diagonal
    end
    #normalize by taus, rowwise
    broadcast!(/,J,J,ntw.membrane_taus)
    isnothing(gradpars) && return J
    # GRADIENTS !  W first
    gradpars.weights .= gradpars.inv_taus * transpose(dgu)
    #broadcast!(/,gradpars.weights,transpose(dgu),ntw.membrane_taus)
    # now u
    ddg!(gradpars.ddgu_alloc,u,ntw.gain_function)
    broadcast!(*,gradpars.u,gradpars.inv_taus,
        ntw.weights, transpose(gradpars.ddgu_alloc))
    return J
end

"""
        jacobian(u,rn::RecurrentNetwork)
Jacobian matrix of the
"""
@inline function jacobian(u,rn::RecurrentNetwork)
    J = similar(rn.weights)
    dgu = dg.(u,rn.gain_function)
    return _jacobian!(J,nothing,u,dgu,rn::RecurrentNetwork)
end




function spectral_abscissa(u,rn::RecurrentNetwork)
    J=jacobian(u,rn)
    return maximum(real.(eigvals(J)))
end

end # module
