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
    gf.α * 2.0ex*log(1. + ex)/(1. + ex)
end
@inline function ddg(x,gf::GFQuad)
    ex= exp(x)
    oex= 1. + ex
    2.0*gf.α*ex*(log(oex)+ex) / (oex*oex)
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

n_neurons(rn::RecurrentNetwork) = size(rm.weights,1)
function Base.copy(rn::RecurrentNetwork)
   return RecurrentNetwork( copy(rm.weights), rn.gain_function,
        copy(rn.membrane_taus) , copy(rn.external_input) )
end

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

function velocity!(v_out,u,rn::RecurrentNetwork)
    gu = rn.gain_function.(u)
    copy!(v_out, rn.external_input)
    v_out .-= u  #  v_out <-  - u +  h
    LinearAlgebra.BLAS.gemv!('N',1.0,rn.weights,gu,1.0,v_out) # W*g(v) - v + h
    return v_out ./= rn.membrane_taus # ( W*g(v) - v + h) / taus
end
velocity(u,rn) = velocity!(similar(u),u,rn)

"""
        jacobian!(J,u,rn::RecurrentNetwork)
Writes the Jacobian matrix of the system dynamics into J
It might be useful to re-normalize the Jacobian by the mean of the time constants.
"""
function jacobian!(J,u,rn::RecurrentNetwork)
    n = size(J,1)
    _dg =  dg.(u,rn.gain_function)
    broadcast!(*,J, rn.weights, Transpose(_dg)) # multiply columnwise
    @simd for i in 1:n
        @inbounds J[i,i] -= 1.0 #subtract diagonal
    end
    #normalize by taus, rowwise
    return broadcast!(/,J,J,rn.membrane_taus)
end

jacobian(u,rn) = jacobian!(similar(rn.weights),u,rn)

function spectral_abscissa(u,rn::RecurrentNetwork)
    J=jacobian(u,rn)
    return maximum(real.(eigvals(J)))
end

#=

These should go somewhere else, to avoid the heavy dependency with
DifferentialEquations !!!

"""
    run_network(x0,t_max,rn::RecurrentNetwork; rungekutta=false,verbose=false)

"""
function run_network(x0::AbstractVector,t_max,rn::RecurrentNetwork;
            verbose::Bool=false)
    ode_solver = Tsit5()
    f(du,u,p,t) = velocity!(du,u,rn)
    prob = ODEProblem(f,x0,(0.,t_max))
    return solve(prob,ode_solver;verbose=verbose)
end


"""
    run_network_to_convergence(u0, rn::RecurrentNetwork ;
            t_max=50. , veltol::Float64=1E-4)

# inputs
  + `u0` : starting point, in terms of membrane potential
  + `rn` : network
  + `t_max` : maximum time considered (in seconds)
  + `veltol` : tolerance for small velocity
Runs the network stopping when the norm of the velocity is below
the tolerance level (stationary point reached)
"""
function run_network_to_convergence(u0, rn::RecurrentNetwork ;
        t_max=50. , veltol::Float64=1E-4)
    n=length(u0) |> Float64
    function  condition(u,t,integrator)
        v = get_du(integrator)
        return norm(v) / n < veltol
    end
    function affect!(integrator)
        savevalues!(integrator)
        return terminate!(integrator)
    end
    cb=DiscreteCallback(condition,affect!)
    ode_solver = Tsit5()
    f(du,u,p,t) = velocity!(du,u,rn)
    prob = ODEProblem(f,u0,(0.,t_max))
    out = solve(prob,Tsit5();verbose=false,callback=cb)
    u_out = out.u[end]
    t_out = out.t[end]
    if isapprox(t_out,t_max; atol=0.05)
        @warn "no convergence after max time $t_max"
        vel = velocity(u_out,rn)
        @info "the norm (divided by n) of the velocity is $(norm(vel)/n) "
    end
    return u_out
end
=#

end # module
