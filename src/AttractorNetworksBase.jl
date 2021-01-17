module AttractorNetworksBase
using LinearAlgebra
using OrdinaryDiffEq
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


abstract type IOFunction{R} end
Base.Broadcast.broadcastable(g::IOFunction)=Ref(g)
Base.copy(g::IOFunction) = g

struct IOQuad{T} <: IOFunction{T}
    α::T
end
@inline function (gf::IOQuad{T})(x::T) where T
     return gf.α*(log(1.0+exp(x)))^2
end
@inline function ioprime(x::T,gf::IOQuad{T}) where T
    ex= exp(x)
    return gf.α * 2.0ex*log(1. + ex)/(1. + ex)
end
@inline function ioprimeprime(x::T,gf::IOQuad{T}) where T
    ex= exp(x)
    oex= 1. + ex
    return 2.0*gf.α*ex*(log(oex)+ex) / (oex*oex)
end
@inline function ioinv(y::T,gf::IOQuad{T}) where T
    return log(exp( sqrt(y/gf.α) )-1.0)
end

struct IOId{T} <:IOFunction{T}
end
@inline function (gf::IOId{T})(x::T) where T
     return x
end
@inline function ioprime(x::T,gf::IOId{T}) where T
    return one(T)
end
@inline function ioprimeprime(x::T,gf::IOId{T}) where T
    return zero(T)
end
@inline function ioinv(x::T,gf::IOId{T}) where T
    return x
end


# struct IORelu{T} <:IOFunction{T}
# end
#

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
function diagtozero!(M::AbstractMatrix{T}) where T<:Real
    ms = minimum(size(M))
    for i in 1:ms
        @inbounds M[i,i] = zero(T)
    end
    return M
end

"""
        norm_sum_rows!(mat)
Rescales the matrix by row so that the sum of each row is 1.0
"""
function norm_sum_rows!(mat::AbstractMatrix{T}) where T<:Real
    normf=abs.(inv.(sum(mat,dims=2)))
    broadcast!(*,mat,mat,normf)
    return mat
end

 # two populations means Dale!
function make_wmat(ne::I,ni::I,wmeanf::M ; noautapses=true) where
            {I<: Integer,M <: AbstractMatrix{<:Real}}
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

abstract type RecurrentNetwork{R} end

struct BaseNetwork{R} <: RecurrentNetwork{R}
    weights::Matrix{R}
    iofunction::IOFunction{R}
    membrane_taus::Vector{R}
    external_input::Vector{R}
end
struct AttractorNetwork{R} <: RecurrentNetwork{R}
    weights::Matrix{R}
    iofunction::IOFunction{R}
    attractors_u::Matrix{R}
    membrane_taus::Vector{R}
    external_input::Vector{R}
end

n_neurons(rn::RecurrentNetwork) = size(rn.weights,1)
n_attractors(rn::AttractorNetwork) = size(rn.attractors_u,2)
Base.ndims(rn::RecurrentNetwork)=n_neurons(rn)

#Base.Broadcast.broadcastable(g::RecurrentNetwork)=Ref(g)

Base.copy(ntw::BaseNetwork{R}) where R =
    BaseNetwork{R}( (copy(getfield(ntw,n))
        for n in fieldnames(BaseNetwork) )...)
Base.copy(ntw::AttractorNetwork{R}) where R =
    AttractorNetwork{R}( (copy(getfield(ntw,n))
                for n in fieldnames(AttractorNetwork) )...)

@inline function iofunction!(dest::AbstractVector{R},u::AbstractVector{R},
    nt::RecurrentNetwork{R}) where R<:Real
  for i in eachindex(dest)
      dest[i]=nt.iofunction(u[i])
  end
  return dest
end
@inline function ioprime!(dest::Vector{R},u::AbstractVector{R},
    nt::RecurrentNetwork{R}) where R<:Real
  for i in eachindex(dest)
      dest[i]=ioprime(u[i],nt.iofunction)
  end
  return dest
end
@inline function ioprime(u::AbstractVector{R},nt::RecurrentNetwork{R}) where R<:Real
    return ioprime!(similar(u),u,nt)
end
@inline function ioprimeprime!(dest::AbstractVector{R},
    u::AbstractVector{R},nt::RecurrentNetwork{R}) where R<:Real
  for i in eachindex(dest)
      dest[i]=ioprimeprime(u[i],nt.iofunction)
  end
  return dest
end
@inline function ioprimeprime(u::AbstractVector{R},nt::RecurrentNetwork{R}) where R<:Real
    return ioprimeprime!(similar(u),u,nt)
end

@inline function ioinv!(dest::Vector{R},u::AbstractVector{R},
    nt::RecurrentNetwork{R}) where R<:Real
  for i in eachindex(dest)
      dest[i]=ioinv(u[i],nt.iofunction)
  end
  return dest
end
@inline function ioinv(u::AbstractVector{R},nt::RecurrentNetwork{R}) where R<:Real
    return ioinv!(similar(u),u,nt)
end

function normalize_membrane_taus!(v::Vector{R},ntw::RecurrentNetwork{R}) where R
    for (i,τ) in enumerate(ntw.membrane_taus)
        v[i] /= τ
    end
    return v
end

function BaseNetwork(ne::I,ni::I;
        gfun::Union{Nothing,G}=nothing,
        taus::Union{Nothing,V}=nothing,
        external_current::Union{Nothing,V}=nothing,
        W::Union{Nothing,M}=nothing,
        noautapses = true) where
                {I<:Integer, G<:IOFunction,M<:AbstractMatrix,V<:AbstractVector}

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
    gfun = something(gfun, IOQuad(default_values[:gainalpha]))
   return BaseNetwork(W,gfun,taus,h)
end
function AttractorNetwork(bn::BaseNetwork{R},attr::Matrix{R}) where R
    return AttractorNetwork{R}(bn.weights,bn.iofunction,attr,
        bn.membrane_taus,bn.external_input)
end

function velocity!(dest::AbstractVector{R},u::AbstractVector{R},
        gu::AbstractVector{R},rn::RecurrentNetwork{R}) where R<:Real
    copy!(dest, rn.external_input)
    dest .-= u  #  v_out <-  - u +  h
    LinearAlgebra.BLAS.gemv!('N',1.0,rn.weights,gu,1.0,dest) # W*g(u) - u + h
    normalize_membrane_taus!(dest,rn) # ( W*g(u) - u + h) / taus
    return dest
end
velocity(u,rn) = velocity!(similar(u),u,rn.iofunction.(u), rn)



# Jacobian stuff ! Compute and derivatives!
# let's define (and test) the derivatives of the jacobian here!

struct JAlloc{R}
    dgu::Vector{R}
    ddgu::Vector{R}
    inv_taus::Vector{R}
end
function JAlloc(ntw::RN) where {R<:Real,RN<:RecurrentNetwork{R}}
    dgu = similar(ntw.membrane_taus)
    ddgu = similar(ntw.membrane_taus)
    JAlloc{R}(dgu,ddgu,inv.(ntw.membrane_taus))
end

function jacobian!(Jdest::AbstractMatrix{R},
        Wgrad::Union{Nothing,Matrix{R}},ugrad::Union{Nothing,Matrix{R}},
        u::AbstractVector{R},ntw::RecurrentNetwork{R},
        Jallocs::Union{Nothing,JAlloc{R}}=nothing) where R
  Jallocs=something(Jallocs,JAlloc(ntw))
  ioprime!(Jallocs.dgu,u,ntw)
  broadcast!(*,Jdest, ntw.weights, transpose(Jallocs.dgu)) # multiply columnwise
  for i in 1:n_neurons(ntw)
    Jdest[i,i] -= 1.0 #subtract diagonal
  end
  #normalize by taus, rowwise
  broadcast!(/,Jdest,Jdest,ntw.membrane_taus)
  # GRADIENTS ! with respect to  W first
  if !isnothing(Wgrad)
    broadcast!(*,Wgrad,Jallocs.inv_taus,transpose(Jallocs.dgu))
  end
  # now with respect to u
  if !isnothing(ugrad)
    ioprimeprime!(Jallocs.ddgu,u,ntw)
    #broadcast!(/,Jallocs.weights,transpose(dgu),ntw.membrane_taus)
    broadcast!(*,ugrad,Jallocs.inv_taus,
      ntw.weights, transpose(Jallocs.ddgu))
  end
  return Jdest
end

"""
        jacobian(u,rn::RecurrentNetwork)
Jacobian matrix of the
"""
@inline function jacobian(u::AbstractVector{R},rn::RecurrentNetwork{R}) where R
    return jacobian!(similar(rn.weights),nothing,nothing,u,rn)
end


function spectral_abscissa(u,rn::RecurrentNetwork)
    J=jacobian(u,rn)
    return maximum(real.(eigvals(J)))
end

# Dynamics
function run_network(ntw::RecurrentNetwork{R},r_start::Vector{R},
    t_end::Real; verbose::Bool=false,stepsize=0.05) where R<:Real
  u0=ioinv(r_start,ntw)
  iofun_alloc=similar(r_start)
  f(du,u,p,t) = velocity!(du,u,iofunction!(iofun_alloc,u,ntw),ntw)
  prob = ODEProblem(f,u0,(0.,t_end))
  solv = solve(prob,Tsit5();verbose=verbose,saveat=stepsize)
  ret_u = hcat(solv.u...)
  ret_r = ntw.iofunction.(ret_u)
  return solv.t,ret_u,ret_r
end

function run_network_noise(ntw::RecurrentNetwork{R},
    r_start::Vector{R},noiselevel::Real,t_end::Real;
    verbose::Bool=false,stepsize=0.05) where R<:Real
  u0=ioinv(r_start,ntw)
  iofun_alloc=similar(r_start)
  f(du,u,p,t) = velocity!(du,u,iofunction!(iofun_alloc,u,ntw),ntw)
  σ_f(du,u,p,t) = fill!(du,noiselevel)
  prob = SDEProblem(f,σ_f,u0,(0.,t_end))
  solv =  solve(prob,EM();verbose=verbose,dt=stepsize)
  ret_u = hcat(solv.u...)
  ret_r = ntw.iofunction.(ret_u)
  return solv.t,ret_u,ret_r
end

"""
        run_network_to_convergence(rn::RecurrentNetwork,r_start::Vector{<:Real} ;
                t_end=80. , veltol=1E-1)
Runs the network as described in [`run_network_nonoise`](@ref), but stops as soon as
`norm(v) / n < veltol` where `v` is the velocity at time `t`.
If this condition is not satisfied (no convergence to attractor), it runs until `t_end` and prints a warning.
# Arguments
- `rn::RecurrentNetwork`
- `r_start::Vector` : initial conditions
- `t_end::Real`: the maximum time considered
- `veltol::Real` : the norm (divided by num. dimensions) for velocity at convergence
# Outputs
- `u_end`::Vector : the final state at convergence
- `r_end`::Vector : the final state at convergence as rate
"""
function run_network_to_convergence(ntw::RecurrentNetwork{R},
     r_start::Vector{R};t_end::Real=50. , veltol::Real=1E-1) where R
    function  condition(u,t,integrator)
        v = get_du(integrator)
        return norm(v) / length(v) < veltol
    end
    function affect!(integrator)
        savevalues!(integrator)
        return terminate!(integrator)
    end
    u0=ioinv(r_start,ntw)
    cb=DiscreteCallback(condition,affect!)
    ode_solver = Tsit5()
    ioprime_alloc=similar(r_start)
    f(du,u,p,t) = velocity!(du,u,iofunction!(ioprime_alloc,u,ntw),ntw)
    prob = ODEProblem(f,u0,(0.,t_end))
    out = solve(prob,Tsit5();verbose=false,callback=cb)
    u_out = out.u[end]
    t_out = out.t[end]
    if isapprox(t_out,t_end; atol=0.05)
        vel = velocity(u_out,ntw)
        @warn "no convergence after max time $t_end"
        @info "the norm (divided by n) of the velocity is $(norm(vel)/length(vel)) "
    end
    ret_r = ntw.iofunction.(u_out)
    return u_out,ret_r
end

# here I make the AttractorNetwork, the one with attractors

"""
        lognorm_reparametrize(m,std) -> distr::LogNormal
# parameters
  + `m`   sample mean
  + `std` sample std
"""
function lognorm_reparametrize(m,std)
    vm2= (std/m)^2
    μ = log(m / sqrt(1. + vm2))
    σ = sqrt(log( 1. + vm2))
    return LogNormal(μ,σ)
end

function make_attractors(ntot,natt,gainf::IOFunction{R};
        mu_r=5.0,std_r=2.0) where R
    distr = lognorm_reparametrize(mu_r,std_r)
    attr_r = rand(distr,(ntot,natt))
    attr_u = ioinv.(attr_r,gainf)
    return (attr_r,attr_u)
end
make_attractors(ntot,natt,ntw::RecurrentNetwork;mu_r=5.0,std_r=2.0) =
    make_attractors(ntot,natt,ntw.iofunction;mu_r=mu_r,std_r=std_r)


function AttractorNetwork(natt::I,ne::I,ni::I;
        gfun::Union{Nothing,G}=nothing,
        taus::Union{Nothing,V}=nothing,
        external_current::Union{Nothing,V}=nothing,
        W::Union{Nothing,M}=nothing,
        noautapses::Bool = true ,
        mu_attr::Real=5.0,
        std_attr::Real=2.0) where
                {I<:Integer, G<:IOFunction,M<:AbstractMatrix,V<:AbstractVector}
  base=BaseNetwork(ne,ni;gfun=gfun,taus=taus,external_current=external_current,
    W=W,noautapses=noautapses)
  _,attr_u = make_attractors(ne+ni,natt,base;mu_r=mu_attr,std_r=std_attr)
  return AttractorNetwork(base,attr_u)
end

end # module
