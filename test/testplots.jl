push!(LOAD_PATH, abspath(@__DIR__,".."))
using Plots
using AttractorNetworksBase ; const A = AttractorNetworksBase
using Calculus
using BenchmarkTools, Cthulhu
using Random
using Test
using Statistics,LinearAlgebra
Random.seed!(0)

using Plots ; theme(:dark)
function plotvs(x::AbstractArray{<:Real},y::AbstractArray{<:Real})
  x,y=x[:],y[:]
  @info """
  The max differences between the two are $(extrema(x .-y ))
  """
  plt=plot()
  scatter!(plt,x,y;leg=false,ratio=1,color=:white)
  lm=xlims()
  plot!(plt,identity,range(lm...;length=3);linestyle=:dash,color=:yellow)
  return plt
end
function test_Jgradient(utest::Vector{R},ntw::A.BaseNetwork) where R
    Jalloc = A.JAlloc(ntw)
    gradW=similar(ntw.weights)
    gradu=similar(ntw.weights)
    J=similar(gradW)
    # analytic gradient
    A.jacobian!(J,gradW,gradu,utest,ntw,Jalloc)
    # numerical gradient weights
    grad_numW = similar(gradW)
    ntw2=copy(ntw)
    for ij in eachindex(ntw2.weights)
        function gradfun(w)
            ntw2.weights[ij] = w
            A.jacobian!(J,nothing,nothing,utest,ntw2,Jalloc)
            return J[ij]
        end
        grad_numW[ij] = Calculus.gradient(gradfun,ntw2.weights[ij])
    end
    # numerical gradient u
    grad_numu = similar(gradu)
    for lj in CartesianIndices(gradu)
        (j,l)=Tuple(lj)
        function gradfun(u)
            u2 = copy(utest)
            u2[l]=u
            A.jacobian!(J,nothing,nothing,u2,ntw,Jalloc)
            return J[lj]
        end
        grad_numu[lj] = Calculus.gradient(gradfun,utest[l])
    end
    return gradW,grad_numW,gradu,grad_numu
end


## linear stable dynamcs goes to zero
ne,ni = 59,33
ntot = ne+ni
ntw = A.BaseNetwork(ne,ni ; gfun=A.IOId{Float64}())
ntw.weights .= 0.3.*randn(ntot)
B = ntw.weights - I
ntw.external_input .= 0.0
ntw.membrane_taus .= 1.0
r_start = 30.0.*randn(ntot)

# check velocity
A.velocity(r_start,ntw)


u_end,r_end=A.run_network_to_convergence(ntw,r_start;veltol=1E-5)


vel_test=similar(u_end)
A.velocity!(vel_test,u_end,ntw.iofunction.(u_end),ntw)


t,rdyn,_=A.run_network(ntw,r_end,4.0)

plot(t,rdyn[4,:];leg=false)




##
fill!(ntw.weights,0.0)
u_start = randn(A.ndims(ntw))
r_start = ntw.iofunction.(u_start)

u_end,r_end=A.run_network_to_convergence(ntw,r_start)
@test all(isapprox.(u_end,ntw.external_input;atol=0.05))

ne,ni = 20,23
ntot = ne+ni
ntw = A.BaseNetwork(ne,ni ; gfun=A.IOQuad(0.02))
u_start = randn(A.ndims(ntw))
r_start = ntw.iofunction.(u_start)
u_end,r_end=A.run_network_to_convergence(ntw,r_start)
t,ut,rt=A.run_network(ntw,r_end,1.0)
@test all(isapprox.(u_end,ut[:,end];atol=0.05))

##



##
Wtest = A.make_wmat(80,20,A.WDalian)
heatmap(Wtest ; ratio=1)

##
pe,pi = A.populations(80,20)
mah = rand(100,100)

testee = pe*pe'
testie = pi*pe'
all( view(testee,pe,pe))
all( view(testie,pi,pe))

view(mah,pe*pe')

heatmap(testie ; ratio=1)
heatmap(pe*pi' ; ratio = 1)
heatmap(pi*pe' ; ratio = 1)


mah[pe,pe] .- mah[1:80,1:80]

##
ne,ni = 34,44
wmeanf = [ 0.333  -1.123
            12.122 -0.06432]
W = A.make_wmat(ne,ni,wmeanf;noautapses=true)

pe,pi = A.populations(ne,ni)

sum(W[pe,pe]; dims=2)
#
# whatevs = A.run_network(10.0 .* rand(ntot), 0.5,ntw)
#
# using Plots; theme(:dark)
# _ = let t=whatevs.t,
#   u12 = [ uu[1:2] for uu in whatevs.u]
#   u12=hcat(u12...) |> permutedims
#   plot(t,u12;leg=false,linewidth=4)
# end

##

ne,ni = 4,3
ntot = ne+ni
ntw = A.BaseNetwork(ne,ni ; gfun=A.IOQuad(0.02))

utest = randn(ntot)

grad_ana,grad_num = test_Jgradient_weights(utest,ntw)
grad_ana,grad_num = test_Jgradient_u(utest,ntw)

grad_ana ./ grad_num

grad_num

grad_ana
