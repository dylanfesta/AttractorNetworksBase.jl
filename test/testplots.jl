using Pkg
Pkg.activate(joinpath(@__DIR__(),".."))
using Plots
using AttractorNetworksBase ; const A = AttractorNetworksBase
using Calculus
using BenchmarkTools, Cthulhu
using Test

##

function test_Jgradient_weights(utest,ntw::A.RecurrentNetwork)
    J=A.jacobian(utest,ntw)
    gpars = A.JGradPars(ntw)
    # analytic gradient
    dgu = A.dg.(utest,ntw.gain_function)
    A._jacobian!(J,gpars,utest,dgu,ntw)
    grad_ana = vec(gpars.weights)
    # numerical gradient
    ntw2 = copy(ntw)
    grad_num = similar(grad_ana)
    for ij in eachindex(ntw2.weights)
        function miniobj(w)
            ntw2.weights[ij] = w
            return A.jacobian(utest,ntw2)[ij]
        end
        grad_num[ij] = Calculus.gradient(miniobj,ntw2.weights[ij])
    end
    return grad_ana,grad_num
end

function test_Jgradient_u(utest,ntw::A.RecurrentNetwork)
    J=A.jacobian(utest,ntw)
    gpars = A.JGradPars(ntw)
    # analytic gradient
    dgu = A.dg.(utest,ntw.gain_function)
    A._jacobian!(J,gpars,utest,dgu,ntw)
    grad_ana = vec(gpars.u)
    # numerical gradient
    grad_num = similar(gpars.u)
    for lj in CartesianIndices(gpars.u)
        (j,l)=Tuple(lj)
        function miniobj(u)
            u2 = copy(utest)
            u2[l]=u
            return A.jacobian(u2,ntw)[lj]
        end
        grad_num[lj] = Calculus.gradient(miniobj,utest[l])
    end
    grad_num = vec(grad_num)
    return grad_ana,grad_num
end


##

ne,ni = 20,23
ntot = ne+ni
ntw = A.RecurrentNetwork(ne,ni ; gfun=A.IOQuad(0.02))
fill!(ntw.weights,0.0)
u_start = randn(A.ndims(ntw))
r_start = ntw.iofunction.(u_start)

u_end,r_end=A.run_network_to_convergence(ntw,r_start)
@test all(isapprox.(u_end,ntw.external_input;atol=0.05))

ne,ni = 20,23
ntot = ne+ni
ntw = A.RecurrentNetwork(ne,ni ; gfun=A.IOQuad(0.02))
u_start = randn(A.ndims(ntw))
r_start = ntw.iofunction.(u_start)
u_end,r_end=A.run_network_to_convergence(ntw,r_start)
t,ut,rt=A.run_network(ntw,r_end,1.0)
@test all(isapprox.(u_end,ut[:,end];atol=0.05))

##

gtest = A.IOQuad(1.133)

gtest(0.0)

xtest = range(-1,5.0 ; length=100)

plot(x->gtest(x),xtest ; leg=false, linewidth=3)


descend_code_warntype(gtest,Tuple{Float64})
##

ne,ni = 20,23
ntot = ne+ni
ntw = A.RecurrentNetwork(ne,ni ; gfun=A.IOQuad(0.02))
xtest = range(-1,5.0 ; length=100)
xfill = similar(xtest)
A.iofun!(similar(xtest),xtest,ntw)
A.iofunB!(similar(xtest),xtest,ntw)

@btime A.iofun!($xfill,$xtest,$ntw)
@btime A.ioprime!($xfill,$xtest,$ntw)


@btime A.iofunB!($xfill,$xtest,$ntw)


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
ntw = A.RecurrentNetwork(ne,ni ; gfun=A.IOQuad(0.02))

utest = randn(ntot)

grad_ana,grad_num = test_Jgradient_weights(utest,ntw)
grad_ana,grad_num = test_Jgradient_u(utest,ntw)

grad_ana ./ grad_num

grad_num

grad_ana
