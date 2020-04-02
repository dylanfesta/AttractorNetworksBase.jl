using Calculus
using Pkg
Pkg.activate(joinpath(@__DIR__(),".."))
using Plots
using AttractorNetworksBase ; const A = AttractorNetworksBase
##

gtest = A.GFQuad(1.133)

gtest(0.0)

xtest = range(-1,5.0 ; length=100)

plot(x->gtest(x),xtest ; leg=false, linewidth=3)

using Cthulhu

descend_code_warntype(gtest,Tuple{Float64})

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

##

ne,ni = 13,10
ntot = ne+ni
ntw = A.RecurrentNetwork(ne,ni ; gfun=A.GFQuad(0.02))

whatevs = A.run_network(10.0 .* rand(ntot), 0.5,ntw)

using Plots; theme(:dark)
_ = let t=whatevs.t,
  u12 = [ uu[1:2] for uu in whatevs.u]
  u12=hcat(u12...) |> permutedims
  plot(t,u12;leg=false,linewidth=4)
end
