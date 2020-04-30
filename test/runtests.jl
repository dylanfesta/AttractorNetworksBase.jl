using AttractorNetworksBase ; const A=AttractorNetworksBase
using Test
using Calculus,LinearAlgebra

const _rtol = 1E-4

@testset "Gain functions" begin
    # identity
    g = A.GFId()
    x = randn(1_000)
    xdest = similar(x)
    @test all(isapprox.(g.(x),x))
    @test all(isapprox.(A.ig.(x,g),x))
    A.g!(xdest,x,g)
    @test all(isapprox.(xdest,x))
    A.dg!(xdest,x,g)
    @test all(isapprox.(xdest,1.0))
    A.ddg!(xdest,x,g)
    @test all(isapprox.(xdest,0.0))
    # quadratic
    x = 3 .* randn(1_000)
    g = A.GFQuad(1+rand())
    y = g.(x)
    @test all( y .> 0.0)
    xdest = similar(x)
    A.g!(xdest,x,g)
    @test all( isapprox.(xdest,y))
    @test all(isapprox.(A.ig.(y,g),x ; rtol=_rtol))
    dgnum = Calculus.gradient.(_x->g(_x),x)
    A.dg!(xdest,x,g)
    @test all(isapprox.(xdest,dgnum; rtol=_rtol))
    ddgnum = Calculus.gradient.(_x->A.dg(_x,g),x)
    A.ddg!(xdest,x,g)
    @test all(isapprox.(xdest,ddgnum; rtol=_rtol))
end

@testset "Weight matrix" begin
    ne = 82
    ne_aux = 22
    ni = 64
    ni_aux = 17
    wmeanf = [ 0.333  -1.123
                12.122 -0.06432]
    W = A.make_wmat(ne,ni,wmeanf;noautapses=true)
    @test tr(W) == 0.0
    @test size(W,1) == size(W,2)
    @test begin
        n_tot = size(W,1)
        n_tot == ne+ni
    end
    # test if Dalian
    pe,pi = A.populations(ne,ni)
    @test all(W[:,pe] .>= 0.0)
    @test all(W[:,pi] .<= 0.0)
    @test  all(isapprox.(sum(W[pe,pe];dims=2), 0.333 ; rtol=1E-3))
    @test  all(isapprox.(sum(W[pi,pe];dims=2), 12.122 ; rtol=1E-3))
end

@testset "Network constructor" begin
    ne = 13
    ni = 10
    ntw = A.RecurrentNetwork(ne,ni ; gfun=A.GFQuad(0.123) )
    @test all( size(ntw.weights) .== (ne+ni) )
    @test A.n_neurons(ntw) == (ne+ni)
end


# Jacobian !
@testset "Jacobian matrix" begin
    ne,ni = 13,10
    ntot = ne+ni
    ntw = A.RecurrentNetwork(ne,ni ; gfun=A.GFQuad(0.123) )
    v_alloc = zeros(ntot)
    veli(u,i) = let v = A.velocity!(v_alloc,u,ntw.gain_function.(u),ntw) ; v[i]; end
    @info "building the Jacobian numerically"
    utest = randn(ntot)
    Jnum = Matrix{Float64}(undef,ntot,ntot)
    for i in 1:ntot
      Jnum[i,:] =  Calculus.gradient( u -> veli(u,i), utest )
    end
    Jan=A.jacobian(utest,ntw)
    @test all(isapprox.(Jnum,Jan;rtol=1E-4))
end
