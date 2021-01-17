using AttractorNetworksBase ; const A=AttractorNetworksBase
using Test
using Calculus,LinearAlgebra,Statistics
using Random ; Random.seed!(1)
const _rtol = 1E-4

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

##

@testset "Gain functions" begin
    # identity
    g = A.IOId{Float64}()
    x = randn(1_000)
    @test all(isapprox.(g.(x),x))
    @test all(isapprox.(A.ioinv.(x,g),x))
    xdest = A.ioprime.(x,g)
    @test all(isapprox.(xdest,1.0))
    xdest = A.ioprimeprime.(x,g)
    @test all(isapprox.(xdest,0.0))
    # quadratic
    x = 3 .* randn(1_000)
    g = A.IOQuad(1+rand())
    y = g.(x)
    @test all( y .> 0.0)
    xdest = g.(x)
    @test all( isapprox.(xdest,y))
    @test all(isapprox.(A.ioinv.(y,g),x ; rtol=_rtol))
    dgnum = Calculus.gradient.(_x->g(_x),x)
    xdest=A.ioprime.(x,g)
    @test all(isapprox.(xdest,dgnum; rtol=_rtol))
    ddgnum = Calculus.gradient.(_x->A.ioprime(_x,g),x)
    xdest=A.ioprimeprime.(x,g)
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
    ntw = A.BaseNetwork(ne,ni ; gfun=A.IOQuad(0.123) )
    @test all( size(ntw.weights) .== (ne+ni) )
    @test A.n_neurons(ntw) == (ne+ni)
end


# Jacobian !
@testset "Jacobian matrix" begin
    ne,ni = 13,10
    ntot = ne+ni
    ntw = A.BaseNetwork(ne,ni ; gfun=A.IOQuad(0.123) )
    v_alloc = zeros(ntot)
    veli(u,i) = let v = A.velocity!(v_alloc,u,ntw.iofunction.(u),ntw) ; v[i]; end
    @info "building the Jacobian numerically"
    utest = randn(ntot)
    Jnum = Matrix{Float64}(undef,ntot,ntot)
    for i in 1:ntot
      Jnum[i,:] =  Calculus.gradient( u -> veli(u,i), utest )
    end
    Jan=A.jacobian(utest,ntw)
    @test all(isapprox.(Jnum,Jan;rtol=1E-4))
end

@testset "Jacobian gradients" begin
    ne,ni = 13,10
    ntot = ne+ni
    ntw = A.BaseNetwork(ne,ni ; gfun=A.IOQuad(0.123) )
    utest = randn(ntot)
    anaW,numW,anau,numu=test_Jgradient(utest,ntw)
    @test all(isapprox.(anaW,numW;rtol=1E-4))
    @test all(isapprox.(anau,numu;rtol=1E-4))
    ne,ni = 23,12
    ntot = ne+ni
    ntw = A.BaseNetwork(ne,ni ; gfun=A.IOId{Float64}() )
    utest = randn(ntot)
    # weights
    anaW,numW,anau,numu=test_Jgradient(utest,ntw)
    @test all(isapprox.(anaW,numW;rtol=1E-4))
    @test all(isapprox.(anau,numu;rtol=1E-4))
end

@testset "Dynamics" begin
    ne,ni = 20,23
    ntot = ne+ni
    ntw = A.BaseNetwork(ne,ni ; gfun=A.IOQuad(0.02))
    fill!(ntw.weights,0.0)
    u_start = randn(A.ndims(ntw))
    r_start = ntw.iofunction.(u_start)
    u_end,_=A.run_network_to_convergence(ntw,r_start)
    @test all(isapprox.(u_end,ntw.external_input;atol=0.05))
    ntw = A.BaseNetwork(ne,ni ; gfun=A.IOQuad(0.02))
    r_start = ntw.iofunction.(u_start)
    u_end,r_end=A.run_network_to_convergence(ntw,r_start)
    t,ut,rt=A.run_network(ntw,r_end,1.0)
    @test all(isapprox.(u_end,ut[:,end];atol=0.05))
end

@testset "Making attractors" begin
    ne,ni = 200,100
    natt=431
    ntw = A.AttractorNetwork(natt,ne,ni ;
        gfun=A.IOQuad(3.45),mu_attr=3.23,std_attr=2.2)
    attr_u = ntw.attractors_u
    attr_r = ntw.iofunction.(attr_u)
    @test isapprox(mean(attr_r),3.23;atol=0.05)
    @test isapprox(std(attr_r),2.2;atol=0.1)
    @test A.n_attractors(ntw) == natt
end
