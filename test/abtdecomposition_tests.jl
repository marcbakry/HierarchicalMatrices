using Test, LinearAlgebra

include("../src/ABtDecomposition.jl")

@testset "Testing the A-Bt decomposition" begin
    M = 4
    K = 2
    N = 5

    A  = rand(Float64,M,K)
    Bt = rand(Float32,K,N)

    ABt = ABtDecomposition(A,Bt)
    ABtt = ABtDecomposition(collect(Bt'),collect(A'))

    # type
    @test eltype(ABt) == Float64
    @test size(ABt,1) == M
    @test size(ABt,2) == N
    @test rank(ABt)   == K
    @test norm(ABt - A*Bt)/norm(ABt) ≤ eps(eltype(ABt))
    @test norm(A*(Bt*(Bt'*A'))- ABt*ABtt)/norm(ABt*ABtt) ≤ eps(eltype(ABt))

    # constructor error
    @test_throws DomainError ABtDecomposition(A,rand(K+1,N))
end