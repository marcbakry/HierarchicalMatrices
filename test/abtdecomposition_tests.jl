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

    # ACA compression
    N = 500; M = 1000
    C = zeros(ComplexF64,M,N)
    X = rand(M,3); Y = rand(N,3) .+ 1.0; V = rand(N) + 1im*rand(N)
    for j=1:N
        for i=1:M
            C[i,j] = exp(1im*10*norm(X[i,:]-Y[j,:]))/norm(X[i,:]-Y[j,:])
        end
    end
    fct(I,J) = C[I,J]
    I = collect(1:M); J = collect(1:N)
    ε = 1e-3
    ABt, flag = ACA(I,J,fct,ε)
    @test norm(C*V - ABt*V)/norm(C*V) ≤ 10*ε
    ε = 1e-4
    ABt, flag = ACA(I,J,fct,ε)
    @test norm(C*V - ABt*V)/norm(C*V) ≤ 10*ε
    ε = 1e-5
    ABt, flag = ACA(I,J,fct,ε)
    @test norm(C*V - ABt*V)/norm(C*V) ≤ 10*ε
    ε = 1e-6
    ABt, flag = ACA(I,J,fct,ε)
    @test norm(C*V - ABt*V)/norm(C*V) ≤ 10*ε
end