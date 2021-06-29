using Test

include("../src/BinTree.jl")

@testset "Testing the BinTree structure" begin

# generate data
N = 1001
X = rand(Float64,N,3)

n = 50

bt = BinTree(X,n)

# this test should succeed whatever the initial distribution as the BinTree splits so that the number of nodes at a given depth remains balanced
@test maxdepth(bt) == 6
@test_throws DomainError BinTree(rand(Float64,N,2),n)

end