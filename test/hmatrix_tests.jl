@testset "Testing the HMatrix builders" begin
    # builder 'single value'
    H = HMatrix(10,20,1e-3,1.0)
    @test norm(data(H) - ones(10,20)) ≤ 1e-12
end