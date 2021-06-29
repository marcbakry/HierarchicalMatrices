using Test

include("../src/BBox.jl")

@testset "Testing the BBox structure" begin
    X = [ 
        0.512363  0.523803  0.610511;
        0.207648  0.54746   0.21826;
        0.472262  0.919383  0.313127;
        0.812478  0.223824  0.593296;
        0.109389  0.569246  0.625016
    ]
    Y = [
        0.551976   0.105557   0.733297
        0.0419843  0.0205705  0.545029
        0.237319   0.844252   0.217371
        0.759811   0.638246   0.404687
    ]

    bx = BBox(X)
    by = BBox(Y)

    # constructor
    @test_throws DomainError BBox([1 2 3 4])

    # minimum
    @test get_min(bx) == [0.109389, 0.223824, 0.21826]
    @test get_min(by) == [0.0419843, 0.0205705, 0.217371]

    # maximum
    @test get_max(bx) == [0.812478, 0.919383, 0.625016]
    @test get_max(by) == [0.759811, 0.844252, 0.733297]

    # center 
    @test get_ctr(bx) ≈ [0.4609335, 0.5716035, 0.421638]
    @test get_ctr(by) ≈ [0.40089765, 0.43241125, 0.475334]

    # edge
    @test get_edg(bx) ≈ [0.703089, 0.695559, 0.406756]
    @test get_edg(by) ≈ [0.7178267, 0.8236815, 0.515926]
end