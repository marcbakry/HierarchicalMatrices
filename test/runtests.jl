using Test, LinearAlgebra, HierarchicalMatrices

# testing BBox
include("bbox_tests.jl")

# testing BinTree
include("bintree_tests.jl")

# A-Bt decomposition
include("abtdecomposition_tests.jl")

# Hierarchical Matrices
include("hmatrix_tests.jl")