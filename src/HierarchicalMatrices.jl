module HierarchicalMatrices

using Base: Integer
import Base

include("BBox.jl")
include("BinTree.jl")

struct HMatrix{T} where{T<:Real}
    m_type::UInt8
    m_size::Tuple{UInt32,UInt32}
    m_data # full matrix or compressed format using ABt decomposition
    m_tol::T
    m_row::Tuple{UInt32,UInt32,UInt32,UInt32}
    m_col::Tuple{UInt32,UInt32,UInt32,UInt32}
    m_chd::Vector{HMatrix{T}}
end

# accessors
Base.:size(hm::HMatrix) = hm.m_size
type(hm::HMatrix) = hm.m_type
tol(hm::HMatrix) = hm.m_tol
row(hm::HMatrix) = hm.m_row
col(hm::HMatrix) = hm.m_col
row(hm::HMatrix,i::U) where{U<:Integer} = hm.m_row[i]
col(hm::HMatrix,i::U) where{U<:Integer} = hm.m_col[i]

end # module
