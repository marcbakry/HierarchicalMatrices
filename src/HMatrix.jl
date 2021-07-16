export HMatrix, size, type, tol, row, col, data

struct HMatrix
    m_type::UInt8
    m_size::Tuple{UInt32,UInt32}
    m_data::AbstractArray # full matrix or compressed format using ABt decomposition
    m_tol::T where {T<:Real}
    m_row::Tuple{Array{UInt32,1},Array{UInt32,1},Array{UInt32,1},Array{UInt32,1}}
    m_col::Tuple{Array{UInt32,1},Array{UInt32,1},Array{UInt32,1},Array{UInt32,1}}
    m_chd::Vector{HMatrix}
end

# accessors
Base.:size(hm::HMatrix) = hm.m_size
type(hm::HMatrix) = hm.m_type
tol(hm::HMatrix) = hm.m_tol
row(hm::HMatrix) = hm.m_row
col(hm::HMatrix) = hm.m_col
row(hm::HMatrix,i::U) where{U<:Integer} = hm.m_row[i]
col(hm::HMatrix,i::U) where{U<:Integer} = hm.m_col[i]
data(hm::HMatrix) = hm.m_data

# constructors

function HMatrix(m::T,n::T,tol::U,v) where {T<:Integer,U<:Real}
    #= Single value constructor which fills a leaf =#
    rowcol = tuple(UInt32[],UInt32[],UInt32[],UInt32[])
    return HMatrix(1,(m,n),ABtDecomposition(fill(v,(m,1)),fill(v,(1,n))),tol,rowcol,rowcol,HMatrix[])
end