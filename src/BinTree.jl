using LinearAlgebra

include("BBox.jl")

struct BinTree{T<:Real}
    m_isleaf::Bool
    m_box::BBox{T}
    m_crd::Array{T,2}
    m_ind::Tuple{Vector{UInt32},Vector{UInt32}}
    m_sub::Array{BinTree{T},1}
end

# non-trivial constructor
function BinTree(X::Array{T,2},n::UInt32) where {T<:Real}
    if size(X,2) != 3
        throw(DomainError("size(X,2) = $(size(X,2)) != 3. The input array should only have three columns."))
    end
    #
    N = size(X,1)
    isleaf = N < n
    bb     = BBox(X)
    # recursion
    if !isleaf
        # if not a leave, get max edge length for splitting
        dim  = argmax(get_edg(bb))
        iloc = sortperm(view(X[1:N,dim])) # dispatch with same number in each new cell
        ind  = (iloc[1:(N÷2)],(N÷2+1):N)
        return BinTree{T}(isleaf,bb,X,ind,[
            BinTree{T}(X[ind[1],1:3],n),
            BinTree{T}(X[ind[2],1:3],n)
        ]) # create the children
    end
    # if current is a leave, the set of child indices is empty, so is the children set
    return BinTree{T}(isleaf,bb,X,(UInt32[],UInt32[]),BinTree{T}[])
end

# accessors
isleaf(bt::BinTree) = bt.m_isleaf
box(bt::BinTree)    = bt.m_box
crd(bt::BinTree)    = bt.m_crd
ind(bt::BinTree)    = bt.m_ind
sub(bt::BinTree)    = bt.m_sub

function leaf(bt::BinTree{T}) where {T<:Real}
    v = zeros(T,size(crd(bt),1))
    if length(sub(bt)) == 2
        v[ind(bt)[1]] .= leaf(sub(bt)[1])
        v[ind(bt)[2]] .= leaf(sub(bt)[2])
    else
        v .= rand(T,1)
    end
    return v
end