using StaticArrays, LinearAlgebra

export BBox, get_min, get_max, get_edg, get_ctr, show, isfar

struct BBox{T<:Real}
    m_ctr::SVector{3,T}
    m_edg::SVector{3,T}
    m_max::SVector{3,T}
    m_min::SVector{3,T}
end

function BBox(X::Array{T,2})  where {T<:Real}
    if size(X,2) != 3
        throw(DomainError("size(X,2) = $(size(X,2)) != 3. The input array should only have three columns."))
    end
    m_min = SVector{3}(minimum(X,dims=1))
    m_max = SVector{3}(maximum(X,dims=1))
    m_edg = m_max - m_min
    return BBox((m_min+m_max)/2,m_edg,m_max,m_min)
end

# accessors
get_max(b::BBox) = b.m_max
get_min(b::BBox) = b.m_min
get_ctr(b::BBox) = b.m_ctr
get_edg(b::BBox) = b.m_edg

# utilities
function isfar(bx::BBox,by::BBox,η)
    d1 = max.(0,get_min(bx) - get_max(by))
    d2 = max.(0,get_min(by) - get_max(bx))
    diamX = norm(get_edg(bx))
    diamY = norm(get_edg(by))
    distXY = sqrt(sum(d1.^2 + d2.^2))
    return min(diamX,diamY) ≤ η*distXY
end

function Base.:show(io::IO, b::BBox)
    print(io,"Bounding box data :\n")
    print(io,"- Minimum : "); print(io,get_min(b))
    print(io,"- Maximum : "); print(io,get_max(b))
    print(io,"- Center  : "); print(io,get_ctr(b))
    print(io,"- Edge    : "); print(io,get_edg(b))
    print(io,"\n")
end