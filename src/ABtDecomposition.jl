using LinearAlgebra

struct ABtDecomposition{T} <: AbstractArray{T,2}
    A::Array{T,2}
    Bt::Array{T,2}
end

# required in order to inherit from AbstractArray
function ABtDecomposition(A::Array{T,2},Bt::Array{U,2}) where{T<:Number,U<:Number}
    if size(A,2) != size(Bt,1)
        throw(DomainError("size(A,2) != size(Bt,1). The input arrays should be compatible for the direct matrix-matrix multiplication."))
    end
    cT = promote_type(T,U)
    return ABtDecomposition{cT}(
        T == cT ? A  : convert(Array{cT,2},A),
        U == cT ? Bt : convert(Array{cT,2},Bt)
    )
end

# get cartesian index
function toCartesian(i,M,N)
    return ((i-1)%M + 1, i÷N + 1)
end

LinearAlgebra.:rank(A::ABtDecomposition) = size(A.A,2)

Base.size(ABt::ABtDecomposition) = (size(ABt.A,1),size(ABt.Bt,2))
Base.size(ABt::ABtDecomposition,i) = begin
    if i == 1
        return size(ABt.A,1)
    elseif i == 2
        return size(ABt.Bt,2)
    else
        throw(ArgumentError("Invalid dimension. Should be 1 or 2."))
    end
end

Base.:*(A::ABtDecomposition,B::AbstractArray) = A.A*(A.Bt*B)
Base.:*(A::AbstractArray,B::ABtDecomposition) = (A*B.A)*B.Bt
Base.:*(A::ABtDecomposition,B::ABtDecomposition) = ABtDecomposition(A.A,(A.Bt*B.A)*B.Bt)

Base.getindex(A::ABtDecomposition,i::Int) = sum(
    A.A[toCartesian(i,size(A.A,1),size(A.A,2))[1],:] .*
    A.Bt[:,toCartesian(i,size(A.Bt,1),size(A.Bt,2))[2]]
)
Base.getindex(A::ABtDecomposition,i::Int,j::Int) = sum(A.A[i,:] .* A.Bt[:,j])