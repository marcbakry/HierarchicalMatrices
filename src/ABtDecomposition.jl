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

Base.:*(A::ABtDecomposition{T},B::Array{S}) where {T,S} = A.A*(A.Bt*B)
Base.:*(A::Array{T},B::ABtDecomposition{S}) where {T,S} = (A*B.A)*B.Bt
Base.:*(A::ABtDecomposition,B::ABtDecomposition) = ABtDecomposition(A.A,(A.Bt*B.A)*B.Bt)

Base.getindex(A::ABtDecomposition,i::Int) = sum(
    A.A[toCartesian(i,size(A.A,1),size(A.A,2))[1],:] .*
    A.Bt[:,toCartesian(i,size(A.Bt,1),size(A.Bt,2))[2]]
)
Base.getindex(A::ABtDecomposition,i::Int,j::Int) = sum(A.A[i,:] .* A.Bt[:,j])

function ACA(I, J, fct::Function, tol::T) where {T<:Real}
    #= Adaptive Cross Approximation algorithm for the
        low-rank approximation of full matrices. =#
    # functions returning the rows or the columns of the matrix
    row(i) = fct(I[i],J)
    col(j) = fct(I,J[j])

    Nr = length(I)
    Nc = length(J)
    # pivot lists
    Ir = collect(1:Nr)
    Ic = collect(1:Nc)

    # first row
    k  = 1
    Bt = reshape(row(Ir[k]),(1,Nc))
    deleteat!(Ir,k)

    k = argmax(abs.(Bt[Ic]))
    c = Ic[k]
    β = Bt[c]
    deleteat!(Ic,k)

    A = col(c)/β
    A = reshape(A,(Nr,1))

    aa = real(dot(A,A)); bb = real(dot(Bt,Bt)); res = aa*bb
    ε = sqrt(aa)*sqrt(bb)/sqrt(res)

    anp1 = A[:,1:1]

    flag = true
    n    = 1
    while ε > tol
        if n*(Nr+Nc) > Nr*Nc
            flag = false
            break
        end

        k = argmax(abs.(anp1[Ir]))
        r = Ir[k]
        deleteat!(Ir,k)

        bnp1 = reshape(row(r),(1,Nc))
        u    = A[r:r,:]
        bnp1 .-= u*Bt

        k = argmax(abs.(bnp1[Ic]))
        c = Ic[k]
        β = bnp1[c]
        deleteat!(Ic,k)

        anp1 = reshape(col(c),(Nr,1))
        v    = Bt[:,c:c]
        anp1 .= (anp1-A*v)/β
        
        aa = real(dot(anp1,anp1)); bb = real(dot(bnp1,bnp1))
        u = A'*anp1
        v = Bt*bnp1'
        ab = real(dot(u,v))

        res += 2*ab+aa*bb
        ε = sqrt(aa)*sqrt(bb)/sqrt(res)

        A  = hcat(A, anp1)
        Bt = vcat(Bt,bnp1)

        n += 1
    end
    return ABtDecomposition(A, Bt), flag
end