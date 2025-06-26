"""
    trophic_levels(Flow::AbstractMatrix) -> Vector{Float64}

Fast computation of fractional trophic level (TL) for *every* node
(no trimming, no omnivory index).

* `Flow[i,j]` is the biomass (or diet share) flowing **from prey _i_ to consumer _j_**.
* If `Flow` is an unweighted 0/1 adjacency matrix, each diet link is
  automatically given equal weight `1 / (# prey of consumer)`, reproducing the
  Pauly et al. (1998) definition used in the original `TrophInd`.

Algorithm
---------
1. **Column-normalise** `P[i,j] = Flow[i,j] / Σ_i Flow[i,j]`.
2. Solve the linear system `(I − P) * TL = 1`.

A small ridge (`1e-12 I`) is added automatically if `(I − P)` is singular
(e.g. a consumer column is a pure self-loop).
"""
function trophic_levels(Flow::AbstractMatrix{T}) where {T<:Real}
    S = size(Flow, 1)
    @assert size(Flow, 2) == S "Flow matrix must be square"

    # --------------------------------------------------------------- #
    # 1. Column-normalised diet matrix P                              #
    # --------------------------------------------------------------- #
    colsum = sum(Flow; dims = 1)
    P      = zeros(Float64, S, S)

    @inbounds for j in 1:S
        s = colsum[j]
        if s > 0
            invs = 1.0 / s
            @simd for i in 1:S
                P[i, j] = Float64(Flow[i, j]) * invs
            end
        end
    end

    # --------------------------------------------------------------- #
    # 2. Solve (I − P') TL = 1  (with ridge fallback)                  #
    # --------------------------------------------------------------- #
    A   = I - transpose(P)
    rhs = ones(S)

    TL = Vector{Float64}(undef, S)           # pre-declare for scope

    try
        TL .= A \ rhs
    catch e
        if e isa LinearAlgebra.SingularException
            TL .= (A + 1e-12I) \ rhs         # tiny ridge restores rank
        else
            rethrow(e)
        end
    end

    return TL
end
