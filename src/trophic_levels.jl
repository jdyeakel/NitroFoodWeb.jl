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

    # ------------------------------------------------------------------ #
    # 1. Build diet-weight matrix P (column-stochastic)                  #
    # ------------------------------------------------------------------ #
    colsum = sum(Flow; dims = 1)                     # 1 × S vector
    P      = zeros(Float64, S, S)                    # allocate once

    @inbounds for j in 1:S                           # consumer column
        s = colsum[j]
        if s > 0
            invs = 1.0 / s
            @simd for i in 1:S                       # prey row
                P[i, j] = Float64(Flow[i, j]) * invs
            end
        end
        # basal consumer column (no prey) remains zeros
    end

    # ------------------------------------------------------------------ #
    # 2. Solve (I − P) * TL = 1  (with ridge fallback)                   #
    # ------------------------------------------------------------------ #
    A  = I - P                         # Float64 matrix
    rhs = ones(S)

    try
        TL = A \ rhs                   # LU factorisation
    catch e
        if e isa LinearAlgebra.SingularException
            TL = (A + 1e-12I) \ rhs    # small ridge restores invertibility
        else
            rethrow(e)
        end
    end

    return TL
end