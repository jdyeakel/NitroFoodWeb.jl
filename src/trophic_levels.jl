"""
    trophic_levels(Flow::AbstractMatrix{<:Real}) -> Vector{Float64}

Return fractional trophic level (TL) for **every** node in `Flow`
(no trimming, no omnivory index).

`Flow[i,j]` is the biomass or diet flow *from* prey *i* *to* consumer *j*.
For an unweighted adjacency matrix the routine
assigns equal diet share `1 / (# prey)` to each link, matching Pauly et al.
(1998) and the original `TrophInd` logic.

### Algorithm
1. Build the diet-weight matrix **P**  
   `P[row=consumer, col=prey] = Flow[prey,consumer] / Σ prey`
2. Solve the linear system  
   `(I − P) · TL = 1`.

This is equivalent to the Neumann‐series solution  
`TL = (I − P)⁻¹ · 1` but is faster and numerically stable for S ≲ 10⁴.

### Notes
* Basal nodes (no incoming diet links) have `TL = 1`.
* Works for `Float64`, `Float32`, or integer 0/1 adjacency matrices.
"""
function trophic_levels(Flow::AbstractMatrix{T}) where {T<:Real}
    S = size(Flow, 1)
    @assert size(Flow, 2) == S "Flow matrix must be square"

    # ------------------------------------------------------------------ #
    # 1. Build diet-weight matrix P (row = consumer, col = prey)         #
    # ------------------------------------------------------------------ #
    colsum = sum(Flow; dims = 1)                     # 1 × S vector
    P      = zeros(Float64, S, S)                    # allocate once

    @inbounds for j in 1:S                           # consumer column
        s = colsum[j]
        if s > 0                                     # consumer with prey
            invs = 1.0 / s
            @simd for i in 1:S                       # prey row
                P[j, i] = Float64(Flow[i, j]) * invs
            end
        end
        # basal consumer column remains all zeros
    end

    # ------------------------------------------------------------------ #
    # 2. Solve (I − P) * TL = 1                                          #
    # ------------------------------------------------------------------ #
    A  = I - P                                         # implicitly converts to Float64
    TL = A \ ones(S)

    return TL
end
