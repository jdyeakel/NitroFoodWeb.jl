function trophic_levels_full(Q::AbstractMatrix)
        S = size(Q, 1)
        return vec((I - Q) \ ones(S))
    end