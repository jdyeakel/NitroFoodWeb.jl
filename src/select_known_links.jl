"""
    select_known_links(Q_prior; pct=0.15, skew=:high, rng=MersenneTwister(0))

Return a Boolean matrix the same size as `Q_prior` where `true` marks links
that will be treated as *known* (fixed) in the optimiser.

Arguments
---------
* `pct`   – fraction of *non-zero* links to mark as known (0–1).
* `skew`  – `:high` (prefer large weights), `:rand` (uniform random), or
            `:percol` (top prey per consumer until quota hit).
"""
function select_known_links(Q_prior;
                            pct  = 0.15,
                            skew = :high,
                            rng  = Random.GLOBAL_RNG)

    S  = size(Q_prior, 1)
    m  = falses(S, S)      # output mask
    links = findall(Q_prior .> 0)
    n_known = round(Int, pct * length(links))

    if skew == :rand
        chosen = sample(rng, links, n_known; replace = false)

    elseif skew == :high
        # sort by descending weight, then pick first n_known
        weights = Q_prior[links]
        idx = sortperm(weights; rev=true)[1:n_known]
        chosen = links[idx]

    elseif skew == :percol
        chosen = Int[]
        per_consumer = ceil(Int, pct * length(links) / S)
        for j in 1:S
            prey = findall(Q_prior[:, j] .> 0)
            k    = min(per_consumer, length(prey))
            if k > 0
                idx = sortperm(Q_prior[prey, j]; rev=true)[1:k]
                LI      = LinearIndices(Q_prior)
                cidx    = CartesianIndex.(prey[idx], fill(j, k))   # vector of CartesianIndex
                append!(chosen, LI[cidx])                          # note square brackets
            end
        end
        chosen = shuffle!(rng, chosen)[1:min(n_known, length(chosen))]

    else
        error("unknown skew=: $skew")
    end

    m[chosen] .= true
    return m
end