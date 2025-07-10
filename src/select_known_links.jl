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
function select_known_links(Q_prior, ftl_obs;
                            pct  = 0.15,
                            skew = :high,
                            rng  = Random.GLOBAL_RNG)

    S  = size(Q_prior, 1)
    m  = falses(S, S)      # output mask
    links = findall(Q_prior .> 0)
    n_known = round(Int, pct * length(links))
    LI = LinearIndices(Q_prior) # helper for Cartesian→linear

    if skew == :rand

        chosen = sample(rng, links, n_known; replace = false)

    elseif skew == :basalrand

        # ----------------  pick random LINKS, bias to low-TL consumers  -----
        tl_clean = copy(ftl_obs)
        finite   = .!isnan.(tl_clean)
        medTL    = median(tl_clean[finite])
        replace!(tl_clean, NaN => medTL)            # NaNs → median TL

        # map every realised link to its consumer column → assign weight
        cols = getindex.(links, 2)         # column (consumer) for each link
        maxTL   = maximum(tl_clean)
        # w_links = (maxTL .- tl_clean[cols]) .+ eps()  # higher weight for basal TL
        # Or use power law to increase contrast on the weights function
        p = 2.0;
        w_links = ((maxTL .- tl_clean[cols]).^p) .+ eps()
        chosen  = sample(rng, links, Weights(w_links), n_known; replace = false)

    elseif skew == :apexrand

        # ----------------  pick random LINKS, bias to high-TL consumers  ----
        tl_clean = copy(ftl_obs)
        finite   = .!isnan.(tl_clean)
        medTL    = median(tl_clean[finite])
        replace!(tl_clean, NaN => medTL)

        cols = getindex.(links, 2)         # column (consumer) for each link        
        # w_links = tl_clean[cols] .+ eps()            # higher weight for apex TL
        # Or use power law to increase contrast on the weights function
        p = 2.0;
        w_links = (tl_clean[cols] .^ p) .+ eps()
        chosen  = sample(rng, links, Weights(w_links), n_known; replace = false)

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
    
    elseif skew == :randsp
        
        # Select pct species at random and assume their ENTIRE diet is known
        chosen = Int[]
        n_sp   = max(1, round(Int, pct * S))                 # quota on species
        species = sample(rng, 1:S, n_sp; replace = false)
        
        for j in species
            prey = findall(Q_prior[:, j] .> 0)
            append!(chosen, LI[CartesianIndex.(prey, fill(j, length(prey)))])
        end

    elseif skew == :apexsp

        # Select pct species but biased towards higher trophic level and assume their ENTIRE diet is known. Requires ftl_obs
        chosen = Int[]
        
        n_sp   = max(1, round(Int, pct * S))
        # tl     = copy(ftl_obs)
        # ---- clean trophic-level weights ---------------------------------
        tl_clean = copy(ftl_obs)
        finite_mask = .!isnan.(tl_clean)
        medTL = median(tl_clean[finite_mask])          # robust mid-range value
        replace!(tl_clean, NaN => medTL)               # NaNs → median

        w = tl_clean .+ eps()                          # strictly positive

        # Bias towards apex species
        species = sample(rng, 1:S, Weights(w), n_sp; replace = false)

        for j in species
            prey = findall(Q_prior[:, j] .> 0)
            append!(chosen, LI[CartesianIndex.(prey, fill(j, length(prey)))])
        end
    
    elseif skew == :none

        chosen = Int[];

    else
        error("unknown skew=: $skew")
    end

    m[chosen] .= true
    return m
end