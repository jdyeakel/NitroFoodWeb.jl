using Random
using Distributions

function nichemodelweb(S, C; rng = Random.GLOBAL_RNG)
    numsp = 0
    niche = Float64[]
    adjmatrix = Array{Float64}(undef,0,0)

    while numsp != S
        # 1. draw niche values
        n = sort(rand(rng, S))

        # 2. feeding‐range distribution
        a, b = 1, (1/(2*C)) - 1
        bdist = Beta(a, b)
        r = n .* rand(rng, bdist, S)

        # 3. feeding centers
        c = rand.(Ref(rng), Uniform.(r ./ 2, n))

        # 4. (re)initialize the Bool adjacency
        adj = falses(S, S)

        # 5. fill in the feeding links
        for i in 1:S
            idx = findall(x -> abs(x - c[i]) < r[i]/2, n)
            adj[idx, i] .= true
        end

        # 6. remove self‑links
        for i in 1:S
            adj[i,i] = false
        end

        # 7. iteratively trim any species with degree zero
        while true
            deg = sum(adj, dims=2) .+ sum(adj, dims=1)'   # out‑ + in‑degree
            keep = vec(deg) .> 0
            if all(keep)
                break
            end
            adj = adj[keep, keep]
            n   = n[keep]
        end

        # 8. update for next iteration
        adjmatrix = Float64.(adj)
        niche     = n
        numsp     = length(n)
    end

    return adjmatrix, niche
end
