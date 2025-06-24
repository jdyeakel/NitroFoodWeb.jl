function quantitativeweb(A; alpha = 1.0, rng = Random.GLOBAL_RNG)
    S, _ = size(A)
    Q    = zeros(Float64, S, S)

    for j in 1:S
        prey = findall(!iszero, A[:, j])
        f    = length(prey)
        f == 0 && continue

        weights = rand(rng, Dirichlet(fill(alpha, f)))  # one line does it all
        Q[prey, j] .= weights
    end
    return Q
end

