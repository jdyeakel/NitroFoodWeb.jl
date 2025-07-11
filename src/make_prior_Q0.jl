"""
    make_prior_Q0(Q_true; deviation = 0.3, rng = Random.GLOBAL_RNG)

Return a prior diet matrix *Q0* that interpolates between the ground-truth
`Q_true` (`deviation = 0`) and an uninformative Dirichlet draw
(`deviation = 1`).  Requires the binary topology to be the same as in
`Q_true` – zeros remain zeros.

Arguments
---------
* `deviation`  – scalar in [0,1] controlling how far to move toward the
                 uninformative prior.
* `rng`        – any `AbstractRNG` you wish; defaults to `GLOBAL_RNG`.
"""
function make_prior_Q0(Q_true; 
    deviation = 1.0,
    rng       = Random.GLOBAL_RNG)

    @assert 0 ≤ deviation ≤ 1 "deviation must be in [0,1]"

    A       = Q_true .> 0                       # topology mask
    Q_unif  = quantitativeweb(A; alpha_dir = 1.0, method = :rand, rng = rng)

    # Weighted average between TRUE and UNINFORMATIVE
    Q0 = (1 - deviation) .* Q_true .+ deviation .* Q_unif

    # Renormalise each consumer column over its realised prey set - colsums = 1
    S = size(Q_true, 1)
    for j in 1:S
        prey = findall(A[:, j])
        if !isempty(prey)
            s = sum(Q0[prey, j])
            Q0[prey, j] ./= s        # now sums to 1 again
        end
    end
    return Q0
end