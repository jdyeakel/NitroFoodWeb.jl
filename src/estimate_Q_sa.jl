"""
    estimate_Q_sa(A, d15N_obs; kwargs...)

Simulated-annealing inference of a quantitative diet matrix **Q** that
matches δ¹⁵N values while honouring the binary topology *A* and any
pre-specified *known* links.

* Locked links are indicated by `known_mask` ( true = fixed ).
* If you know their exact weights pass them in `Q_known`; otherwise leave
  `Q_known = nothing` and SA will estimate them but never modify their
  existence.
"""
function estimate_Q_sa(
        A::AbstractMatrix{Bool}, d15N_obs::AbstractVector;
        ΔTN::Real            = 3.5,
        alpha0::Real         = 0.5,
        steps::Int           = 10_000,
        wiggle::Real         = 0.05,
        known_mask::AbstractMatrix{Bool} = falses(size(A)),
        Q_known::Union{Nothing,AbstractMatrix} = nothing,
        rng::AbstractRNG     = Random.GLOBAL_RNG,
        T0_user              = nothing)

    S, ϵ = size(A, 1), 1e-12

    scrub!(v, ϵ) = @inbounds for i in eachindex(v)
        y = v[i]
        if !isfinite(y) || y <= 0
            v[i] = ϵ
        end
    end

    # ------------------------------------------------------------------ #
    # 0. initial quantitative guess                                      #
    # ------------------------------------------------------------------ #
    Q = quantitativeweb(A; alpha = alpha0, rng = rng)

    if Q_known !== nothing
        Q[known_mask] .= Q_known[known_mask]
    end

    # renormalise every consumer column so it still sums to 1
    for j in 1:S
        locked = known_mask[:, j]
        free   = A[:, j] .& .!locked
        locked_sum = sum(Q[locked, j])
        share      = max(ϵ, 1 - locked_sum)

        if any(free)
            Qfree = Q[free, j] ./ sum(Q[free, j])
            Q[free, j] .= Qfree .* share
        else
            Q[locked, j] ./= locked_sum
        end
    end

    # ------------------------------------------------------------------ #
    # 1. helpers                                                         #
    # ------------------------------------------------------------------ #
    ftl_obs = 1 .+ d15N_obs ./ ΔTN
    sse(Qm) = sum((trophic_levels(Qm) .- ftl_obs).^2)

    err       = sse(Q)
    best_err  = err
    trace     = Float64[err]
    free_cols = [j for j in 1:S if any(A[:, j] .& .!known_mask[:, j])]

    # ------------------------------------------------------------------ #
    # 2. adaptive T₀                                                     #
    # ------------------------------------------------------------------ #
    T = T0_user === nothing ? begin
        Δs = Float64[]
        for _ in 1:100
            j    = rand(rng, free_cols)
            prey = findall(A[:, j] .& .!known_mask[:, j])
            q_old = copy(Q[prey, j])
            scrub!(q_old, ϵ)                                # <-- NEW
            Q[prey, j] .= rand(rng, Dirichlet((q_old .+ ϵ)/wiggle))
            push!(Δs, sse(Q) - err)
            Q[prey, j] .= q_old
        end
        median(abs.(Δs)) / log(3)
    end : T0_user

    # ------------------------------------------------------------------ #
    # 3. main SA loop                                                    #
    # ------------------------------------------------------------------ #
    for k in 1:steps
        j    = rand(rng, free_cols)
        prey = findall(A[:, j] .& .!known_mask[:, j])
        isempty(prey) && continue

        q_old = copy(Q[prey, j])
        scrub!(q_old, ϵ)                                      # <-- NEW

        # Dirichlet proposal with clamping + safe renorm
        Qprop  = rand(rng, Dirichlet((q_old .+ ϵ) / wiggle))
        Qprop .= max.(Qprop, ϵ)
        den    = sum(Qprop)
        den == 0 && (Qprop .= ϵ; den = ϵ * length(prey))
        Qprop ./= den

        share  = max(ϵ, 1 - sum(Q[known_mask[:, j], j]))
        Qprop .*= share

        Q[prey, j] .= Qprop
        err_new     = sse(Q)

        if err_new < err || rand(rng) < exp(-(err_new - err)/T)
            err = err_new
            best_err = min(best_err, err)
        else
            Q[prey, j] .= q_old
        end

        push!(trace, best_err)
        T *= 0.9995
    end

    return Q, trace
end
