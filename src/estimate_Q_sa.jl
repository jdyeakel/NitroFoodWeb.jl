function estimate_Q_sa(A::AbstractMatrix{Bool}, d15N_obs; 
                       ΔTN      = 3.5,
                       alpha0   = 0.5,
                       steps    = 10_000,
                       wiggle   = 0.05,
                       rng      = Random.GLOBAL_RNG,
                       T0_user  = nothing)         # allow optional manual T0

    S  = size(A, 1)
    ϵ  = 1e-12

    # ----------------------------------------------------------------------
    # initial quantitative guess and helpers
    # ----------------------------------------------------------------------
    Q = quantitativeweb(A; alpha = alpha0, rng = rng)

    ftl_obs = 1 .+ d15N_obs ./ ΔTN
    sse(Q)  = sum((TrophInd(Q) .- ftl_obs).^2)

    err        = sse(Q)
    prey_cols  = [j for j in 1:S if any(A[:, j])]

    # ----------------------------------------------------------------------
    # quick pre-run calibration of T0  (≈70 % acceptance for median ΔE)
    # ----------------------------------------------------------------------
    if T0_user === nothing
        sample_moves = 100
        Δs = Float64[]
        for _ in 1:sample_moves
            j    = rand(rng, prey_cols)
            prey = findall(A[:, j])

            q_old  = copy(Q[prey, j])
            α_prop = (q_old .+ ϵ) ./ wiggle
            Q[prey, j] .= rand(rng, Dirichlet(α_prop))
            push!(Δs, sse(Q) - err)
            Q[prey, j] .= q_old                  # revert
        end
        σΔ = median(abs.(Δs))
        T  = σΔ / log(3)                         # ≈70 % median uphill acceptance
    else
        T  = T0_user                             # honour user-supplied value
    end

    trace     = Float64[err]                     # best-so-far trace
    best_err  = err

    # ----------------------------------------------------------------------
    # main SA loop
    # ----------------------------------------------------------------------
    for k in 1:steps
        j    = rand(rng, prey_cols)
        prey = findall(A[:, j])

        q_old  = copy(Q[prey, j])
        α_prop = (q_old .+ ϵ) ./ wiggle
        Q[prey, j] .= rand(rng, Dirichlet(α_prop))
        Q[prey, j] .= max.(Q[prey, j], ϵ)
        Q[prey, j] ./= sum(Q[prey, j])

        err_new = sse(Q)

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
