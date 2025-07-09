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
        A::AbstractMatrix{Bool}, ftl_obs::AbstractVector;
        alpha0::Real         = 0.5,
        steps::Int           = 10_000,
        wiggle::Real         = 0.05,
        known_mask::AbstractMatrix{Bool} = falses(size(A)),
        Q_known::Union{Nothing,AbstractMatrix} = nothing,
        rng::AbstractRNG     = Random.GLOBAL_RNG,
        T0_user              = nothing)

    S, ϵ = size(A, 1), 1e-12

    # scrub!(v, ϵ) = @inbounds for i in eachindex(v)
    #     y = v[i]
    #     if !isfinite(y) || y <= 0
    #         v[i] = ϵ
    #     end
    # end

    scrub!(v, ϵ) = @inbounds for i in eachindex(v)
        y = v[i]
        if !isfinite(y) || y < 0           # strictly < 0
            v[i] = ϵ
        end
    end

    # Define known observations
    observed_idx = findall(!isnan,ftl_obs)

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
            s = sum(Q[free, j])
            Q[free, j] ./= s == 0 ? length(Q[free, j]) : s  # avoid /0
            Q[free, j] .*= share
        elseif locked_sum > 0
            Q[locked, j] ./= locked_sum                     # all-locked col
        end
    end

    # ------------------------------------------------------------------ #
    # 1. helpers                                                         #
    # ------------------------------------------------------------------ #
    # ftl_obs = 1 .+ d15N_obs ./ ΔTN
    sse(Qm) = sum((trophic_levels(Qm)[observed_idx] .- ftl_obs[observed_idx]).^2)

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

        # ---------------------------------------------------- #
        # 1. Dirichlet proposal for the editable prey slice    #
        # ---------------------------------------------------- #
        q_old = copy(Q[prey, j])
        scrub!(q_old, ϵ)

        Qprop  = rand(rng, Dirichlet((q_old .+ ϵ) / wiggle))
        Qprop .= max.(Qprop, ϵ)
        Qprop ./= sum(Qprop)                    # now sums to 1

        share  = max(ϵ, 1 - sum(Q[known_mask[:, j], j]))
        Qprop .*= share                         # scale to residual mass

        # ---------------------------------------------------- #
        # 2. Assemble candidate full column                    #
        # ---------------------------------------------------- #
        old_col = copy(@view Q[:, j])           # immutable copy for rollback
        new_col = copy(old_col)                 # start with current
        new_col[prey] .= Qprop                  # update editable slice
        scrub!(new_col, ϵ)                      # ensure finite

        # ---------------------------------------------------- #
        # 3. Evaluate and accept/reject                        #
        # ---------------------------------------------------- #
        Q[:, j] .= new_col                      # tentative write
        err_new  = sse(Q)

        if err_new < err || rand(rng) < exp(-(err_new - err)/T)
            err      = err_new
            best_err = min(best_err, err)
        else
            Q[:, j] .= old_col                  # full rollback
        end

        push!(trace, best_err)
        T *= 0.9995
    end


    return Q, trace
end
