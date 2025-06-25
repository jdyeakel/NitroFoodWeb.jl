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

    # ------------------------------------------------------------------
    # 0. initial quantitative guess
    # ------------------------------------------------------------------
    Q = quantitativeweb(A; alpha = alpha0, rng = rng)

    # overwrite with supplied true values, if any ----------------------
    if Q_known !== nothing
        Q[known_mask] .= Q_known[known_mask]
    end

    # renormalise every consumer column so it still sums to 1 ----------
    for j in 1:S
        locked = known_mask[:, j]
        free   = A[:, j] .& .!locked
        locked_sum = sum(Q[locked, j])
        share      = max(ϵ, 1 - locked_sum)          # never negative

        if any(free)
            Q[free, j] ./= sum(Q[free, j])           # rescale to 1
            Q[free, j] .*= share
        else
            # all prey locked – renormalise locked slice to 1
            Q[locked, j] ./= locked_sum
        end
    end

    # ------------------------------------------------------------------
    # 1. helpers
    # ------------------------------------------------------------------
    ftl_obs = 1 .+ d15N_obs ./ ΔTN
    # sse(Qm) = sum((TrophInd(Qm) .- ftl_obs).^2)

    function sse(Qm)
        # replicate the trimming rule: keep nodes with at least one in‐ or out‐link
        deg  = vec(sum(Qm; dims = 1)) .+ vec(sum(Qm; dims = 2))
        keep = findall(!iszero, deg)              # indices TrophInd will keep

        t    = TrophInd(Qm)                       # length == length(keep)
        return sum((t .- ftl_obs[keep]).^2)       # match lengths
    end

    err        = sse(Q)
    best_err   = err
    trace      = Float64[err]

    free_cols  = [j for j in 1:S if any(A[:, j] .& .!known_mask[:, j])]

    # ------------------------------------------------------------------
    # 2. adaptive T₀ (70 % accept for median uphill move)
    # ------------------------------------------------------------------
    T = T0_user === nothing ? begin
        Δs = Float64[]
        for _ in 1:100
            j    = rand(rng, free_cols)
            prey = findall(A[:, j] .& .!known_mask[:, j])
            q_old = copy(Q[prey, j])
            Q[prey, j] .= rand(rng, Dirichlet((q_old .+ ϵ)/wiggle))
            push!(Δs, sse(Q) - err)
            Q[prey, j] .= q_old
        end
        median(abs.(Δs)) / log(3)
    end : T0_user

    # ------------------------------------------------------------------
    # 3. main SA loop
    # ------------------------------------------------------------------
    for k in 1:steps
        j    = rand(rng, free_cols)
        prey = findall(A[:, j] .& .!known_mask[:, j])  # editable cells

        q_old = copy(Q[prey, j])
        Q[prey, j] .= rand(rng, Dirichlet((q_old .+ ϵ)/wiggle))

        # renormalise free slice only
        share = 1 - sum(Q[known_mask[:, j], j])   # residual mass in this column
        share = max(ϵ, share)                    # clamp to a small positive value

        if isempty(prey) || share ≤ ϵ            # no editable cells or nothing to allocate
            continue                             # skip to next iteration
        end

        Q[prey, j] .= max.(Q[prey, j], ϵ)
        Q[prey, j] ./= sum(Q[prey, j])
        Q[prey, j] .*= share

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



# function estimate_Q_sa(A::AbstractMatrix{Bool}, d15N_obs; 
#                        ΔTN      = 3.5,
#                        alpha0   = 0.5,
#                        steps    = 10_000,
#                        wiggle   = 0.05,
#                        rng      = Random.GLOBAL_RNG,
#                        T0_user  = nothing)         # allow optional manual T0

#     S  = size(A, 1)
#     ϵ  = 1e-12

#     # ----------------------------------------------------------------------
#     # initial quantitative guess and helpers
#     # ----------------------------------------------------------------------
#     Q = quantitativeweb(A; alpha = alpha0, rng = rng)

#     ftl_obs = 1 .+ d15N_obs ./ ΔTN
#     sse(Q)  = sum((TrophInd(Q) .- ftl_obs).^2)

#     err        = sse(Q)
#     prey_cols  = [j for j in 1:S if any(A[:, j])]

#     # ----------------------------------------------------------------------
#     # quick pre-run calibration of T0  (≈70 % acceptance for median ΔE)
#     # ----------------------------------------------------------------------
#     if T0_user === nothing
#         sample_moves = 100
#         Δs = Float64[]
#         for _ in 1:sample_moves
#             j    = rand(rng, prey_cols)
#             prey = findall(A[:, j])

#             q_old  = copy(Q[prey, j])
#             α_prop = (q_old .+ ϵ) ./ wiggle
#             Q[prey, j] .= rand(rng, Dirichlet(α_prop))
#             push!(Δs, sse(Q) - err)
#             Q[prey, j] .= q_old                  # revert
#         end
#         σΔ = median(abs.(Δs))
#         T  = σΔ / log(3)                         # ≈70 % median uphill acceptance
#     else
#         T  = T0_user                             # honour user-supplied value
#     end

#     trace     = Float64[err]                     # best-so-far trace
#     best_err  = err

#     # ----------------------------------------------------------------------
#     # main SA loop
#     # ----------------------------------------------------------------------
#     for k in 1:steps
#         j    = rand(rng, prey_cols)
#         prey = findall(A[:, j])

#         q_old  = copy(Q[prey, j])
#         α_prop = (q_old .+ ϵ) ./ wiggle
#         Q[prey, j] .= rand(rng, Dirichlet(α_prop))
#         Q[prey, j] .= max.(Q[prey, j], ϵ)
#         Q[prey, j] ./= sum(Q[prey, j])

#         err_new = sse(Q)

#         if err_new < err || rand(rng) < exp(-(err_new - err)/T)
#             err = err_new
#             best_err = min(best_err, err)
#         else
#             Q[prey, j] .= q_old
#         end

#         push!(trace, best_err)
#         T *= 0.9995
#     end

#     return Q, trace
# end
