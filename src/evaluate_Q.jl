function evaluate_Q(Q_true::AbstractMatrix,
                    Q_est::AbstractMatrix;
                    known_mask = falses(size(Q_true)),
                    eps        = 1e-12)

    @assert size(Q_true) == size(Q_est) == size(known_mask)

    # ------------------------------------------------------------------ #
    # 1.  Vector-level metrics on UNKNOWN links only                     #
    # ------------------------------------------------------------------ #
    present   = Q_true .> 0                          # topology mask
    unknown   = present .& .!known_mask              # links we want to score

    true_vec  = vec(Q_true[unknown])
    est_vec   = vec(Q_est[unknown])

    mae   = mean(abs.(est_vec .- true_vec))          # mean abs error
    rmse  = sqrt(mean((est_vec .- true_vec).^2))     # root-MSE
    r     = length(true_vec) > 1 ? cor(true_vec, est_vec) : NaN

    # ------------------------------------------------------------------ #
    # 2.  Column-wise KL on UNKNOWN portion (renormalised)               #
    # ------------------------------------------------------------------ #
    #the average, over all consumers j of the Kullback–Leibler divergence between the true and estimated diet‐share distributions

    S  = size(Q_true, 1)
    KL = zeros(S)

    for j in 1:S
        mask_j = unknown[:, j]
        if any(mask_j)
            t = Q_true[mask_j, j] .+ eps
            e = Q_est[mask_j, j] .+ eps
            t ./= sum(t)
            e ./= sum(e)
            KL[j] = sum(t .* log.(t ./ e))
        else
            KL[j] = 0.0          # entire column was 'known' or empty
        end
    end
    mean_KL = mean(KL)

    # ------------------------------------------------------------------ #
    # 3.  Flag badly estimated consumers (KL > 3 × median)               #
    # ------------------------------------------------------------------ #
    med = median(KL)
    bad = findall(KL .> 3med)

    return (; mae, rmse, r, mean_KL, KL, bad)
end


