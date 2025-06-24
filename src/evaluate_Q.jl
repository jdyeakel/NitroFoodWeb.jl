function evaluate_Q(Q_true, Q_est; eps = 1e-12)
    # mask of diet links that exist in the true web (same as A .> 0)
    m = Q_true .> 0

    # 6.1 element-wise error metrics on those links
    true_vec = vec(Q_true[m])
    est_vec  = vec(Q_est[m])

    mae   = mean(abs.(est_vec .- true_vec))                     # mean abs error
    rmse  = sqrt(mean((est_vec .- true_vec).^2))                # root-mse
    r     = cor(true_vec, est_vec)                              # Pearson R

    # 6.2 K-L divergence (per consumer)  & Jensen-Shannon if you prefer symmetry
    S   = size(Q_true, 1)
    KL  = zeros(S)
    for j in 1:S
        t = Q_true[:, j] .+ eps          # avoid log(0)
        e = Q_est[:,  j] .+ eps
        KL[j] = sum(t .* log.(t ./ e))
    end
    mean_KL = mean(KL)

    # 6.3 find “problem” consumers where KL > 3 × median
    bad = findall(KL .> 3median(KL))

    return (; mae, rmse, r, mean_KL, KL, bad)
end