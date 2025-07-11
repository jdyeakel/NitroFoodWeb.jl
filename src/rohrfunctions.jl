##############################################################################
# 1)  Negative log-likelihood of the Rohr body-size model
##############################################################################
"""
    rohr_negloglik(params, A, mass)

Return **negative** log-likelihood of the quadratic logit body-size model
(α, β, γ) = `params`, for a binary predation matrix `A` (rows = prey,
columns = predators) and vector of body masses `mass`.

Basal species (columns with no outgoing links) are ignored, exactly as in Rohr.
"""
##############################################################################
# 1)  Negative log-likelihood  (fixed sign)
##############################################################################
function rohr_negloglik(params::NTuple{3, <:Real},
                        A::AbstractMatrix{<:Integer},
                        mass::AbstractVector{<:Real})::Float64
    α, β, γ = params
    n_prey, n_pred = size(A)
    predators = findall(!iszero, vec(sum(A, dims = 1)))

    ll = 0.0                               # log-likelihood (to be MAXIMISED)
    @inbounds for j in predators           # predator (columns)
        mj = mass[j]
        for i in 1:n_prey                  # prey (rows)
            mi  = mass[i]
            z   = log(mi / mj)             # prey ÷ predator
            lin = α + β*z + γ*z*z
            if A[i, j] == 1
                ll -= log1p(exp(-lin))     # add log p
            else
                ll -= log1p(exp( lin))     # add log (1-p)
            end
        end
    end
    return -ll                             # ← MINIMISE this (positive) value
end

##############################################################################
# 2)  Expected fraction of correctly classified interactions
##############################################################################
"""
    rohr_fraction_correct(params, A, mass; threshold = 0.5)

Return the **expected fraction correct** (soft accuracy) and a matrix of
predicted probabilities.  Follows eq. above; basal predators excluded.

`threshold` is *only* used to illustrate hard classification in the comments;
it does not influence the fraction-correct calculation.
"""
function rohr_fraction_correct(params::NTuple{3, <:Real},
                               A::AbstractMatrix{<:Integer},
                               mass::AbstractVector{<:Real};
                               threshold::Real = 0.5)

    α, β, γ = params
    n_prey, n_pred = size(A)
    predators  = findall(!iszero, vec(sum(A, dims = 1)))

    P̂ = zeros(Float64, n_prey, n_pred)   # predicted linking probs
    correct_exp = 0.0
    total_pairs = 0

    @inbounds for j in predators
        mj = mass[j]
        for i in 1:n_prey
            mi = mass[i]
            lin = α + β*log(mi/mj) + γ*log(mi/mj)^2
            p   = 1/(1 + exp(-lin))       # logistic
            P̂[i, j] = p

            # Expected correctness for this pair
            correct_exp += A[i, j]*p + (1 - A[i, j])*(1 - p)
            total_pairs += 1
        end
    end

    frac_correct = correct_exp / total_pairs
    return frac_correct, P̂
end