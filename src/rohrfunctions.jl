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


function rohr_param_estimate(foodweb::Symbol)

    if foodweb == :Benguela
        ##############################################################################
        # 1)  Load Benguela data
        ##############################################################################
        A_path    = smartpath("../data/foodweb/bengula_A_matrix.csv")   # your helper
        mass_path = smartpath("../data/foodweb/bengula_masses.csv")

        A_df      = CSV.read(A_path,  DataFrame; header = true)
        mass_df   = CSV.read(mass_path, DataFrame; header = true)

        #Options to load different food web matrices/mass vectors

    end


    # First column of the csv is the species name → drop it
    A = Matrix(A_df[:, 2:end])                # prey = rows, predators = columns
    mass = Float64.(mass_df.mass_kg)          # adjust column name if different

    @assert size(A, 1) == length(mass) "Mass vector length ≠ number of species"


    ##############################################################################
    # 2)  Estimate α, β, γ with Nelder–Mead
    ##############################################################################
    negloglik_vec(x) = rohr_negloglik( (x[1], x[2], x[3]), A, mass )

    initial_x    = zeros(3)

    opt_settings = Optim.Options(        # ← put solver options *here*
                        iterations  = 10_000,
                        store_trace = true,
                        show_trace  = false)

    result   = optimize(negloglik_vec,
                        initial_x,
                        NelderMead(),
                        opt_settings)     # 4th positional arg

    params_hat = Optim.minimizer(result)     # Vector{Float64}(3)
    α̂, β̂, γ̂ = params_hat

    return α̂, β̂, γ̂

end