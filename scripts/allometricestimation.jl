using Revise

using NitroFoodWeb

using StatsBase
using Graphs
using IterTools #Can remove
using Combinatorics
using GraphPlot
using Colors
using Random
using Distributions 
using DataFrames
using LinearAlgebra 
using JLD2 
# using LightGraphs
using Base.Threads
using Plots

using Optim
using CSV

##############################################################################
# 1)  Load Benguela data
##############################################################################
A_path    = smartpath("../data/foodweb/bengula_A_matrix.csv")   # your helper
mass_path = smartpath("../data/foodweb/bengula_masses.csv")

A_df      = CSV.read(A_path,  DataFrame; header = true)
mass_df   = CSV.read(mass_path, DataFrame; header = true)

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
println("Estimated parameters:")
println("   α = $α̂")
println("   β = $β̂")
println("   γ = $γ̂")
println("Converged in $(result.iterations) iterations;  NLL = $(result.minimum)")


##############################################################################
# 3)  Expected fraction correct and link probabilities
##############################################################################
frac_correct, P̂ = rohr_fraction_correct((α̂, β̂, γ̂), A, mass)

println("Expected fraction correct = $(round(frac_correct, digits = 4))")

# If you want to save the probability matrix for later use:
# using JLD2
# @save "benguela_predictions.jld2" P̂ α̂ β̂ γ̂ frac_correct








######## LATIN HYPERCUBE OPTION ###########

# 2) latin-hypercube draw (k × 3 matrix)
##############################################################################
# Latin-hypercube generator  (fixed)
##############################################################################
function lhs(k, d; rng = Random.GLOBAL_RNG)
    # 1. random offsets
    x = rand(rng, k, d)

    # 2. add stratification levels (column vector) and scale
    x .= (x .+ reshape(0:k-1, k, 1)) ./ k      # k×1 broadcasts over columns

    # 3. independent random permutation in each dimension
    for j in 1:d
        shuffle!(rng, @view x[:, j])
    end
    return x                                   # k × d matrix
end

##############################################################################
# 4)  Multistart bounds & Latin-hypercube sample
##############################################################################
lo = [-5.0, -5.0, -1.0]   # lower bounds for α, β, γ
hi = [ 5.0,  5.0,  1.0]   # upper bounds
k  = 40                   # number of LHS starts

X = lhs(k, 3)             # k×3 in [0,1]^3

A    = Matrix(A_df[:, 2:end])               # drop species names column
m    = Float64.(mass_df.mass_kg)
@assert size(A,1) == length(m)

##############################################################################
# 5)  Run Nelder–Mead from each start and keep the best
##############################################################################
f(x) = rohr_negloglik( (x[1], x[2], x[3]), A, mass )   # x is Vector
opts = Optim.Options(iterations = 5_000, show_trace = false)

best_val  = Inf
best_pars = zeros(3)

for r in 1:k
    start = lo .+ X[r, :] .* (hi .- lo)             # scale LHS point
    res   = optimize(f, start, NelderMead(), opts)  # local search
    if res.minimum < best_val
        best_val  = res.minimum
        best_pars = Optim.minimizer(res)
    end
end

println("α̂ = $(best_pars[1])   β̂ = $(best_pars[2])   γ̂ = $(best_pars[3])")
println("best NLL = $best_val")

##############################################################################
# 6)  Report results
##############################################################################
α̂, β̂, γ̂ = best_pars
println("\n≈≈≈ Best of $k Latin-hypercube starts ≈≈≈")
println("  α̂ = $(α̂)  β̂ = $(β̂)  γ̂ = $(γ̂)")
println("  negative log-likelihood = $(best_val)")

frac, _ = rohr_fraction_correct((α̂, β̂, γ̂), A, mass)

println("  Expected fraction correct (soft) = $(round(frac*100; digits = 2)) %")