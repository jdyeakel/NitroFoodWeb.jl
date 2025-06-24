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

S = 100;
C = 0.02;

# Build adjacency matrix
A, niche = nichemodelweb(S,C)
A_bool = A .> 0
# Get fractional trophic levels
tl = TrophInd(A)
# Sort by trophic level
tlsp = sortperm(tl); 
#Redefine sorted list/matrix
tl = tl[tlsp];
A = adjmatrix[tlsp,tlsp];
# Plot adjacency matrix
p_a = heatmap(A);

# Create a uniformly random quantitative web
# alpha >> 1 ~ diets forced to equal weights
# alpha = 1 ~ uninformative, so diets uniformly distributed
# 0 < alpha << 1 ~ increasingly long tailed; specialists common
Q_true = quantitativeweb(A; alpha=0.5)
p_q = heatmap(Q_true);

# Plot Adjacency and Quantitative
# plot(p_a, p_q; layout = (2, 1), size = (400, 600))

# Look at histogram of weights
# histogram(vec(Q[(Q .> 0) .& (Q .< 1)]))

# Calculate the actual FRACTIONAL TROPHIC LEVEL based on Q
# We will use this to get the observed d15N
ftl_true = TrophInd(Q_true)
# scatter(tl,ftl)

# Provide *actual* d15N values to each species based on fractional trophic level
# enrichment per step (‰)
ΔTN    = 3.5;
d15N_true = ((ftl_true .- 1) .* ΔTN);

#Goal is to find the Q_true, which we assume is unknown
# 1) start with random Q_est with equal weights... starting guess
# 2) calculate ftl_est
# 3) convert ftl_est to d15N_est
# 4) compare d15N_est to d15N_obs to calculate an error metric
# 5) alter Q_est, repeat the steps to get d15N_est. If error is reduced, we accept the altered Q_est and continue
# 6) continue the process until we minimize the error and have produced a 'best guess' Q_est


###############################################################
# 4.  Estimate Q with simulated annealing
###############################################################
Q_est, err_trace = estimate_Q_sa(A_bool, d15N_true;
                                 ΔTN     = ΔTN,
                                 alpha0  = 1.0,    # unbiased starting guess
                                 steps   = 10_000,
                                 wiggle  = 0.05)

plot(err_trace)

ftl_est = TrophInd(Q_est)
@show cor(ftl_est, ftl_true)^2        # should be ≥ 0.99

scatter(ftl_true, ftl_est; ms=3, xlabel="observed TL", ylabel="estimated TL")
plot!([minimum(ftl_true), maximum(ftl_true)], [minimum(ftl_true), maximum(ftl_true)]; lc=:red, l=:dash, label="1:1")


scatter(Q_true[Q_true .> 0], Q_est[Q_true .> 0]; ms=3, α=0.6,
        xlabel="true weight", ylabel="estimated weight",
        title="Per-link comparison", label=:none,
        xscale=:log10,
        yscale=:log10,
        xlims=[0.01,1.1],
        ylims=[0.01,1.1])
plot!([10^-10, maximum(Q_true)], [10^-10, maximum(Q_true)]; lc=:red, l=:dash, label=:none)

#Print different stats on Q_est accuracy
stats = evaluate_Q(Q_true, Q_est)
values = [
    stats.mae,
    stats.rmse,
    stats.r,
    stats.mean_KL
]
df_stats = DataFrame(
    Metric = [
        "Mean absolute error",
        "Root-mean-sq error",
        "Pearson R (weights)",
        "Mean KL divergence"
    ],
    Value  = round.(values; digits = 4)   # broadcasted round
)
show(df_stats, allrows = true, allcols = true)

println("\nConsumers with large KL (potentially mis-fit): ",
        stats.bad)

