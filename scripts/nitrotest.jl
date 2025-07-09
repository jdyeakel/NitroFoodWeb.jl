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
# tltest = TrophInd(A)
tl = trophic_levels(A)
# Sort by trophic level
tlsp = sortperm(tl); 
#Reorder sorted list/matrices for analysis/presentation
tl = tl[tlsp];
A = A[tlsp,tlsp];
A_bool = A_bool[tlsp,tlsp];
# Plot adjacency matrix
p_a = Plots.heatmap(A);

# Create a random quantitative web
# 0 < alpha << 1 ~ increasingly long tailed; specialists common
# alpha = 1 ~ uninformative, so diets uniformly distributed
# alpha >> 1 ~ diets forced to equal weights
Q_true = quantitativeweb(A; alpha=1.0);
p_q = Plots.heatmap(Q_true);

# Plot Adjacency and Quantitative
# plot(p_a, p_q; layout = (2, 1), size = (400, 600))

# Look at histogram of weights
# histogram(vec(Q_true[(Q_true .> 0) .& (Q_true .< 1)]))
# histogram(vec(Q_true[(Q_true .> 0)]),bins=20)

# Calculate the actual FRACTIONAL TROPHIC LEVEL based on Q
# We will use this to get the observed d15N
ftl_true = trophic_levels(Q_true)
# scatter(tl,ftl)

# A function to derive observed trophic levels
##### We assume we know all primary producers #####
# Allow a percentage of NON-PRIMARY PRODUCER trophic levels to be drawn: ftl_prop
# Allow a certain amount of error on NON-PRIMARY PRODUCER observed trophic levels: ftl_error
ftl_obs = ftl_inference(ftl_true; ftl_prop = 1.0, ftl_error = 0.0)

# #Plot check
# scatter(ftl_true,ftl_obs)
# plot!(collect(1:5),collect(1:5))

# Provide *actual* d15N values to each species based on fractional trophic level
# enrichment per step (‰)
# ΔTN    = 3.5;
# d15N_true = ((ftl_true .- 1) .* ΔTN);

# ftl_obs = 1 .+ d15N_true ./ ΔTN

#Goal is to find the Q_true, which we assume is unknown
# 1) start with random Q_est with equal weights... starting guess
# 2) calculate ftl_est
# 3) convert ftl_est to d15N_est
# 4) compare d15N_est to d15N_obs to calculate an error metric
# 5) alter Q_est, repeat the steps to get d15N_est. If error is reduced, we accept the altered Q_est and continue
# 6) continue the process until we minimize the error and have produced a 'best guess' Q_est


# Specify known links using one of the following methods:
# skew = :rand; 1. Collect every non-zero link in Q_prior. 2. Draw pct × (# links) without replacement using sample.
# skew = :high; 1. Rank all links by descending weight. 2. Pick the first pct × (# links) of that sorted list.
# skew = :percol; 1. For each consumer j: find its prey, sort them by weight, and keep the top ⌈pct·(# total links)/S⌉ prey (or all prey if fewer). 2. Pool those top-k sets across consumers. 3. Shuffle the pooled list, then trim to the global quota pct × (# links).
# skew = :randsp; select pct SPECIES and know their full diets
# skew = :apexsp; select pct SPECIES with choices biased towards apex pos

###############################################################
# 3.  Lock in known links ~ not sure this works 100%
###############################################################

known_mask = select_known_links(Q_true, ftl_obs; pct = 0.25, skew = :apexsp);
p_m = Plots.heatmap(known_mask);

# plot(p_a, p_m; layout = (2, 1), size = (400, 600))

###############################################################
# 4.  Estimate Q with simulated annealing
###############################################################
Q_est, err_trace  = estimate_Q_sa(A_bool, ftl_obs;
                        known_mask = known_mask,  # true/false links to lock
                        Q_known    = Q_true, # values for the locked links
                        alpha0 = 1.0, 
                        steps = 20_000,
                        wiggle  = 0.05)

# Smaller wiggle ⇒ each αᵢ is larger ⇒ the Dirichlet is concentrated near the current weights.
# Larger wiggle ⇒ αᵢ shrinks ⇒ the Dirichlet is flatter, so proposed weights can differ a lot.

plot(err_trace,yscale=:log10)


#################################
# PLOT TROPHIC LEVEL CORRELATION
#################################
ftl_est = trophic_levels(Q_est)
@show cor(ftl_est, ftl_true)^2        # should be ≥ 0.99
scatter(ftl_true, ftl_est; ms=3, xlabel="observed TL", ylabel="estimated TL")
plot!([minimum(ftl_true), maximum(ftl_true)], [minimum(ftl_true), maximum(ftl_true)]; lc=:red, l=:dash, label="1:1")


#################################
# PLOT Q WEIGHTS CORRELATION
#################################
 
# flatten matrices and collect indices ------------------------------------
true_vec   = Q_true[Q_true .> 0]               # all non-zero true weights
est_vec    = Q_est[Q_true .> 0]                # estimated counterparts
known_vec  = known_mask[Q_true .> 0]           # Boolean mask in same order
idx_known     = findall(known_vec)
idx_unknown   = findall(.!known_vec)
scatter(true_vec[idx_unknown], est_vec[idx_unknown];
        ms      = 3,  α = 0.6,
        # xscale  = :log10, yscale = :log10,
        xlabel  = "true weight",
        ylabel  = "estimated weight",
        title   = "Per-link comparison",
        xlims   = (0.0, 1.1),
        ylims   = (0.0, 1.1),
        label   = "free links")
scatter!(true_vec[idx_known], est_vec[idx_known];
         ms = 5, mc = :orange, markerstrokecolor = :black,
         label = "locked links")
# 1:1 reference line -------------------------------------------------------
plot!([1e-10, maximum(true_vec)], [1e-10, maximum(true_vec)];
      lc = :red, l = :dash, label = "1 : 1")



################################################################
# PRINT STATS - evaluate only unknown links (ignores known_mask)
################################################################
# #Print different stats on Q_est accuracy
# stats = evaluate_Q(Q_true, Q_est)

stats = evaluate_Q(Q_true, Q_est;
           known_mask = known_mask,   # Bool matrix same size as Q
           eps        = 1e-12)

values = [
    stats.mae,
    stats.wmae,
    stats.rmse,
    stats.wrmse,
    stats.r,
    stats.mean_KL
]
df_stats = DataFrame(
    Metric = [
        "Mean abs error",
        "Weighted mean abs error",
        "Root-mean-sq error",
        "Weighted root-mean-sq error",
        "Pearson R (weights)",
        "Mean KL divergence"
    ],
    Value  = round.(values; digits = 4)   # broadcasted round
)
show(df_stats, allrows = true, allcols = true)

println("\nConsumers with large KL (potentially mis-fit): ",
        stats.bad)

