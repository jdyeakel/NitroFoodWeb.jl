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
using CSV
using Optim
# using LightGraphs
using Base.Threads
using Plots

S = 100;
C = 0.01;

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


# #ALLOMETRIC
# #Fit Allometric web
# α̂, β̂, γ̂ = rohr_param_estimate(:Benguela)
# rohr_params = (α̂, β̂, γ̂)
# #Plot link probability from (α̂, β̂, γ̂)
# σ(x) = 1 / (1 + exp(-x))
# p_ratio(r) = σ(α̂ + β̂ * log(r) + γ̂ * log(r)^2)
# ratios = 10 .^ range(-4, 2; length = 400)   # log-spaced
# probs = p_ratio.(ratios)
# plot(ratios, probs;
#      xscale = :log10,
#      xlabel = "Prey : Predator mass ratio (mᵢ / mⱼ)",
#      ylabel = "Link probability  pᵢⱼ",
#      width = 2,
#      frame = :box,
#      legend = false)
# 
# Q_true = quantitativeweb(A; 
#                         alpha_dir = 0.5, 
#                         method = :allometric, 
#                         nichevalues = niche, 
#                         rohr_params = rohr_params);


# Create a random quantitative web
# 0 < alpha << 1 ~ increasingly long tailed; specialists common
# alpha = 1 ~ uninformative, so diets uniformly distributed
# alpha >> 1 ~ diets forced to equal weights
#RANDOM
Q_true = quantitativeweb(A; 
                        alpha_dir = 0.5, 
                        method = :rand);




p_q = Plots.heatmap(Q_true);



# Plot Adjacency and Quantitative
plot(p_a, p_q; layout = (2, 1), size = (400, 600))

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
# skew = :basalrand; Same as :rand but bias towards lower trophic level links
# skew = :apexrand; Same as :rand but bias towards higher trophic level links
# skew = :high; 1. Rank all links by descending weight. 2. Pick the first pct × (# links) of that sorted list.
# skew = :percol; 1. For each consumer j: find its prey, sort them by weight, and keep the top ⌈pct·(# total links)/S⌉ prey (or all prey if fewer). 2. Pool those top-k sets across consumers. 3. Shuffle the pooled list, then trim to the global quota pct × (# links).
# skew = :randsp; select pct SPECIES and know their full diets
# skew = :apexsp; select pct SPECIES with choices biased towards apex pos

###############################################################
# 3.  Lock in known links ~ not sure this works 100%
###############################################################

known_mask = select_known_links(Q_true, ftl_obs; pct = 0.0, skew = :rand);
p_m = Plots.heatmap(known_mask);

# plot(p_a, p_m; layout = (2, 1), size = (400, 600))

# test mask
# pct   = 0.10          # 10 % of realised links
# reps  = 20            # how many random masks to try
# skews = (:rand, :basalrand, :apexrand)

# for skew in skews
#     mean_TL = Float64[]
#     for r = 1:reps
#         mask = select_known_links(Q_true, ftl_obs; pct = pct,
#                                   skew = skew,
#                                   rng  = MersenneTwister(r))

#         # consumer column of every selected link
#         cols = getindex.(findall(mask), 2)
#         push!(mean_TL, mean(ftl_obs[cols]))
#     end
#     println(lpad(skew,10), "  mean TL of chosen consumers = ",
#             round(mean(mean_TL); digits = 3))
# end

###############################################################
# 4.  Estimate Q with simulated annealing
###############################################################

# Propose an initial Q0 - uninformative
# Q0 = quantitativeweb(A; alpha = 1.0)

# Propose an initial Q0 - informative (divergence = 0) to uninformative (divergence = 1)

Q0 = make_prior_Q0(Q_true; deviation = 1.0);
p_q0 = Plots.heatmap(Q0);
# plot(p_q, p_q0; layout = (2, 1), size = (400, 600))
# scatter(vec(Q_true),vec(Q0))

# Simulated annealing - find a best-fit Q_est
Q_est, err_trace  = estimate_Q_sa(A_bool, ftl_obs, Q0;
                        known_mask = known_mask,  # true/false links to lock
                        Q_known    = Q_true, # values for the locked links
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

Q0_uninformed = make_prior_Q0(Q_true; deviation = 1.0);
stats_Q0 = evaluate_Q(Q_true, Q0_uninformed;
           known_mask = known_mask,   # Bool matrix same size as Q
           eps        = 1e-12);

values_Q0 = [
    stats_Q0.mae,
    stats_Q0.wmae,
    stats_Q0.rmse,
    stats_Q0.wrmse,
    stats_Q0.r,
    stats_Q0.mean_KL
]

# Combine Q_est and Q0 values into a single DataFrame
df_stats_combined = DataFrame(
    Metric = [
        "Mean abs error",
        "Weighted mean abs error",
        "Root-mean-sq error",
        "Weighted root-mean-sq error",
        "Pearson R (weights)",
        "Mean KL divergence"
    ],
    Q0_Value   = round.(values_Q0; digits = 4),
    Qest_Value = round.(values; digits = 4),
    compscore = round.(values ./ values_Q0; digits = 4)
)

show(df_stats_combined, allrows = true, allcols = true)
