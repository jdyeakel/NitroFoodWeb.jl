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

using ProgressMeter
using Statistics
using UnicodePlots

# ------------------------------------------------------------
# experiment settings
# ------------------------------------------------------------
S, C        = 100, 0.02
alpha_true      = 1.0                 # Dirichlet for Q_true
pct_grid    = 0.0:0.05:0.50       # 0–50 % known links
n_rep       = 5
steps_sa    = 15_000
wiggle_sa   = 0.05
rng_global  = MersenneTwister(20250624)  # reproducible

# ------------------------------------------------------------
# output containers
# ------------------------------------------------------------
rows = Int(length(pct_grid) * n_rep)
df   = DataFrame(pct = Float64[], rep = Int[], R2 = Float64[])

# ------------------------------------------------------------
@showprogress for pct in pct_grid, rep in 1:n_rep
    rng = MersenneTwister(hash((20250624, pct, rep)))

    #### 1. build niche web & true diets ####################################
    A, _   = nichemodelweb(S, C; rng = rng)
    tl     = trophic_levels(A)
    order  = sortperm(tl)
    A      = A[order, order]
    A_bool = (A .> 0)

    Q_true = quantitativeweb(A; alpha = alpha_true, rng = rng)
    ftl    = trophic_levels(Q_true)
    ΔTN    = 3.5
    d15N   = (ftl .- 1) .* ΔTN

    #### 2. choose links to lock ###########################################
    mask = select_known_links(Q_true; pct = pct, skew = :percol, rng = rng)

    #### 3. infer Q ########################################################
    Q_est, _ = estimate_Q_sa(
                    A_bool, d15N;
                    ΔTN        = ΔTN,
                    known_mask = mask,
                    Q_known    = Q_true,   # comment out to treat weights unknown
                    alpha0     = 1.0,
                    steps      = steps_sa,
                    wiggle     = wiggle_sa,
                    rng        = rng)

    #### 4. record R² ######################################################
    stats = evaluate_Q(Q_true, Q_est;
           known_mask = mask,   # Bool matrix same size as Q
           eps        = 1e-12)

    r2 = stats.r;

    push!(df, (pct, rep, r2))
end

# ------------------------------------------------------------
# summarise across replicates
# ------------------------------------------------------------
df_summary = combine(DataFrames.groupby(df, :pct)) do sub
    (; mean_R2 = mean(sub.R2), sd_R2 = std(sub.R2))
end

println("\nMean ± SD of R² for each pct locked")
show(df_summary, allrows=true, allcols=true)





# PARALLEL VERSION

###############################################################################
# experiment constants
###############################################################################
S, C         = 100, 0.02
alpha_true       = 10           # builds Q_true
pct_grid     = 0.0:0.05:0.50 # fraction of links locked
n_rep        = 10
steps_sa     = 15_000
wiggle_sa    = 0.05
ΔTN          = 3.5
base_seed    = 20250624      # any integer

###############################################################################
# pre-allocate result arrays (thread-safe, no push!)
###############################################################################
Nruns   = length(pct_grid) * n_rep
pct_v   = Vector{Float64}(undef, Nruns)
rep_v   = Vector{Int}(undef, Nruns)
r2_v    = Vector{Float64}(undef, Nruns)

###############################################################################
# threaded sweep
###############################################################################
@showprogress Threads.@threads for idx in 1:Nruns
    ipct = div(idx-1, n_rep) + 1          # 1-based index into pct_grid
    rep  = mod(idx-1, n_rep) + 1          # replicate number
    pct  = pct_grid[ipct]

    # independent RNG per task – deterministic but parallel-safe
    rng  = MersenneTwister(hash((base_seed, Threads.threadid(), idx)))

    # ----- build web & true diets -----------------------------------------
    A, _      = nichemodelweb(S, C; rng = rng)
    tl        = trophic_levels(A)
    order     = sortperm(tl)
    A         = A[order, order]
    A_bool    = (A .> 0)

    Q_true    = quantitativeweb(A; alpha = alpha_true, rng = rng)
    ftl_true  = trophic_levels(Q_true)
    d15N_true = (ftl_true .- 1) .* ΔTN

    # ----- lock links -----------------------------------------------------
    mask      = select_known_links(Q_true; pct = pct, skew = :rand, rng = rng)

    # ----- simulated annealing -------------------------------------------
    Q_est, _  = estimate_Q_sa(
                    A_bool, d15N_true;
                    ΔTN        = ΔTN,
                    known_mask = mask,
                    Q_known    = Q_true,     # comment out if weights unknown
                    alpha0     = 1.0,
                    steps      = steps_sa,
                    wiggle     = wiggle_sa,
                    rng        = rng)

    # Pearson R² over all realised links
    # r2        = cor(Q_true[A .> 0], Q_est[A .> 0])^2

    stats = evaluate_Q(Q_true, Q_est;
           known_mask = mask,   # Bool matrix same size as Q
           eps        = 1e-12)

    r2 = stats.r;

    pct_v[idx] = pct
    rep_v[idx] = rep
    r2_v[idx]  = r2
end

###############################################################################
# summarise & print
###############################################################################
df  = DataFrame(pct = pct_v, rep = rep_v, R2 = r2_v);
df_summary = combine(DataFrames.groupby(df, :pct)) do sub
    (; mean_R2 = mean(sub.R2), sd_R2 = std(sub.R2))
end;
println("\nMean ± SD of R² across replicates")
show(df_summary, allrows = true, allcols = true)

UnicodePlots.lineplot(df_summary[!,:mean_R2])





# PARALLEL VERSION - INCLUDING ALPHA_TRUE LOOP

###############################################################################
# experiment constants
###############################################################################
S, C          = 100, 0.02
alpha_list    = [0.5, 1.0, 10.0]          # <—— three diet breadths to test
pct_grid      = 0.0:0.05:0.50             # fraction of links locked
n_rep         = 10
steps_sa      = 15_000
wiggle_sa     = 0.05
ΔTN           = 3.5
base_seed     = 20250624

###############################################################################
# pre-allocate result arrays (thread-safe, no push!)
###############################################################################
Nruns   = length(pct_grid) * n_rep * length(alpha_list)
pct_v   = Vector{Float64}(undef, Nruns)
rep_v   = Vector{Int}(undef, Nruns)
r2_v    = Vector{Float64}(undef, Nruns)


###############################################################################
# threaded sweep
###############################################################################
@showprogress Threads.@threads for idx in 1:Nruns
    # decode flat index -> (ia , ipct , rep)
    rep  = (idx-1)              % n_rep                + 1
    ipct = ((idx-1) ÷ n_rep)    % length(pct_grid)     + 1
    ia   = ((idx-1) ÷ (n_rep*length(pct_grid)))        + 1

    alpha_true = alpha_list[ia]
    pct        = pct_grid[ipct]

    rng = MersenneTwister(hash((base_seed, Threads.threadid(), idx)))

    # --- build web & true diets ------------------------------------------
    A, _       = nichemodelweb(S, C; rng = rng)
    A_bool     = (A .> 0)
    Q_true     = quantitativeweb(A; alpha = alpha_true, rng = rng)
    ftl_true   = trophic_levels(Q_true)
    d15N_true  = (ftl_true .- 1) .* ΔTN

    # --- lock links ------------------------------------------------------
    mask       = select_known_links(Q_true; pct = pct, skew = :rand, rng = rng)

    # --- anneal ----------------------------------------------------------
    Q_est, _   = estimate_Q_sa(A_bool, d15N_true;
                               ΔTN        = ΔTN,
                               known_mask = mask,
                               Q_known    = Q_true,   # comment to let SA estimate them
                               alpha0     = 1.0,
                               steps      = steps_sa,
                               wiggle     = wiggle_sa,
                               rng        = rng)

    stats = evaluate_Q(Q_true, Q_est; known_mask = mask, eps = 1e-12)
    r2    = stats.r

    alpha_v[idx]   = alpha_true
    pct_v[idx] = pct
    rep_v[idx] = rep
    r2_v[idx]  = r2
end

###############################################################################
# build DataFrame, dropping the NaN rows
###############################################################################

df     = DataFrame(alpha = alpha_v,
                  pct   = pct_v,
                  rep   = rep_v,
                  R2    = r2_v)

df_summary = DataFrames.combine(groupby(df, [:alpha, :pct])) do sub
    (; mean_R2 = mean(sub.R2), sd_R2 = std(sub.R2))
end

println("\nMean ± SD of R² on UNKNOWN links")
show(df_summary, allrows = true, allcols = true)

###############################################################################
# plot three curves with UnicodePlots
###############################################################################
p = nothing
for alpha in alpha_list
    sub = df_summary[df_summary.alpha .== alpha, :]
    if p === nothing
        p = lineplot(sub.pct, sub.mean_R2;
                     title = "R² vs. fraction known (n = $n_rep per point)",
                     xlabel = "pct known",
                     ylabel = "mean R²",
                     width = 70, height = 20,
                     labels = ["alpha = $alpha"])
    else
        lineplot!(p, sub.pct, sub.mean_R2; labels = ["alpha = $alpha"])
    end
end
println()
display(p)

