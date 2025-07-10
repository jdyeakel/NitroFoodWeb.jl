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

# PARALLEL VERSION - INCLUDING ALPHA_TRUE LOOP

###############################################################################
# experiment constants
###############################################################################
S, C            = 100, 0.02
alpha           = 0.5                # single diet breadth to test
n_rep           = 100                 # reps per parameter combination

# PARAMETERS TO SWEEP
wiggle_sa       = 0.05               # single SA wiggle value
Q0_div_list     = 0.0:0.25:1.0       # degree of deviation in the prior
pct_list        = 0.0:0.1:0.5        # fraction of links locked (known)

ftl_prop        = 1.0                # Assume perfect knowledge of ftls
ftl_error       = 0.0

steps_sa        = 20_000
base_seed       = 20250624
skew_setting    = "rand"

# --------------------------------------------------------------------------- #
# build parameter grids                                                       #
# --------------------------------------------------------------------------- #
Q0_param  = repeat(repeat(Q0_div_list, inner = n_rep), outer = length(pct_list))
pct_param = repeat(pct_list,               inner = length(Q0_div_list) * n_rep)
rep_param = repeat(collect(1:n_rep), outer = length(Q0_div_list) * length(pct_list))

@assert length(Q0_param) == length(pct_param) == length(rep_param)
Nruns = length(rep_param)


###############################################################################
# pre-allocate result arrays (thread-safe, no push!)
###############################################################################
rep_v   = Vector{Int}(undef, Nruns);
r2_v    = Vector{Float64}(undef, Nruns);
mae_v    = Vector{Float64}(undef, Nruns);
wmae_v    = Vector{Float64}(undef, Nruns);
rmse_v   = Vector{Float64}(undef, Nruns);
wrmse_v   = Vector{Float64}(undef, Nruns);
meanKL_v = Vector{Float64}(undef, Nruns);
pct_v    = Vector{Float64}(undef, Nruns)
Q0_div_v = Vector{Float64}(undef, Nruns)

###############################################################################
# threaded sweep
###############################################################################
@showprogress Threads.@threads for idx in 1:Nruns

    Q0_div  = Q0_param[idx]
    pct     = pct_param[idx]
    rep    = rep_param[idx]

    alpha_true = alpha        # constant defined in header

    rng = MersenneTwister(hash((base_seed, Threads.threadid(), idx)))

    # --- build web & true diets ------------------------------------------
    A, _       = nichemodelweb(S, C; rng = rng)
    A_bool     = (A .> 0)
    Q_true     = quantitativeweb(A; alpha = alpha_true, rng = rng)
    ftl_true   = trophic_levels(Q_true)

    ftl_obs = ftl_inference(ftl_true; ftl_prop = ftl_prop, ftl_error = ftl_error)
    
    # d15N_true  = (ftl_true .- 1) .* ΔTN
    # ftl_obs    = 1 .+ d15N_true ./ ΔTN

    # --- lock links ------------------------------------------------------
    mask       = select_known_links(Q_true, ftl_obs; pct = pct, skew = Symbol(skew_setting), rng = rng)

    # --- anneal ----------------------------------------------------------
    Q0 = make_prior_Q0(Q_true; deviation = Q0_div, rng = rng);
    Q_est, _   = estimate_Q_sa(A_bool, ftl_obs, Q0;
                               known_mask = mask,
                               Q_known    = Q_true,   # comment to let SA estimate them
                               steps      = steps_sa,
                               wiggle     = wiggle_sa,
                               rng        = rng)

    stats = evaluate_Q(Q_true, Q_est; known_mask = mask, eps = 1e-12)
    r2_Q    = stats.r;
    mae_Q   = stats.mae;
    wmae_Q   = stats.wmae;
    rmse_Q  = stats.rmse;
    wrmse_Q  = stats.wrmse;
    meanKL_Q = stats.mean_KL;

    Q0_div_v[idx]  = Q0_div
    pct_v[idx]   = pct
    rep_v[idx] = rep;
    r2_v[idx]  = r2_Q;
    mae_v[idx] = mae_Q;
    wmae_v[idx] = wmae_Q;
    rmse_v[idx] = rmse_Q;
    wrmse_v[idx] = wrmse_Q;
    meanKL_v[idx] = meanKL_Q;
end

#save data file
filename = smartpath("../data/prior_knownlinks_$(skew_setting).jld")
@save filename S C wiggle_sa Q0_div_list pct_list n_rep steps_sa ftl_prop ftl_error base_seed skew_setting pct_v Q0_div_v rep_v r2_v mae_v wmae_v rmse_v wrmse_v meanKL_v



skew_setting = :randsp;
filename = smartpath("../data/prior_knownlinks_$(skew_setting).jld")
@load filename S C wiggle_sa Q0_div_list pct_list n_rep steps_sa ftl_prop ftl_error base_seed skew_setting pct_v Q0_div_v rep_v r2_v mae_v wmae_v rmse_v wrmse_v meanKL_v



###############################################################################
# build DataFrame, dropping the NaN rows
###############################################################################

df = DataFrame(pct    = pct_v,
               Q0_div = Q0_div_v,
               rep    = rep_v,
               R2     = r2_v,
               MAE    = mae_v,
               WMAE   = wmae_v,
               RMSE   = rmse_v,
               WRMSE  = wrmse_v,
               meanKL = meanKL_v)


df_summary = combine(
  DataFrames.groupby(df, [:pct, :Q0_div]),

  # R²: drop any non-finite values
  :R2    => (x -> mean(filter(isfinite,    x))) => :mean_R2,
  :R2    => (x -> std(filter(isfinite,    x))) => :sd_R2,

  # MAE: drop any missing values
  :MAE   => (x -> mean(filter(!ismissing,   x))) => :mean_MAE,
  :MAE   => (x -> std(filter(!ismissing,   x))) => :sd_MAE,

  :WMAE   => (x -> mean(filter(!ismissing,   x))) => :mean_WMAE,
  :WMAE   => (x -> std(filter(!ismissing,   x))) => :sd_WMAE,

  # RMSE: same as MAE
  :RMSE  => (x -> mean(filter(!ismissing,   x))) => :mean_RMSE,
  :RMSE  => (x -> std(filter(!ismissing,   x))) => :sd_RMSE,

  :WRMSE  => (x -> mean(filter(!ismissing,   x))) => :mean_WRMSE,
  :WRMSE  => (x -> std(filter(!ismissing,   x))) => :sd_WRMSE,

  # meanKL: same pattern
  :meanKL=> (x -> mean(filter(!ismissing,   x))) => :mean_KL,
  :meanKL=> (x -> std(filter(!ismissing,   x))) => :sd_KL,
)



println("\nMean ± SD of R² on UNKNOWN links")
show(df_summary, allrows = true, allcols = true)

###############################################################################
# plot three curves with UnicodePlots
###############################################################################
nerrs = length(pct_list)                 # number of FTL‑error levels
colmap = cgrad(:viridis, nerrs; categorical = true)   # distinct colours
getcol(i) = colmap[i]                          # 1‑based colour fetch

#R2
p = nothing;
for (i, p_val) in enumerate(pct_list)
    sub = df_summary[df_summary.pct .== p_val, :]

    if p === nothing
        p = plot(sub.Q0_div, sub.mean_R2;
                 xlabel = "Degree of prior deviation (Q0_div)",
                 ylabel = "mean R²",
                 color  = getcol(i),
                 label  = "pct = $p_val",
                 size   = (500, 400),
                 frame = :box,
                 width = 2);
    else
        plot!(p, sub.Q0_div, sub.mean_R2;
              color  = getcol(i),
              label = "pct = $p_val",
              width = 2);
    end
end

#MAE
pmae = nothing;
for (i, p_val) in enumerate(pct_list)
    sub = df_summary[df_summary.pct .== p_val, :]

    if pmae === nothing
        pmae = plot(sub.Q0_div, sub.mean_MAE;
                 xlabel = "Degree of prior deviation (Q0_div)",
                 ylabel = "mean MAE",
                 color  = getcol(i),
                 size   = (500, 400),
                 frame = :box,
                 label = false,
                 width = 2);
    else
        plot!(pmae, sub.Q0_div, sub.mean_MAE;
            color  = getcol(i),  
            label = false,
            width = 2);
    end
end

#WMAE
pwmae = nothing;
for (i, p_val) in enumerate(pct_list)
    sub = df_summary[df_summary.pct .== p_val, :]

    if pwmae === nothing
        pwmae = plot(sub.Q0_div, sub.mean_WMAE;
                 xlabel = "Degree of prior deviation (Q0_div)",
                 ylabel = "mean WMAE",
                 color  = getcol(i),
                 size   = (500, 400),
                 frame = :box,
                 label = false,
                 width = 2);
    else
        plot!(pwmae, sub.Q0_div, sub.mean_WMAE;
            label = false,
            color  = getcol(i),
            width = 2);
    end
end

#RMSE
prmse = nothing;
for (i, p_val) in enumerate(pct_list)
    sub = df_summary[df_summary.pct .== p_val, :]

    if prmse === nothing
        prmse = plot(sub.Q0_div, sub.mean_RMSE;
                 xlabel = "Degree of prior deviation (Q0_div)",
                 ylabel = "mean RMSE",
                 color  = getcol(i),
                 size   = (500, 400),
                 frame = :box,
                 label = false,
                 width = 2);
    else
        plot!(prmse, sub.Q0_div, sub.mean_RMSE;
            label = false,
            color  = getcol(i),
            width = 2);
    end
end

#WRMSE
pwrmse = nothing;
for (i, p_val) in enumerate(pct_list)
    sub = df_summary[df_summary.pct .== p_val, :]

    if pwrmse === nothing
        pwrmse = plot(sub.Q0_div, sub.mean_WRMSE;
                 xlabel = "Degree of prior deviation (Q0_div)",
                 ylabel = "mean WRMSE",
                 color  = getcol(i),
                 size   = (500, 400),
                 frame = :box,
                 label = false,
                 width = 2);
    else
        plot!(pwrmse, sub.Q0_div, sub.mean_WRMSE;
            color  = getcol(i),
            label = false,
            width = 2);
    end
end


#MEAN KL
pkl = nothing;
for (i, p_val) in enumerate(pct_list)
    sub = df_summary[df_summary.pct .== p_val, :]

    if pkl === nothing
        pkl = plot(sub.Q0_div, sub.mean_KL;
                 xlabel = "Degree of prior deviation (Q0_div)",
                 ylabel = "mean KL",
                 color  = getcol(i),
                 size   = (500, 400),
                 frame = :box,
                 label = false,
                 width = 2);
    else
        plot!(pkl, sub.Q0_div, sub.mean_KL;
            color  = getcol(i),
            label = false,
            width = 2);
    end
end

combplot = plot(
    p, pmae, pwmae, prmse, pwrmse, pkl;                 # your three sub-plots
    layout = (2, 3),               # 1 row, 3 columns
    size   = (1200, 600),          
    # legend = :lowerright,          # put a single legend per panel
    margin = 7mm                   # tighten up the white space
);

display(combplot)


filename = smartpath("../figures/fig_alphaknown_$(skew_setting).pdf")
Plots.savefig(combplot,filename)
