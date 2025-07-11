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
using Optim
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
S, C         = 100, 0.02;
alpha_true   = 0.5;          # <—— three diet breadths to test
pct          = 0.0;             # fraction of links locked
n_rep        = 100;

ftl_prop_list      = 0.25:0.05:1.0;      # Assume perfect knowledge of ftls
ftl_error_list     = 0.0:0.05:0.25;

steps_sa      = 20_000;
wiggle_sa     = 0.05;
# ΔTN           = 3.5;
base_seed     = 20250624;

skew_setting  = "rand";

ftlp_param = repeat(ftl_prop_list, inner = length(ftl_error_list)*n_rep)
ftle_param   = repeat(repeat(ftl_error_list, inner = n_rep), outer = length(ftl_prop_list))
rep_param   = repeat(collect(1:n_rep),  outer = length(ftl_prop_list)*length(ftl_error_list))

@assert length(ftlp_param) == length(ftle_param) == length(rep_param)
Nruns = length(ftlp_param)


###############################################################################
# pre-allocate result arrays (thread-safe, no push!)
###############################################################################
ftlp_v = Vector{Float64}(undef, Nruns);
ftle_v   = Vector{Float64}(undef, Nruns);
rep_v   = Vector{Int}(undef, Nruns);
r2_v    = Vector{Float64}(undef, Nruns);
mae_v    = Vector{Float64}(undef, Nruns);
wmae_v    = Vector{Float64}(undef, Nruns);
rmse_v   = Vector{Float64}(undef, Nruns);
wrmse_v   = Vector{Float64}(undef, Nruns);
meanKL_v = Vector{Float64}(undef, Nruns);

@showprogress Threads.@threads for idx in 1:Nruns
    # decode flat index -> (ia , ipct , rep)
    # rep  = (idx-1)              % n_rep                + 1
    # ipct = ((idx-1) ÷ n_rep)    % length(pct_grid)     + 1
    # ia   = ((idx-1) ÷ (n_rep*length(pct_grid)))        + 1

    # alpha_true = alpha_list[ia]
    # pct        = pct_grid[ipct]

    ftl_prop  = ftlp_param[idx]      # ← use the pre-expanded vector
    ftl_error = ftle_param[idx]
    rep       = rep_param[idx]

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
    Q0 = quantitativeweb(A; alpha = 1.0, rng = rng)
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

    ftlp_v[idx]   = ftl_prop;
    ftle_v[idx] = ftl_error;
    rep_v[idx] = rep;
    r2_v[idx]  = r2_Q;
    mae_v[idx] = mae_Q;
    wmae_v[idx] = wmae_Q;
    rmse_v[idx] = rmse_Q;
    wrmse_v[idx] = wrmse_Q;
    meanKL_v[idx] = meanKL_Q;
end


# #save data file
filename = smartpath("../data/trophiclevelerror_$(skew_setting).jld")
@save filename S C alpha pct ftl_prop_list ftl_error_list n_rep steps_sa wiggle_sa base_seed skew_setting ftlp_v ftle_v rep_v r2_v mae_v wmae_v rmse_v wrmse_v meanKL_v

# #load data file
skew_setting = "rand";
filename = smartpath("../data/trophiclevelerror_$(skew_setting).jld")
@load filename S C alpha pct ftl_prop_list ftl_error_list n_rep steps_sa wiggle_sa base_seed skew_setting ftlp_v ftle_v rep_v r2_v mae_v wmae_v rmse_v wrmse_v meanKL_v


df     = DataFrame(ftl_prop = ftlp_v,
                  ftl_error   = ftle_v,
                  rep   = rep_v,
                  R2    = r2_v,
                  MAE   = mae_v,
                  WMAE  = wmae_v,
                  RMSE  = rmse_v,
                  WRMSE = wrmse_v,
                  meanKL = meanKL_v);


df_summary = combine(
  DataFrames.groupby(df, [:ftl_prop, :ftl_error]),

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


nerrs = length(ftl_error_list)                 # number of FTL‑error levels
colmap = cgrad(:viridis, nerrs; categorical = true)   # distinct colours
getcol(i) = colmap[i]                          # 1‑based colour fetch

#R2
p = nothing;
for (i, err) in enumerate(ftl_error_list)

    sub = df_summary[df_summary.ftl_error .== err, :]

    if p === nothing
        p = plot(sub.ftl_prop, sub.mean_R2;
                 xlabel = "Proportion TL known",
                 ylabel = "mean R²",
                 color  = getcol(i),
                 label  = "err = $err",
                 size   = (500, 400),
                 frame  = :box,
                 width  = 2);
    else
        plot!(p, sub.ftl_prop, sub.mean_R2;
              color = getcol(i),
              label = "err = $err",
              width = 2);
    end
end

#MAE
pmae = nothing;
for (i, err) in enumerate(ftl_error_list)

    sub = df_summary[df_summary.ftl_error .== err, :]

    if pmae === nothing
        pmae = plot(sub.ftl_prop, sub.mean_MAE;
                 xlabel = "Proportion TL known",
                 ylabel = "mean MAE",
                 color  = getcol(i),
                 size   = (500, 400),
                 frame = :box,
                 label = false,
                 width = 2);
    else
        plot!(pmae, sub.ftl_prop, sub.mean_MAE;
              color = getcol(i),
              label = false,
              width = 2);
    end
end

#WMAE
pwmae = nothing;
for (i, err) in enumerate(ftl_error_list)

    sub = df_summary[df_summary.ftl_error .== err, :]

    if pwmae === nothing
        pwmae = plot(sub.ftl_prop, sub.mean_WMAE;
                 xlabel = "Proportion TL known",
                 ylabel = "mean WMAE",
                 color  = getcol(i),
                 size   = (500, 400),
                 frame = :box,
                 label = false,
                 width = 2);
    else
        plot!(pwmae, sub.ftl_prop, sub.mean_WMAE;
              color = getcol(i),
              label = false,
              width = 2);
    end
end

#RMSE
prmse = nothing;
for (i, err) in enumerate(ftl_error_list)

    sub = df_summary[df_summary.ftl_error .== err, :]

    if prmse === nothing
        prmse = plot(sub.ftl_prop, sub.mean_RMSE;
                 xlabel = "Proportion TL known",
                 ylabel = "mean RMSE",
                 color  = getcol(i),
                 size   = (500, 400),
                 frame = :box,
                 label = false,
                 width = 2);
    else
        plot!(prmse, sub.ftl_prop, sub.mean_RMSE;
              color = getcol(i),
              label = false,
              width = 2);
    end
end

#WRMSE
pwrmse = nothing;
for (i, err) in enumerate(ftl_error_list)

    sub = df_summary[df_summary.ftl_error .== err, :]

    if pwrmse === nothing
        pwrmse = plot(sub.ftl_prop, sub.mean_WRMSE;
                 xlabel = "Proportion TL known",
                 ylabel = "mean WRMSE",
                 color  = getcol(i),
                 size   = (500, 400),
                 frame = :box,
                 label = false,
                 width = 2);
    else
        plot!(pwrmse, sub.ftl_prop, sub.mean_WRMSE;
              color = getcol(i),
              label = false,
              width = 2);
    end
end


#MEAN KL
pkl = nothing;
for (i, err) in enumerate(ftl_error_list)

    sub = df_summary[df_summary.ftl_error .== err, :]

    if pkl === nothing
        pkl = plot(sub.ftl_prop, sub.mean_KL;
                 xlabel = "Proportion TL known",
                 ylabel = "mean KL",
                 color  = getcol(i),
                 size   = (500, 400),
                 frame = :box,
                 label = false,
                 width = 2);
    else
        plot!(pkl, sub.ftl_prop, sub.mean_KL;
              color = getcol(i),
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


filename = smartpath("../figures/fig_trophiclevelerror_$(skew_setting).pdf")
Plots.savefig(combplot,filename)
