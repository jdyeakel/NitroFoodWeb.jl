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
###############################################################################
# experiment constants
###############################################################################
S, C          = 100, 0.02;
alpha_list    = [0.5,1.0,10.0];          # single diet breadth to test
pct_grid      = 0.0:0.05:0.50;             # fraction of links locked
n_rep         = 50;

ftl_prop      = 1.0;                       # Assume perfect knowledge of ftls
ftl_error     = 0.0;

steps_sa      = 20_000;
wiggle_sa     = 0.05;
# ΔTN           = 3.5;
base_seed     = 20250624;

skew_setting  = "apexrand";

alpha_param = repeat(alpha_list, inner = length(pct_grid)*n_rep)
pct_param   = repeat(repeat(pct_grid, inner = n_rep), outer = length(alpha_list))
rep_param   = repeat(collect(1:n_rep),  outer = length(alpha_list)*length(pct_grid))

@assert length(alpha_param) == length(pct_param) == length(rep_param)
Nruns = length(alpha_param)


###############################################################################
# pre-allocate result arrays (thread-safe, no push!)
###############################################################################
pct_v   = Vector{Float64}(undef, Nruns);
rep_v   = Vector{Int}(undef, Nruns);
alpha_v = Vector{Float64}(undef, Nruns);
r2_v    = Vector{Float64}(undef, Nruns);
mae_v    = Vector{Float64}(undef, Nruns);
wmae_v    = Vector{Float64}(undef, Nruns);
rmse_v   = Vector{Float64}(undef, Nruns);
wrmse_v   = Vector{Float64}(undef, Nruns);
meanKL_v = Vector{Float64}(undef, Nruns);

###############################################################################
# threaded sweep
###############################################################################
@showprogress Threads.@threads for idx in 1:Nruns
    # decode flat index -> (ia , ipct , rep)
    # rep  = (idx-1)              % n_rep                + 1
    # ipct = ((idx-1) ÷ n_rep)    % length(pct_grid)     + 1
    # ia   = ((idx-1) ÷ (n_rep*length(pct_grid)))        + 1

    # alpha_true = alpha_list[ia]
    # pct        = pct_grid[ipct]

    alpha_true = alpha_param[idx]
    pct    = pct_param[idx]
    rep    = rep_param[idx]


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

    alpha_v[idx]   = alpha_true;
    pct_v[idx] = pct;
    rep_v[idx] = rep;
    r2_v[idx]  = r2_Q;
    mae_v[idx] = mae_Q;
    wmae_v[idx] = wmae_Q;
    rmse_v[idx] = rmse_Q;
    wrmse_v[idx] = wrmse_Q;
    meanKL_v[idx] = meanKL_Q;
end

#save data file
filename = smartpath("../data/alphaknown_setalpha_$(skew_setting).jld")
@save filename S C alpha_list pct_grid n_rep steps_sa wiggle_sa ftl_prop ftl_error base_seed skew_setting alpha_v pct_v rep_v r2_v mae_v rmse_v meanKL_v



skew_setting = :randsp;
filename = smartpath("../data/alphaknown_setalpha_$(skew_setting).jld")
@load filename S C alpha_list pct_grid n_rep steps_sa wiggle_sa ftl_prop ftl_error base_seed skew_setting alpha_v pct_v rep_v r2_v mae_v rmse_v meanKL_v



###############################################################################
# build DataFrame, dropping the NaN rows
###############################################################################

df     = DataFrame(alpha = alpha_v,
                  pct   = pct_v,
                  rep   = rep_v,
                  R2    = r2_v,
                  MAE   = mae_v,
                  WMAE  = wmae_v,
                  RMSE  = rmse_v,
                  WRMSE = wrmse_v,
                  meanKL = meanKL_v);


df_summary = combine(
  DataFrames.groupby(df, [:alpha, :pct]),

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

#R2
p = nothing;
for α in alpha_list
    sub = df_summary[df_summary.alpha .== α, :]

    if p === nothing
        p = plot(sub.pct, sub.mean_R2;
                 xlabel = "Prop. links known",
                 ylabel = "mean R²",
                 label  = "α = $α",
                 size   = (500, 400),
                 frame = :box,
                 width = 2);
    else
        plot!(p, sub.pct, sub.mean_R2;
              label = "α = $α",
              width = 2);
    end
end

#MAE
pmae = nothing;
for α in alpha_list
    sub = df_summary[df_summary.alpha .== α, :]

    if pmae === nothing
        pmae = plot(sub.pct, sub.mean_MAE;
                 xlabel = "Prop. links known",
                 ylabel = "mean MAE",
                 size   = (500, 400),
                 frame = :box,
                 label = false,
                 width = 2);
    else
        plot!(pmae, sub.pct, sub.mean_MAE;
              label = false,
              width = 2);
    end
end

#WMAE
pwmae = nothing;
for α in alpha_list
    sub = df_summary[df_summary.alpha .== α, :]

    if pwmae === nothing
        pwmae = plot(sub.pct, sub.mean_WMAE;
                 xlabel = "Prop. links known",
                 ylabel = "mean WMAE",
                 size   = (500, 400),
                 frame = :box,
                 label = false,
                 width = 2);
    else
        plot!(pwmae, sub.pct, sub.mean_WMAE;
              label = false,
              width = 2);
    end
end

#RMSE
prmse = nothing;
for α in alpha_list
    sub = df_summary[df_summary.alpha .== α, :]

    if prmse === nothing
        prmse = plot(sub.pct, sub.mean_RMSE;
                 xlabel = "Prop. links known",
                 ylabel = "mean RMSE",
                 size   = (500, 400),
                 frame = :box,
                 label = false,
                 width = 2);
    else
        plot!(prmse, sub.pct, sub.mean_RMSE;
              label = false,
              width = 2);
    end
end

#WRMSE
pwrmse = nothing;
for α in alpha_list
    sub = df_summary[df_summary.alpha .== α, :]

    if pwrmse === nothing
        pwrmse = plot(sub.pct, sub.mean_WRMSE;
                 xlabel = "Prop. links known",
                 ylabel = "mean WRMSE",
                 size   = (500, 400),
                 frame = :box,
                 label = false,
                 width = 2);
    else
        plot!(pwrmse, sub.pct, sub.mean_WRMSE;
              label = false,
              width = 2);
    end
end


#MEAN KL
pkl = nothing;
for α in alpha_list
    sub = df_summary[df_summary.alpha .== α, :]

    if pkl === nothing
        pkl = plot(sub.pct, sub.mean_KL;
                 xlabel = "Prop. links known",
                 ylabel = "mean KL",
                 size   = (500, 400),
                 frame = :box,
                 label = false,
                 width = 2);
    else
        plot!(pkl, sub.pct, sub.mean_KL;
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

