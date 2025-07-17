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
# ------------------------------------------------------------------ #
# experiment constants
# ------------------------------------------------------------------ #
S_list  = collect(10:2:80)        # species richness values
C_list  = collect(0.015:0.005:0.1)   # connectance values
alpha_true = 0.5                   # diet breadth (fixed)
pct    = 0.0                       # fraction of links locked (fixed)
n_rep  = 50                        # replicates per (S,C) pair

ftl_prop      = 1.0;               # Assume perfect knowledge of ftls
ftl_error     = 0.0;

base_seed     = 20250624;
steps_sa      = 20_000;
wiggle_sa     = 0.05;

S_param   = repeat(S_list, inner = length(C_list)*n_rep)
C_param   = repeat(repeat(C_list, inner = n_rep), outer = length(S_list))
rep_param = repeat(collect(1:n_rep), outer = length(S_list)*length(C_list))

@assert length(S_param) == length(C_param) == length(rep_param)
Nruns = length(S_param)


###############################################################################
# pre-allocate result arrays (thread-safe, no push!)
###############################################################################
# pct_v   = Vector{Float64}(undef, Nruns);
rep_v   = Vector{Int}(undef, Nruns);
# alpha_v = Vector{Float64}(undef, Nruns);
r2_v    = Vector{Float64}(undef, Nruns);
mae_v    = Vector{Float64}(undef, Nruns);
wmae_v    = Vector{Float64}(undef, Nruns);
rmse_v   = Vector{Float64}(undef, Nruns);
wrmse_v   = Vector{Float64}(undef, Nruns);
meanKL_v = Vector{Float64}(undef, Nruns);

r2Q0_v = Vector{Float64}(undef, Nruns);
maeQ0_v = Vector{Float64}(undef, Nruns);
wmaeQ0_v = Vector{Float64}(undef, Nruns);
rmseQ0_v = Vector{Float64}(undef, Nruns);
wrmseQ0_v = Vector{Float64}(undef, Nruns);
meanKLQ0_v = Vector{Float64}(undef, Nruns);

S_v = Vector{Int}(undef, Nruns)
C_v = Vector{Float64}(undef, Nruns)

skew_setting = "rand";

###############################################################################
# threaded sweep
###############################################################################
@showprogress Threads.@threads for idx in 1:Nruns
    # decode flat index -> (S, C, rep)
    S   = S_param[idx]
    C   = C_param[idx]
    rep = rep_param[idx]

    rng = MersenneTwister(hash((base_seed, Threads.threadid(), idx)))

    # --- build web & true diets ------------------------------------------
    A, _       = nichemodelweb(S, C; rng = rng)
    A_bool     = (A .> 0)
    Q_true     = quantitativeweb(A;
                                alpha_dir = alpha_true,
                                method    = :rand,
                                rng       = rng)
    ftl_true   = trophic_levels(Q_true)

    ftl_obs = ftl_inference(ftl_true; ftl_prop = ftl_prop, ftl_error = ftl_error)
    
    # --- lock links ------------------------------------------------------
    mask       = select_known_links(Q_true, ftl_obs; pct = pct, skew = Symbol(skew_setting), rng = rng)

    # --- anneal ----------------------------------------------------------
    Q0 = make_prior_Q0(Q_true; deviation = 1.0, rng = rng);
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

    statsQ0 = evaluate_Q(Q_true, Q0; known_mask = mask, eps = 1e-12)
    r2_Q0    = statsQ0.r;
    mae_Q0   = statsQ0.mae;
    wmae_Q0   = statsQ0.wmae;
    rmse_Q0  = statsQ0.rmse;
    wrmse_Q0  = statsQ0.wrmse;
    meanKL_Q0 = statsQ0.mean_KL;

    # alpha_v[idx]   = alpha_true;
    # pct_v[idx] = pct;
    rep_v[idx] = rep;
    r2_v[idx]  = r2_Q;
    mae_v[idx] = mae_Q;
    wmae_v[idx] = wmae_Q;
    rmse_v[idx] = rmse_Q;
    wrmse_v[idx] = wrmse_Q;
    meanKL_v[idx] = meanKL_Q;

    r2Q0_v[idx]  = r2_Q0;
    maeQ0_v[idx] = mae_Q0;
    wmaeQ0_v[idx] = wmae_Q0;
    rmseQ0_v[idx] = rmse_Q0;
    wrmseQ0_v[idx] = wrmse_Q0;
    meanKLQ0_v[idx] = meanKL_Q0;

    S_v[idx] = S
    C_v[idx] = C
end

# #save data file
filename = smartpath("../data/infogain_SC.jld")
@save filename S_list C_list n_rep steps_sa wiggle_sa base_seed skew_setting S_v C_v rep_v r2_v mae_v wmae_v rmse_v wrmse_v meanKL_v r2Q0_v maeQ0_v wmaeQ0_v rmseQ0_v wrmseQ0_v meanKLQ0_v 


filename = smartpath("../data/infogain_SC.jld")
@load filename S_list C_list n_rep steps_sa wiggle_sa base_seed skew_setting S_v C_v rep_v r2_v mae_v wmae_v rmse_v wrmse_v meanKL_v r2Q0_v maeQ0_v wmaeQ0_v rmseQ0_v wrmseQ0_v meanKLQ0_v 

###############################################################################
# build DataFrame, dropping the NaN rows
###############################################################################

df     = DataFrame(S = S_v,
                  C = C_v,
                  rep = rep_v,
                  R2 = r2_v,
                  MAE = mae_v,
                  WMAE = wmae_v,
                  RMSE = rmse_v,
                  WRMSE = wrmse_v,
                  meanKL = meanKL_v,
                  R2Q0 = r2Q0_v,
                  MAEQ0 = maeQ0_v,
                  WMAEQ0 = wmaeQ0_v,
                  RMSEQ0 = rmseQ0_v,
                  WRMSEQ0 = wrmseQ0_v,
                  meanKLQ0 = meanKLQ0_v,);


df_summary = combine(
  DataFrames.groupby(df, [:S, :C]),

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

  # R²: drop any non-finite values
  :R2Q0    => (x -> mean(filter(isfinite,    x))) => :mean_R2Q0,
  :R2Q0    => (x -> std(filter(isfinite,    x))) => :sd_R2Q0,

  # MAE: drop any missing values
  :MAEQ0   => (x -> mean(filter(!ismissing,   x))) => :mean_MAEQ0,
  :MAEQ0   => (x -> std(filter(!ismissing,   x))) => :sd_MAEQ0,

  :WMAEQ0   => (x -> mean(filter(!ismissing,   x))) => :mean_WMAEQ0,
  :WMAEQ0   => (x -> std(filter(!ismissing,   x))) => :sd_WMAEQ0,

  # RMSE: same as MAE
  :RMSEQ0  => (x -> mean(filter(!ismissing,   x))) => :mean_RMSEQ0,
  :RMSEQ0  => (x -> std(filter(!ismissing,   x))) => :sd_RMSEQ0,

  :WRMSEQ0  => (x -> mean(filter(!ismissing,   x))) => :mean_WRMSEQ0,
  :WRMSEQ0  => (x -> std(filter(!ismissing,   x))) => :sd_WRMSEQ0,

  # meanKL: same pattern
  :meanKLQ0=> (x -> mean(filter(!ismissing,   x))) => :mean_KLQ0,
  :meanKLQ0=> (x -> std(filter(!ismissing,   x))) => :sd_KLQ0,
)


println("\nMean ± SD of R² on UNKNOWN links")
show(df_summary, allrows = true, allcols = true)


RMSEratio = df_summary.mean_RMSE ./ df_summary.mean_RMSEQ0
df_sub = DataFrame(S = df_summary.S, C = df_summary.C, RMSEratio = RMSEratio)

# 1.  Extract sorted, unique axis values
Svals = sort(unique(df_sub.S))          # e.g. [25, 30, …, 100]
Cvals = sort(unique(df_sub.C))          # e.g. [0.02, 0.04, …, 0.20]
# 2.  Pivot (C as rows, S as columns) – unstack does the heavy lifting
wide = unstack(df_sub, :C, :S, :RMSEratio)
scol_syms   = Symbol.(string.(Svals))              #  :”25” :”30” … :”100”
wide_sorted = select(wide, [:C; scol_syms]...)     # keep :C first, rest in order
# 3.  Convert everything except the first column (C) to a matrix
mat = Matrix(wide_sorted[:, Not(:C)])

# 4.  Plot – the length of Cvals must equal size(mat,1) and Svals size(mat,2)
plt = Plots.heatmap(
   string.(Svals),          # x tick labels  – matches matrix columns
    string.(Cvals),          # y tick labels  – matches matrix rows
    mat,                       # z-value matrix
    xlabel = "Species, S",
    ylabel = "Connectance, C",
    colorbar_title = "RMSE ratio",
    title = "RMSE ratio across (S, C)",
    aspect_ratio = :equal,     # square cells
    c = :viridis               # any palette you like
)
# 2. overlay contour where value == 1
contour!(
    plt,
    string.(Svals),          # x tick labels  – matches matrix columns
    string.(Cvals), mat;
    levels = [1.0],          # draw only the z = 1 isopleth
    linewidth = 2,
    linecolor = :white,      # or :black, :red, etc.
    linestyle = :solid
)

display(plt)



###############################################################################
# plot three curves with UnicodePlots
###############################################################################

#R2
p = nothing;
for S in S_list
    sub = df_summary[df_summary.S .== S, :]

    if p === nothing
        p = plot(sub.C, sub.mean_R2;
                 xlabel = "Connectance (C)",
                 ylabel = "mean R²",
                 label  = "S = $S",
                 size   = (500, 400),
                 frame = :box,
                 width = 2);
    else
        plot!(p, sub.C, sub.mean_R2;
              label = "S = $S",
              width = 2);
    end
end

#MAE
pmae = nothing;
for S in S_list
    sub = df_summary[df_summary.S .== S, :]

    if pmae === nothing
        pmae = plot(sub.C, sub.mean_MAE;
                 xlabel = "Connectance (C)",
                 ylabel = "mean MAE",
                 size   = (500, 400),
                 frame = :box,
                 label = false,
                 width = 2);
    else
        plot!(pmae, sub.C, sub.mean_MAE;
              label = false,
              width = 2);
    end
end

#WMAE
pwmae = nothing;
for S in S_list
    sub = df_summary[df_summary.S .== S, :]

    if pwmae === nothing
        pwmae = plot(sub.C, sub.mean_WMAE;
                 xlabel = "Connectance (C)",
                 ylabel = "mean MAE",
                 size   = (500, 400),
                 frame = :box,
                 label = false,
                 width = 2);
    else
        plot!(pwmae, sub.C, sub.mean_WMAE;
              label = false,
              width = 2);
    end
end

#RMSE
prmse = nothing;
for S in S_list
    sub = df_summary[df_summary.S .== S, :]

    if prmse === nothing
        prmse = plot(sub.C, sub.mean_RMSE;
                 xlabel = "Connectance (C)",
                 ylabel = "mean RMSE",
                 size   = (500, 400),
                 frame = :box,
                 label = false,
                 width = 2);
    else
        plot!(prmse, sub.C, sub.mean_RMSE;
              label = false,
              width = 2);
    end
end

#WRMSE
pwrmse = nothing;
for S in S_list
    sub = df_summary[df_summary.S .== S, :]

    if pwrmse === nothing
        pwrmse = plot(sub.C, sub.mean_WRMSE;
                 xlabel = "Connectance (C)",
                 ylabel = "mean RMSE",
                 size   = (500, 400),
                 frame = :box,
                 label = false,
                 width = 2);
    else
        plot!(pwrmse, sub.C, sub.mean_WRMSE;
              label = false,
              width = 2);
    end
end


#MEAN KL
pkl = nothing;
for S in S_list
    sub = df_summary[df_summary.S .== S, :]

    if pkl === nothing
        pkl = plot(sub.C, sub.mean_KL;
                 xlabel = "Connectance (C)",
                 ylabel = "mean KL",
                 size   = (500, 400),
                 frame = :box,
                 label = false,
                 width = 2);
    else
        plot!(pkl, sub.C, sub.mean_KL;
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


filename = smartpath("../figures/fig_SCknown_$(skew_setting).pdf")
Plots.savefig(combplot,filename)
