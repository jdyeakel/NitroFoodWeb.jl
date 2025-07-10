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

Afilename = smartpath("../data/foodweb/bengula_A_matrix.csv")
Adata_bg = CSV.read(Afilename,header=true,DataFrame);
numsp = size(Adata_bg)[1];
Mfilename = smartpath("../data/foodweb/bengula_masses.csv")
massdata_bg = CSV.read(Mfilename,header=true,DataFrame);

## BENGUELA
#Use species 5-29 (species 1-4 have missing diet data)
Aprime_bg = Matrix(Adata_bg[1:numsp,2:numsp+1]);
# nolinks = findall(iszero,sum(Aprime_bg,dims=1));
# A_bg = Matrix(Adata[5:29,6:30]);
A_bg = copy(Aprime_bg);
# massvec_bg = massdata[:mass][5:numsp];
massvec_bg = copy(vec(massdata_bg[!,:mass_kg]));
#Sort by body size
# sortsp_bg = sortperm(massvec_bg);
# Asort_bg = A_bg[sortsp_bg,sortsp_bg];



## BENGUELA
x0 = [0.0,0.0,0.0];
results_bg = optimize(x->rohr_lfunc(x,A_bg,massvec_bg),x0,NelderMead());
results_bg.minimizer
xmax_bg = results_bg.minimizer;
fcorr_bg, Apredict_bg = rohr_fc(xmax_bg,A_bg,massvec_bg)
