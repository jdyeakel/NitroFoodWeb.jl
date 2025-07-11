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

using Test

S = 100;
C = 0.06;

# Build adjacency matrix
A, niche = nichemodelweb(S,C)
A_bool = A .> 0
# Get fractional trophic levels
tl = TrophInd(A)

tlalt = trophic_levels(A)

maximum(abs.(tl - tlalt))

#now try weighted
Q = quantitativeweb(A; alpha=0.5)

@time ftl = TrophInd(Q)
@time ftlalt = trophic_levels(Q)

maximum(abs.(ftl - ftlalt))

#NOTE: trophic_levels produces the same result and is much faster.