
# InternalNetwork function in Julia
function InternalNetwork(Tij::AbstractMatrix{Float64}, node_names::Vector{String};
                         Import::Union{Vector{Float64}, Nothing}=nothing,
                         Export::Union{Vector{Float64}, Nothing}=nothing)

    # Calculate Import if not provided
    if Import === nothing
        Import = sum(Tij, dims=2) .- sum(Tij, dims=1)'
        Import[Import .< 0] .= 0
    end
    # Calculate Export if not provided
    if Export === nothing
        Export = sum(Tij, dims=1)' .- sum(Tij, dims=2)
        Export[Export .< 0] .= 0
    end
    N = Dict{Symbol, Any}()
    N[:Tint] = Tij
    N[:Import] = Import
    N[:Export] = Export
    N[:TotIn] = Import .+ sum(Tij, dims=1)'
    N[:TotOut] = Export .+ sum(Tij, dims=2)
    N[:iN] = findall((vec(N[:TotIn]) .> 0) .|| (vec(N[:TotOut]) .> 0))
    N[:Tint] = N[:Tint][N[:iN], N[:iN]]
    N[:Import] = N[:Import][N[:iN]]
    N[:Export] = N[:Export][N[:iN]]
    N[:TotIn] = N[:TotIn][N[:iN]]
    N[:TotOut] = N[:TotOut][N[:iN]]
    N[:node_names] = node_names[N[:iN]]
    return N
end

# Diet function in Julia
function Diet(Tint::AbstractMatrix{Float64}, Dead::Union{Vector{Int}, Nothing}=nothing)

    ncomp = size(Tint, 2)
    p = zeros(ncomp, ncomp)
    for i in 1:ncomp
        totin = sum(Tint[:, i])
        if totin > 0
            p[i, :] = (Tint[:, i] / totin)'
        end
    end
    if Dead !== nothing
        p[Dead, :] .= 0
    end
    return p
end

# TrophInd function in Julia
# TrophInd function in Julia with keyword arguments
function TrophInd(Flow::AbstractMatrix{Float64};
    node_names::Union{Vector{String}, Nothing}=nothing,
    Import::Union{Vector{Float64}, Nothing}=nothing,
    Export::Union{Vector{Float64}, Nothing}=nothing,
    Dead::Union{Vector{Int}, Vector{String}, Nothing}=nothing)

    # Determine the number of nodes
    num_nodes = size(Flow, 1)

    # If node_names is not provided, generate default names
    if node_names === nothing
        node_names = ["Sp_$(i)" for i in 1:num_nodes]
    end

    # Transpose Flow to get Tij
    Tij = Flow
    # Process Dead nodes
    if Dead isa Vector{String}
        dead = findall(name -> name in Dead, node_names)
    elseif Dead isa Vector{Int}
        dead = Dead
    else
        dead = Int[]
    end
    if Dead !== nothing && length(dead) != length(Dead)
    error("Dead not recognized")
    end
    # Call InternalNetwork with keyword arguments
    N = InternalNetwork(Tij, node_names; Import=Import, Export=Export)
    # Call Diet
    p = Diet(N[:Tint], dead)
    ncomp = size(N[:Tint], 2)
    # Compute A matrix
    A = -p
    for i in 1:ncomp
    A[i, i] = 1
    end
    # Compute B vector
    B = ones(ncomp)
    # Compute Trophic Levels (TL)
    TL = pinv(A) * B
    # Compute Omnivory Index (OI)
    OI = zeros(ncomp)
    for i in 1:ncomp
    OI[i] = sum((TL .- (TL[i] - 1)).^2 .* p[i, :])
    end
    # Return DataFrame with results
    tldf = DataFrame(Node = N[:node_names], TL = TL, OI = OI);
    tl = round.(tldf[!,:TL]; sigdigits=6) 
    return tl
end