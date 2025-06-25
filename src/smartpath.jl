function smartpath(filename::AbstractString; indices::Union{Nothing, AbstractVector{<:Integer}}=nothing)
    # Path to data directory, relative to package root
    main_dir = joinpath(@__DIR__, "..")

    # Remove extension if present, save it for later
    ext = splitext(filename)[2]
    filename_cut = splitext(filename)[1]

    # Compose index string if provided
    indexstring = indices === nothing ? "" : "_" * join(indices, "_")

    # Construct full filename with extension
    fname = filename_cut * indexstring * ext

    # Return full path
    return joinpath(main_dir, fname)
end
