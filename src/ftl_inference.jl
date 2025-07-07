function ftl_inference(ftl_true::AbstractVector;
                       ftl_prop::Real = 1.0,
                       ftl_error::Real = 0.0)

    # index of first consumer (ftl > 1)
    posfirstconsumer = findfirst(>(1.0), ftl_true)

    #Edge case - shouldn't ever happen
    if posfirstconsumer === nothing
        return copy(ftl_true)          # retruns ftl_true if no consumers to observe
    end

    consumer_idx = posfirstconsumer:length(ftl_true)
    n_cons       = length(consumer_idx)

    # how many consumers are observed?
    numdraws = clamp(round(Int, ftl_prop * n_cons), 0, n_cons)

    # sample observed consumers
    posdraws     = sort(sample(consumer_idx, numdraws; replace = false))
    posunobserved = setdiff(consumer_idx, posdraws)

    # observation error
    errorvec = rand(Normal(0, ftl_error), length(ftl_true))
    errorvec[1:posfirstconsumer-1] .= 0.0               # producers measured exactly

    # build observed vector
    ftl_obs = copy(ftl_true)
    ftl_obs[posdraws]      .+= errorvec[posdraws]        # noisy draws
    ftl_obs[posunobserved] .= NaN                       # missing consumers

    # clamp below 1.0 (unlikely after NaN assignment, but safe)
    return clamp.(ftl_obs, 1.0, Inf)
end


# function ftl_inference(ftl_true::AbstractVector;
#     ftl_prop::Real  = 1.0,
#     ftl_error::Real  = 0.0)

#     # First position with ftl > 1.0
#     posfirstconsumer = findfirst(>(1.0), ftl_true)
#     @assert posfirstconsumer !== nothing "ftl_true has no consumers (ftl > 1)"

#     ftl_true_consumer = ftl_true[posfirstconsumer:end]

#     l_ftl = length(ftl_true);
#     l_ftlcons = length(ftl_true_consumer);
#     #Number of observed values
#     numdraws = round(Int, ftl_prop * l_ftlcons)

#     # Sample without replacement
#     posdraws = sort(sample(posfirstconsumer:l_ftl, numdraws; replace = false))

#     posunobserved = setdiff(collect(posfirstconsumer:l_ftl),posdraws)

#     # Observed error Distribution
#     errordist = Normal(0,ftl_error)
#     errorvec = rand.(errordist,l_ftl)
#     errorvec[1:(posfirstconsumer - 1)] = repeat([0.0],outer = (posfirstconsumer - 1))


#     # Create observed values
#     ftl_obs = copy(ftl_true);
#     ftl_obs[posdraws] = ftl_obs[posdraws] .+ errorvec[posdraws]
    
#     if length(posunobserved) > 0
#         ftl_obs[posunobserved] = repeat([NaN],outer = length(posunobserved))
#     end


#     #We can't have estimates below 1.0 so clmap at (1.0, Infinity)
#     ftl_obs = clamp.(ftl_obs,1.0,Inf)

#     return ftl_obs

# end