function quantitativeweb(A::AbstractArray;
                         alpha_dir::Real = 1.0,
                         method::Symbol = :rand,
                         nichevalues::AbstractVector = [],
                         rohr_params::Tuple = (),
                         rng::AbstractRNG = Random.GLOBAL_RNG)


    S, _ = size(A)
    Q    = zeros(Float64, S, S)

    if method == :rand

        for j in 1:S
            prey = findall(!iszero, A[:, j])
            f    = length(prey)
            f == 0 && continue

            weights = rand(rng, Dirichlet(fill(alpha_dir, f)))  # one line does it all
            Q[prey, j] .= weights
        end

    elseif method == :allometric
        isempty(nichevalues) &&
            throw(ArgumentError("nichevalues must be supplied when method = :allometric"))
        isempty(rohr_params) &&
            throw(ArgumentError("rohr_params (α̂,β̂,γ̂) must be supplied when method = :allometric"))

        mass = nichevalues

        α̂, β̂, γ̂ = rohr_params #-1.1777, 0.4126, -0.0239

        for j in 1:S
            prey = findall(!iszero, A[:, j])
            isempty(prey) && continue

            ratio   = log.(mass[prey] ./ mass[j])                       # flipped!
            weights = @. 1 / (1 + exp(-(α̂ + β̂*ratio + γ̂*ratio^2)))      # logit
            weights ./= sum(weights)                                    # safe: prey non-empty
            Q[prey, j] .= weights
        end

    else
        throw(ArgumentError("method must be :rand or :allometric"))
    end

    return Q

end

