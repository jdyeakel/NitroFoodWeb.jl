function rohr_fc(xmax,A,massvec)

    alphamax = xmax[1];
    betamax = xmax[2];
    gammamax = xmax[3];

    numprey = size(A)[1];
    numpred = size(A)[2];
    massvec_pred = massvec;
    massvec_prey = massvec;

    Apredict = copy(A).*0.0;

    numpreyperpred = vec(sum(A,dims=1));
    consumers = findall(!iszero,numpreyperpred);

    # global cumnum = 0.0;
    # global cumdenom = 0.0;
    # cnumerator = Array{Float64}(undef,1);
    # cdenominator = Array{Float64}(undef,1);
    let cumnum = 0.0, cumdenom = 0.0
        for i=consumers
        # for i=1:numpred
            for j=1:numprey

                # if i != j

                    aij = copy(A[j,i]);
                    mi = massvec_prey[j]; #j is prey
                    mj = massvec_pred[i]; #i is predator
                    
                    pij = exp(alphamax + betamax*log(mi/mj) + gammamax*log(mi/mj)^2)/(1+exp(alphamax + betamax*log(mi/mj) + gammamax*log(mi/mj)^2));
                    
                    num = aij*pij + (1-aij)*(1-pij);
                    denom = aij + (1-aij);
                    
                    Apredict[j,i] = num;
                    
                    cumnum += num;
                    cumdenom += denom;

                
                # end

            end
        end
        global cnumerator = copy(cumnum);
        global cdenominator = copy(cumdenom);
        # global cratio = copy(ratio);
    end
    
    fractioncorrect = cnumerator/cdenominator;
    # fractioncorrect = copy(cratio);
    
    return fractioncorrect, Apredict

end