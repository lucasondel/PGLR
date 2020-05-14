module BayesianModels

using ExpFamilyDistributions

export Model
export BayesianParameter
export AbstractConjugateParameter
export ConjugateParameter
export HierarchicalConjugateParameter
export elbo
export getconjugateparams



"""
    BayesianParameter

Supertype for all Bayesian parameters, i.e. parameter having a prior
and a posterior.
"""
abstract type BayesianParameter end


"""
    AbstractConjugateParameter <: BayesianParameter

Bayesian parameter having the same type of prior and posterior.
"""
abstract type AbstractConjugateParameter <: BayesianParameter end

Base.show(io::IO, p::AbstractConjugateParameter) = print(io, "$(typeof(p))")



"""
    ConjugateParameter <: AbstractConjugateParameter

Standard implementation of a conjugate parameter.
"""
mutable struct ConjugateParameter{T <: ExpFamilyDistribution} <: AbstractConjugateParameter
    prior::T
    posterior::T
    accumulator::Function

    function ConjugateParameter(prior::ExpFamilyDistribution,
                                post::ExpFamilyDistribution)
        new{typeof(post)}(prior, post)
    end
end


"""
    getconjugateparams(model)

Return the list of the conjugate parameters of an object.
"""
function getconjugateparams(model)
    params = []
    for name in fieldnames(typeof(model))
        field = getfield(model, name)
        if typeof(field) <: ConjugateParameter
            push!(params, field)
        end
    end
    return params
end

"""
    HierarchicalConjugateParameter <: AbstractConjugateParameter
Σ
Conjugate parameter whose prior depends on another parameter.
"""
mutable struct HierarchicalConjugateParameter{T <: ExpFamilyDistribution} <: AbstractConjugateParameter
    getprior::Function
    posterior::T
    accumulator::Function

    function HierarchicalConjugateParameter(getprior::Function,
                                            post::ExpFamilyDistribution)
        new{typeof(post)}(getprior, post)
    end
end


function Base.getproperty(p::HierarchicalConjugateParameter, sym::Symbol)
    if sym ≡ :prior
        return p.getprior()
    else
        return getfield(p, sym)
    end
end


"""
    Model

Supertype for JuBeer models.
"""
abstract type Model end

# Imitate pytorch printing style of model.
function Base.show(io::IO, model::Model; pad = 0)
    originalpad = pad
    modelname = "$(typeof(model))("
    println(io, modelname)
    pad += 2
    for name in propertynames(model)
        pval = getproperty(model, name)
        if isa(pval, Model)
            pstr = "($(name)): "
            print(io, lpad(pstr, length(pstr) + pad))
            Base.show(io, pval, pad=pad)
        elseif isa(pval, BayesianParameter)
            pstr = "($(name)): $(pval)"
            println(io, lpad(pstr, length(pstr) + pad))
        end
    end
    println(io, lpad(")", 1 + originalpad))
    return nothing
end

"""
    elbo(model)

Compute the evidence lower-bound of the model.
"""
function elbo(model::Model, X...)
    KL = sum([kldiv(param.posterior, param.prior)
              for param in getconjugateparams(model)])
    return sum(model(X...)) - KL
end

include("LogisticRegression.jl")

# Helper functions for plotting parameters of the models.
include("Plotting.jl")

end # BayesianModels

