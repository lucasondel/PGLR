# Bayesian Logistic Regression using the Polya-Gamma augmentation.
#

using LinearAlgebra

export BinaryLogisticRegression
export _augment_posterior

#######################################################################
# Binary logistic regression

struct BinaryLogisticRegression <: Model
    β::ConjugateParameter{<:Normal}
    hasbias::Bool

    function BinaryLogisticRegression(β::ConjugateParameter{<:Normal}, hasbias::Bool)
        model = new(β, hasbias)
        β.accumulator = data -> accumulator_β(model, data...)
        return model
    end
end


function BinaryLogisticRegression(μ₀::AbstractVector, Σ₀::AbstractMatrix,
                                  μ::AbstractVector, Σ::AbstractMatrix,
                                  hasbias::Bool)
    BinaryLogisticRegression(ConjugateParameter(Normal(μ₀, Σ₀), Normal(μ, Σ)),
                             hasbias)
end

function BinaryLogisticRegression(T; inputdim::Integer, initσ::Real = 0.0,
                                  pseudocounts::Real = 1.0, hasbias::Bool = True)

    dim = hasbias ? inputdim + 1 : inputdim

    # Prior parameters
    μ₀ = zeros(T, dim)
    Σ₀ = (1/pseudocounts) * Matrix{T}(I, dim, dim)

    # Initial paremeterization of the posterior
    μ = μ₀ .+ initσ .* randn()
    Σ = deepcopy(Σ₀)

    BinaryLogisticRegression(μ₀, Σ₀, μ, Σ, hasbias)
end

function BinaryLogisticRegression(;inputdim::Integer, initσ::Real = 0.0,
                                  pseudocounts::Real = 1.0, hasbias::Bool = True)
    BinaryLogisticRegression(Float64, inputdim = inputdim, initσ = initσ,
                             pseudocounts = pseudocounts, hasbias = hasbias)
end

function regressors(model::BinaryLogisticRegression, X::AbstractMatrix)
    if model.hasbias
        # We add an extra dimension always set to "1" to incorporate a bias
        # parameter to the model.
        return vcat(X, ones(eltype(X), 1, size(X, 2)))
    end
    return X
end


# Return the Polya-Gamma posterior of the augmented
# logistic function
function _augment_posterior(ψ²::AbstractVector)
    b = ones(eltype(ψ²), length(ψ²))
    qω = PolyaGamma(b)
    update!(qω, -.5 * ψ²)
    return qω
end

# Return the sufficient statistics of the log likelihood and the
# the expected statistics of β.
function _precompute(model::BinaryLogisticRegression, X::AbstractMatrix,
                     z::AbstractVector)
    X̂ = regressors(model, X)

    # Quadratic expansion of the regressors
    D, N = size(X̂)
    X̂X̂ᵀ = reshape(X̂, D, 1, N) .* reshape(X̂, 1, D, N)
    vec_X̂X̂ᵀ = reshape(X̂X̂ᵀ, :, N)

    # Expectation of the regressors: E[ T(β) ] = E[ ( β, vec(ββᵀ) )]
    E_Tβ = gradlognorm(model.β.posterior)
    vec_E_ββᵀ = E_Tβ[D+1:end]

    # Augmentation of the logistic function
    pω = PolyaGamma(ones(eltype(X̂), size(X̂, 2)))
    ψ² = vec_X̂X̂ᵀ' * vec_E_ββᵀ
    qω = _augment_posterior(ψ²)
    E_ω = mean(qω)

    stats = vcat(reshape((z .- 0.5), 1, N) .* X̂,
                 -.5 .* reshape(E_ω, 1, :) .* vec_X̂X̂ᵀ)

    stats, E_Tβ, pω, qω
end

function (model::BinaryLogisticRegression)(X::AbstractMatrix, z::AbstractVector)
    stats, E_Tβ, pω, qω = _precompute(model, X, z)

    # KL divegerence betwen qω and pω for each sample.
    q_η, p_η = naturalparam(qω), naturalparam(pω)
    KL = lognorm(pω, perdim = true) - lognorm(qω, perdim = true) - (p_η .- q_η) .* gradlognorm(qω)

    stats' * E_Tβ .- KL .- log(2)
end

function accumulator_β(model::BinaryLogisticRegression, X::AbstractMatrix, z::AbstractVector)
    stats, E_Tβ, _, _ = _precompute(model, X, z)
    dropdims(sum(stats, dims=2), dims=2)
end

