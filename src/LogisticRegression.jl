# Bayesian Logistic Regression using the Polya-Gamma augmentation.
#

using LinearAlgebra

export BinaryLogisticRegression
export LogisticRegression
export predict

#######################################################################
# Helper functions
#

function regressors(hasbias::Bool, X::Matrix{T}) where T <: AbstractFloat
    if hasbias
        # We add an extra dimension always set to "1" to incorporate a bias
        # parameter to the model.
        return vcat(X, ones(T, 1, size(X, 2)))
    end
    return X
end

# Return the Polya-Gamma prior and posterior of the augmented
# logistic function.
function pgaugmentation(ψ²::Vector{T}) where T <: AbstractFloat
    # Default parameterization is b = 1, c = 0
    pω = PolyaGamma{T, length(ψ²)}()

    b = ones(T, length(ψ²))
    qω = PolyaGamma(b, sqrt.(ψ²))

    return pω, qω
end

# KL divergence between N pairs of independent PG distributions
function kldivperdim(qω::PolyaGamma{T, N}, pω::PolyaGamma{T, N}) where {T <: AbstractFloat, N}
    pη, qη = naturalparam(pω), naturalparam(qω)
    lognorm(pω, perdim = true) - lognorm(qω, perdim = true) - (pη .- qη) .*gradlognorm(qω)
end

#######################################################################
# Binary logistic regression

struct BinaryLogisticRegression <: Model
    β::ConjugateParameter{<:Normal}
    hasbias::Bool

    function BinaryLogisticRegression(β::ConjugateParameter{<:Normal}, hasbias::Bool)
        model = new(β, hasbias)
        β.stats = data -> stats_β(model, data...)
        return model
    end
end


function BinaryLogisticRegression(μ₀::Vector{T}, Σ₀::Matrix{T},
                                  μ::Vector{T}, Σ::Matrix{T},
                                  hasbias::Bool) where T <: AbstractFloat
    BinaryLogisticRegression(ConjugateParameter(Normal(μ₀, Σ₀), Normal(μ, Σ)),
                             hasbias)
end

function BinaryLogisticRegression(T = Float64; inputdim::Integer, initσ::Real = 0.0,
                                  pseudocounts::Real = 1.0, hasbias::Bool = true)
    dim = hasbias ? inputdim + 1 : inputdim

    # Prior parameters
    μ₀ = zeros(T, dim)
    Σ₀ = T(1/pseudocounts) * Matrix{T}(I, dim, dim)

    # Initial paremeterization of the posterior
    μ = μ₀ .+ T(initσ) .* randn(T)
    Σ = deepcopy(Σ₀)

    BinaryLogisticRegression(μ₀, Σ₀, μ, Σ, hasbias)
end

function (model::BinaryLogisticRegression)(
    X::Matrix{T1},
    z::Vector{T2},
    Nₖ::Union{Vector{T2}, Nothing} = nothing
) where {T1 <: AbstractFloat, T2 <: Real}

    X̂ = regressors(model.hasbias, X)

    # Expectation of the parameters β
    μ, Σ = stdparam(model.β.posterior)
    E_β = μ
    E_ββᵀ = Σ + μ * μ'

    # Pre-compute the terms of the lower-bound
    zstats = isnothing(Nₖ) ? (z .- 0.5) : z .- Nₖ * 0.5
    E_βᵀX̂ = X̂' * E_β
    X̂ᵀE_ββᵀX̂ = dropdims(sum(X̂ .* (E_ββᵀ * X̂), dims = 1), dims = 1)

    # Polya-Gamma augmentation: compute the prior/posterior over ω
    pω, qω = pgaugmentation(X̂ᵀE_ββᵀX̂)
    E_ω = mean(qω)

    # KL divegerence betwen qω and pω for each sample.
    KL = kldivperdim(qω, pω)

    zstats .* E_βᵀX̂ .-.5 .* E_ω .* X̂ᵀE_ββᵀX̂ .- log(2) .- KL
end

function stats_β(model::BinaryLogisticRegression,
    X::Matrix{T1},
    z::Vector{T2},
    Nₖ::Union{Vector{T2}, Nothing} = nothing
) where {T1 <: AbstractFloat, T2 <: Real}

    X̂ = regressors(model.hasbias, X)

    # Expectation of the parameters β
    μ, Σ = stdparam(model.β.posterior)
    E_β = μ
    E_ββᵀ = Σ + μ * μ'

    # Polya-Gamma augmentation: compute the prior/posterior over ω
    X̂ᵀE_ββᵀX̂ = dropdims(sum(X̂ .* (E_ββᵀ * X̂), dims = 1), dims = 1)
    pω, qω = pgaugmentation(X̂ᵀE_ββᵀX̂)
    E_ω = mean(qω)

    # 1st order statistics
    zstats = isnothing(Nₖ) ? (z .- 0.5) : z .- Nₖ * 0.5
    s1 = zeros(T1, size(X̂, 1))
    for i = 1:length(z)
        s1 .+= X̂[:, i] * zstats[i]
    end

    # 2nd order statistics.
    s2 = zeros(T1, size(X̂, 1), size(X̂, 1))
    for i = 1:length(z)
        x̂ᵢ = X̂[:, i]
        s2 .+= (E_ω[i] * x̂ᵢ) * x̂ᵢ'
    end
    s2 .*= -.5

    vcat(s1, vec(s2))
end

# Predict the classes using the Maximum A Posteriori (MAP) parameters
function predict_map(model::BinaryLogisticRegression, X::Matrix{T}) where T <: AbstractFloat

    X̂ = regressors(model.hasbias, X)
    μ, _ = stdparam(model.β.posterior)
    y = X̂' * μ
    1 ./ (1 .+ exp.(-y))
end

# Predict the classes the Posteriori (MAP) parameters
function predict_marginal(model::BinaryLogisticRegression,
                 X::Matrix{T}; a::T = 0.368) where T <: AbstractFloat
    # Following https://arxiv.org/pdf/1703.00091.pdf, we use the
    # approximation:
    # ⟨σ(ψ)⟩ ≈ σ(μ / √(1 + a * σ²))
    # where a = 0.368
    # In Bishop 4.5.2 a very similar approximation uses a = π / 8 ≈ 0.392...

    X̂ = regressors(model.hasbias, X)
    μ, Σ = stdparam(model.β.posterior)
    E_ββᵀ = Σ + μ * μ'

    ψ_μ = X̂' * μ
    ψ² = dropdims(sum(X̂ .* (E_ββᵀ * X̂), dims = 1), dims = 1)
    ψ_σ² = ψ² .- (ψ_μ .^ 2)

    y = ψ_μ ./ sqrt.(1 .+ a * ψ_σ²)
    1 ./ (1 .+ exp.(-y))
end

function predict(model::BinaryLogisticRegression, X::Matrix{T};
                 marginalize::Bool = true) where T <: AbstractFloat
    if marginalize
        return predict_marginal(model, X)
    end
    return predict_map(model, X)
end

#######################################################################
# K-class Logistic Regression (LR) based on the stick-breaking process
#
# References
# ----------
# [1] Dependent Multinomial Modes Made Easy: Stick-Breaing with the
#     Polya-Gamma Augmentation (https://arxiv.org/pdf/1506.05843.pdf)


struct LogisticRegression{K} <: Model
    # The LR is defined by K - 1 binary LR models (see eq. (7) in [1])
    stickbreaking::Array{BinaryLogisticRegression}

    function LogisticRegression(sb::Array{BinaryLogisticRegression})
        model = new{length(sb) + 1}(sb)

        # We override the stats
        for (k, binarylr) in enumerate(sb)
            binarylr.β.stats = data -> stats_β(model, k, data...)
        end

        return model
    end
end

function LogisticRegression(μ₀::Vector{T}, Σ₀::Matrix{T}, μ::Vector{T},
                            Σ::Matrix{T}, hasbias::Bool; nclasses::Integer) where T <: AbstractFloat
    LogisticRegression([BinaryLogisticRegression(μ₀, Σ₀, deepcopy(μ), deepcopy(Σ), hasbias)
                        for i in 1:nclasses-1])
end

function LogisticRegression(T = Float64; inputdim::Integer, nclasses::Integer,
                            initσ::Real = 0.0, pseudocounts::Real = 1.0,
                            hasbias::Bool = true)
    dim = hasbias ? inputdim + 1 : inputdim

    # Prior parameters
    μ₀ = zeros(T, dim)
    Σ₀ = T(1/pseudocounts) * Matrix{T}(I, dim, dim)

    # Initial paremeterization of the posterior
    μ = μ₀ .+ T(initσ) .* randn(T)
    Σ = deepcopy(Σ₀)

    LogisticRegression(μ₀, Σ₀, μ, Σ, hasbias, nclasses = nclasses)
end

getconjugateparams(model::LogisticRegression) = [m.β for m in model.stickbreaking]

function _encode(::LogisticRegression{K}, z::Vector{T}) where {K, T <: Integer}
    onehot = zeros(T, K - 1, length(z))
    for i = 1:K-1
        onehot[i, z .== i] .= 1
    end

    # See eq. (6) in [1]. In our case, N = 1
    Nₖ = zeros(T, K - 1, length(z))
    for i = 1:K-1
        Nₖ[i, z .>= i] .= 1
    end

   onehot, Nₖ
end

function (model::LogisticRegression)(X::Matrix{T1}, z::Vector{T2},) where {T1 <: AbstractFloat,
                                                                           T2 <: Integer}
    onehot, Nₖ = _encode(model, z)
    retval = zeros(T1, size(X, 2))
    for (k, m) in enumerate(model.stickbreaking)
        s1 = onehot[k, z .>= k]
        if length(s1) < 1
            continue
        end
        retval[z .>= k] .+= m(X[:, z .>= k], onehot[k, z .>= k], Nₖ[k, z .>= k])
    end
    retval
end

function stats_β(model::LogisticRegression, k::Integer,
                         X::Matrix{T1},
                         z::Vector{T2}) where {T1 <: AbstractFloat, T2 <: Real}
    onehot, Nₖ = _encode(model, z)
    stats_β(model.stickbreaking[k], X[:, z .>= k], onehot[k, z .>= k],
                  Nₖ[k, z .>= k])
end

function predict(model::LogisticRegression{K}, X::Matrix{T};
                 marginalize::Bool = true) where {K, T <: AbstractFloat}
    retval = zeros(T, K, size(X, 2))
    for k in 1:K
        residual = dropdims(sum(retval[1:k, :], dims = 1), dims = 1)
        if k < K
            retval[k, :] = predict(model.stickbreaking[k], X,
                                   marginalize = marginalize) .* (1 .- residual)
        else
            retval[k, :] = (1 .- residual)
        end
    end
    retval
end

