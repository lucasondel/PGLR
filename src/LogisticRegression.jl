# Bayesian Logistic Regression using the Polya-Gamma augmentation.
#

using LinearAlgebra

export BinaryLogisticRegression
export BinaryLogisticRegressionIso
export LogisticRegression
export predict
export regressors

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

# Return the Polya-Gamma posterior of the augmented
# logistic function
function _augment_posterior(ψ²::Vector{T}) where T <: AbstractFloat
    b = ones(T, length(ψ²))
    qω = PolyaGamma(b)
    update!(qω, -.5 * ψ²)
    return qω
end

#######################################################################
# Binary logistic regression
#

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

# Return the sufficient statistics of the log likelihood and the
# the expected statistics of β.
function _precompute(model::BinaryLogisticRegression, X::Matrix{T1},
                     z::Vector{T2},
                     Nₖ::Union{Vector{T2}, Nothing} = nothing) where {T1 <: AbstractFloat,
                                                                      T2 <: Real}
    X̂ = regressors(model.hasbias, X)

    # Quadratic expansion of the regressors
    D, N = size(X̂)
    X̂X̂ᵀ = reshape(X̂, D, 1, N) .* reshape(X̂, 1, D, N)
    vec_X̂X̂ᵀ = reshape(X̂X̂ᵀ, :, N)

    # Expectation of the regressors: E[ T(β) ] = E[ ( β, vec(ββᵀ) )]
    E_Tβ = gradlognorm(model.β.posterior)
    vec_E_ββᵀ = E_Tβ[D+1:end]

    # Augmentation of the logistic function
    pω = PolyaGamma{T1, size(X̂, 2)}()
    ψ² = vec_X̂X̂ᵀ' * vec_E_ββᵀ
    qω = _augment_posterior(ψ²)
    E_ω = mean(qω)

    zstats = isnothing(Nₖ) ? (z .- 0.5) : z .- Nₖ * 0.5

    stats = vcat(reshape(zstats, 1, N) .* X̂,
                 -.5 .* reshape(E_ω, 1, :) .* vec_X̂X̂ᵀ)

    stats, E_Tβ, pω, qω
end

function (model::BinaryLogisticRegression)(X::Matrix{T1},
                                           z::Vector{T2},
                                           Nₖ::Union{Vector{T2}, Nothing} = nothing) where {T1 <: AbstractFloat,
                                                                                            T2 <: Real}
    stats, E_Tβ, pω, qω = _precompute(model, X, z, Nₖ)

    # KL divegerence betwen qω and pω for each sample.
    q_η, p_η = naturalparam(qω), naturalparam(pω)
    KL = lognorm(pω, perdim = true) - lognorm(qω, perdim = true) - (p_η .- q_η) .* gradlognorm(qω)

    stats' * E_Tβ .- KL .- log(2)
end

function stats_β(model::BinaryLogisticRegression,
                       X::Matrix{T1},
                       z::Vector{T2},
                       Nₖ::Union{Vector{T2}, Nothing} = nothing) where {T1 <: AbstractFloat,
                                                                        T2 <: Real}
    stats, E_Tβ, _, _ = _precompute(model, X, z, Nₖ)
    stats
end

function predict(model::BinaryLogisticRegression,
                 X::Matrix{T}; a::T = 0.368) where T <: AbstractFloat
    # Following https://arxiv.org/pdf/1703.00091.pdf, we use the
    # approximation:
    # ⟨σ(ψ)⟩ ≈ σ(μ / √(1 + a * σ²))
    # where a = 0.368
    # In Bishop 4.5.2 a very similar approximation uses a = π / 8 ≈ 0.392...

    μ, Σ = stdparam(model.β.posterior)
    X̂ = regressors(model.hasbias, X)

    σ²s = [sum((Σ .* x̂') .* x̂) for x̂ in eachcol(X̂)]
    ψs = (X̂' * μ) ./ sqrt.(1 .+ a * σ²s)
    1 ./ (1 .+ exp.(-ψs))
end

#######################################################################
# Binary logistic regression with isotropic cov. posterior
#

struct BinaryLogisticRegressionIso <: Model
    β::ConjugateParameter{<:NormalIso}
    hasbias::Bool

    function BinaryLogisticRegressionIso(β::ConjugateParameter{<:NormalIso}, hasbias::Bool)
        model = new(β, hasbias)
        # The model is not fully conjugate so no easy stats accumulation.
        #β.stats = data -> stats_β(model, data...)
        return model
    end
end

function BinaryLogisticRegressionIso(μ₀::Vector{T}, v₀::Real,
                                     μ::Vector{T}, v::Real,
                                     hasbias::Bool) where T <: AbstractFloat
    param = ConjugateParameter(NormalIso(μ₀, v₀), NormalIso(μ, v))
    BinaryLogisticRegressionIso(param, hasbias)
end

function BinaryLogisticRegressionIso(T = Float64; inputdim::Integer, initσ::Real = 0.0,
                                     pseudocounts::Real = 1.0, hasbias::Bool = true)
    dim = hasbias ? inputdim + 1 : inputdim

    # Prior parameters
    μ₀ = zeros(T, dim)
    v₀ = T(1/pseudocounts)

    # Initial paremeterization of the posterior
    μ = μ₀ .+ T(initσ) .* randn(T)
    v = T(1/pseudocounts)

    BinaryLogisticRegressionIso(μ₀, v₀, μ, v, hasbias)
end

# Return the sufficient statistics of the log likelihood and the
# the expected statistics of β.
function _precompute(model::BinaryLogisticRegressionIso, X::Matrix{T1},
                     z::Vector{T2},
                     Nₖ::Union{Vector{T2}, Nothing} = nothing) where {T1 <: AbstractFloat,
                                                                      T2 <: Real}
    X̂ = regressors(model.hasbias, X)
    D, N = size(X̂)

    # Mean and (isotropic) variance of the variational posterior
    m, v = stdparam(model.β.posterior)

    # "Isotropic" quadratic expansion of the regressors
    X̂ᵀX̂ = sum(X̂ .* X̂, dims = 1) / D
    E_ψ = m' * X̂
    E_ψ² = dropdims(v * X̂ᵀX̂ .+ E_ψ .^ 2, dims = 1)

    # Augmentation of the logistic function
    pω = PolyaGamma{T1, size(X̂, 2)}()
    qω = _augment_posterior(E_ψ²)

    m, v, dropdims(E_ψ, dims = 1), E_ψ², mean(qω), pω, qω
end

function (model::BinaryLogisticRegressionIso)(
    X::Matrix{T1},
    z::Vector{T2},
    Nₖ::Union{Vector{T2}, Nothing} = nothing
) where {T1 <: AbstractFloat, T2 <: Real}

    m, v, E_ψ, E_ψ², E_ω, pω, qω = _precompute(model, X, z, Nₖ)

    # KL divegerence betwen qω and pω for each sample.
    q_η, p_η = naturalparam(qω), naturalparam(pω)
    KL = lognorm(pω, perdim = true) - lognorm(qω, perdim = true) - (p_η .- q_η) .* gradlognorm(qω)


    (z .- 0.5) .* E_ψ - 0.5 * E_ω .* E_ψ²   .- log(2) .- KL
end

function stats_β(
    model::BinaryLogisticRegressionIso,
    X::Matrix{T1},
    z::Vector{T2},
    Nₖ::Union{Vector{T2}, Nothing} = nothing
) where {T1 <: AbstractFloat, T2 <: Real}

    _precompute(model, X, z, Nₖ)[1]
end

function predict(model::BinaryLogisticRegressionIso,
                 X::Matrix{T}; a::T = 0.368) where T <: AbstractFloat
    # Following https://arxiv.org/pdf/1703.00091.pdf, we use the
    # approximation:
    # ⟨σ(ψ)⟩ ≈ σ(μ / √(1 + a * σ²))
    # where a = 0.368
    # In Bishop 4.5.2 a very similar approximation uses a = π / 8 ≈ 0.392...

    μ, v = stdparam(model.β.posterior)
    X̂ = regressors(model.hasbias, X)

    σ²s = [sum((v .* x̂') .* x̂) for x̂ in eachcol(X̂)]
    ψs = (X̂' * μ) ./ sqrt.(1 .+ a * σ²s)
    1 ./ (1 .+ exp.(-ψs))
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

function predict(model::LogisticRegression{K}, X::Matrix{T}) where {K, T <: AbstractFloat}
    retval = zeros(T, K, size(X, 2))
    for k in 1:K
        residual = dropdims(sum(retval[1:k, :], dims = 1), dims = 1)
        if k < K
            retval[k, :] = predict(model.stickbreaking[k], X) .* (1 .- residual)
        else
            retval[k, :] = (1 .- residual)
        end
    end
    retval
end

