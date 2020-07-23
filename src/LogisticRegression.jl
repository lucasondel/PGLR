# Bayesian Logistic Regression using the Polya-Gamma augmentation.
#

using LinearAlgebra
using StatsFuns: log1pexp

export BinaryLogisticRegression
export LogisticRegression
export predict
export reorder

# TO REMOVE
export regressors
export encode
export pgaugmentation

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


# Mean of PolyaGamma distributions
function pgmean(c::Union{Vector{T}, Matrix{T}}; b::Real = 1) where T <: AbstractFloat
    retval = (b ./ (2 * c)) .* tanh.(c ./ 2)

    # When c = 0 the mean is not defined but can be extended by
    # continuity observing that lim_{x => 0} (e^(x) - 1) / x = 0             !
    # which lead to the mean = b / 4
    idxs = isnan.(retval)
    retval[idxs] .= b / 4
    return retval
end

# Log-normalizer of PolyaGamma distributions
function pglognorm(c::Union{Vector{T}, Matrix{T}}; b::Real = 1) where T <: AbstractFloat
    return -b .* (log1pexp.(c) .- log(2) .- c ./ 2)
end

# Compute the necessary results of the PolyaGamma augmentation:
#   1. mean of the ω variables
#   2. KL divergence between the variational factors q(ω) and p(ω)
function pgaugmentation(Ψ²::Union{Vector{T}, Matrix{T}}; b::Real = 1) where T <: AbstractFloat
    c = sqrt.(Ψ²)
    E_ω = pgmean(c)
    kl = -.5 .* (c.^2) .* E_ω .- pglognorm(c)
    E_ω, kl
end

# For the purpose of operating with matrices the function encodes
# the vector of classes `z` as:
#   * onehot: one-hot encoding for the classes, where the class
#     'n_classes' is encoded as all 0's
#   * N_up_to: for each sample of class c, row i is 1 if i>=c and
#     0 otherwise. Note that if c=n_classes, all rows are 0's
#
# Note: K is the number of classes
function encode(z::Vector{T}, K::Int) where T <: Int
    onehot = zeros(T, K - 1, length(z))
    for i = 1:K-1 onehot[i, z .== i] .= 1 end

    N_up_to = zeros(T, K - 1, length(z))
    for i = 1:K-1 N_up_to[i, z .>= i] .= 1 end

    onehot, N_up_to
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
) where {T1 <: AbstractFloat, T2 <: Real}

    X̂ = regressors(model.hasbias, X)

    # Expectation of the parameters β
    μ, Σ = stdparam(model.β.posterior)
    E_β = μ
    E_ββᵀ = Σ + μ * μ'

    # Pre-compute the terms of the lower-bound
    ψ = E_βᵀX̂ = X̂' * E_β
    ψ² = dropdims(sum(X̂ .* (E_ββᵀ * X̂), dims = 1), dims = 1)

    # PolyaGamma augmentation
    E_ω, kl_ω = pgaugmentation(ψ²)

    (z .- 0.5) .* ψ .-.5 .* E_ω .* ψ² .- log(2) .- kl_ω
end

function stats_β(model::BinaryLogisticRegression,
    X::Matrix{T1},
    z::Vector{T2},
) where {T1 <: AbstractFloat, T2 <: Real}

    X̂ = regressors(model.hasbias, X)

    # Expectation of the parameters β
    μ, Σ = stdparam(model.β.posterior)
    E_β = μ
    E_ββᵀ = Σ + μ * μ'

    # Polya-Gamma augmentation: compute the prior/posterior over ω
    ψ² = dropdims(sum(X̂ .* (E_ββᵀ * X̂), dims = 1), dims = 1)

    # PolyaGamma augmentation
    E_ω, kl_ω = pgaugmentation(ψ²)

    # 1st order statistics
    zstats = (z .- 0.5)
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
    stickbreaking::Vector{BinaryLogisticRegression}
    ordering::Dict{Int, Int}

    function LogisticRegression(stickbreaking::Vector{BinaryLogisticRegression},
                                hasbias::Bool)
        K = length(stickbreaking) + 1
        model = new{K}(stickbreaking, Dict(i => i for i in 1:K))

        # We override the stats
        for (k, blr) in enumerate(stickbreaking)
            blr.β.stats = data -> stats_β(model, k, data...)
        end

        return model
    end
end

function LogisticRegression(μ₀::Vector{T}, Σ₀::Matrix{T}, μ::Vector{T},
                            Σ::Matrix{T}, hasbias::Bool; nclasses::Integer
                           ) where T <: AbstractFloat
    stickbreaking = Vector{BinaryLogisticRegression}()
    for i = 1:nclasses - 1
        push!(stickbreaking,
              BinaryLogisticRegression(μ₀, Σ₀, deepcopy(μ), deepcopy(Σ), hasbias))
    end
    LogisticRegression(stickbreaking, hasbias)
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

getconjugateparams(model::LogisticRegression) = [blr.β for blr in model.stickbreaking]

function reorder(
    model::LogisticRegression{K},
    X::Matrix{T1},
    c::Vector{T2}
) where {K, T1 <: AbstractFloat, T2 <: Integer}

    # Compute the order-dependent terms of the ELBO
    ln1_ν = ones(T1, K-1, length(c))
    for (i, blr) in enumerate(model.stickbreaking)
        X̂ = regressors(blr.hasbias, X)
        μ, Σ = stdparam(blr.β.posterior)
        E_β = μ
        E_ββᵀ = Σ + μ * μ'

        ψ = X̂' * E_β
        ψ² = dropdims(sum(X̂ .* (E_ββᵀ * X̂), dims = 1), dims = 1)

        E_ω, kl_ω = pgaugmentation(ψ²)

        ln1_ν[i, :] = .5 * (-ψ .- E_ω .* ψ² .- kl_ω)
    end

    # Sum over the frame dimension. The result is a KxK-1 matrix.
    M = zeros(K, K-1)
    for i in 1:3 M[i, :] = sum(ln1_ν .* (c .== i)', dims = 2) end

    idxs = sortperm(sum(M, dims = 2)[:, 1], rev = false)
    for (i, j) in enumerate(idxs)
        model.ordering[i] = j
    end
end

function (model::LogisticRegression{K})(
    X::Matrix{T1},
    c::Vector{T2}
) where {K, T1 <: AbstractFloat, T2 <: Integer}

    # Re-map the labels to the "ideal" ordering
    c = [model.ordering[cₙ] for cₙ in c]

    llh = zeros(T1, length(c))
    for i = 1:K-1
        idxs = c .≥ i
        zᵢ= Int.(c[idxs] .== i)
        llh[idxs] .+= model.stickbreaking[i](X[:, idxs], zᵢ)
    end
    return llh
end

function stats_β(
    model::LogisticRegression,
    i::Integer,
    X::Matrix{T1},
    c::Vector{T2}
) where {T1 <: AbstractFloat, T2 <: Real}

    # Re-map the labels to the "ideal" ordering
    c = [model.ordering[cₙ] for cₙ in c]

    idxs = c .≥ i
    zᵢ= Int.(c[idxs] .== i)
    stats_β(model.stickbreaking[i], X[:, idxs], zᵢ)
end

function predict(
    model::LogisticRegression{K},
    X::Matrix{T};
    marginalize::Bool = true
) where {K, T <: AbstractFloat}

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

    # Reverse "ideal" ordering of the target labels
    rordering = Dict(value => key for (key, value) in model.ordering)
    idxs = [rordering[i] for i in 1:K]

    retval[idxs, :]
end

