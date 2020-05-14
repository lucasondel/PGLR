using LinearAlgebra
using Plots

export plotnormal2d

"""
    plotnormal2d(p, μ::AbstractVector, Σ::AbstractMatrix; ncontours=2, args...)

Plot the contours of a 2d Normal density. `ncontours` is the number of
contour line to plots.
"""
function plotnormal2d(p, μ::AbstractVector, Σ::AbstractMatrix; ncontours=2, color="blue", label="", args...)
    λ, U = eigen(Σ)
    for i in 1:ncontours
        B = U * diagm(i * sqrt.(λ))
        θ = range(0, stop=2 * pi, length=1000)
        circle = hcat(sin.(θ), cos.(θ))'
        contour = B * circle .+ μ
        plot!(p, contour[1, :], contour[2, :]; color=color, label=i > 1 ? "" : label, args...)
    end
    p
end

