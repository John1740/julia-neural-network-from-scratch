using MLDatasets, Random, DataFrames, Statistics

function sigmoid(z::AbstractMatrix)
        return 1.0 ./ (1.0 .+ exp.(-z))
end

function softmax(z::AbstractMatrix)
        # subtract max per column for numerical stability
        z_max = maximum(z, dims=1)
        exp_z = exp.(z .- z_max)
        return exp_z ./ sum(exp_z, dims=1)
end


function standardize_cols(X::AbstractMatrix)
    μ = mean(X, dims=2)
    σ = std(X, dims=2)
    return (X .- μ) ./ (σ .+ eps())
end

function standardize(x)
    μ = mean(x)
    σ = std(x)
    return σ > 0 ? (x .- μ) ./ σ : fill(0.0, length(x))
end


function kaiming_normal(out_dim::Int, in_dim::Int; rng=Random.GLOBAL_RNG, T=Float32)
        fan_in = in_dim
        std = sqrt(T(2) / T(fan_in))
        return randn(rng, T, out_dim, in_dim) .* std
end
