using JLD2
using Plots

function architecture_from_weights(weights)
    isempty(weights) && error("The saved network contains no weight matrices")
    sizes = [size(weights[1], 2); [size(weight, 1) for weight in weights]]
    for index in 2:length(weights)
        size(weights[index], 2) == size(weights[index - 1], 1) ||
            error("Weight matrices $(index - 1) and $index have incompatible dimensions")
    end
    return sizes
end

neuron_positions(count) = count == 1 ? [0.0] : collect(range(1.0, -1.0, length=count))

"Add every edge, grouped into color bins to keep large networks practical to render."
function draw_weight_edges!(p, weights, xs, ys; bins=61)
    max_abs = maximum(abs, vcat(vec.(weights)...))
    max_abs = max(max_abs, eps(Float32))
    gradient = cgrad(["#DC2626", "#D1D5DB", "#2563EB"])

    for layer in eachindex(weights)
        weight = weights[layer]
        grouped_x = [Float64[] for _ in 1:bins]
        grouped_y = [Float64[] for _ in 1:bins]

        for destination in axes(weight, 1), source in axes(weight, 2)
            normalized = clamp(Float64(weight[destination, source]) / max_abs, -1, 1)
            bin = clamp(round(Int, (normalized + 1) * (bins - 1) / 2) + 1, 1, bins)
            append!(grouped_x[bin], (xs[layer], xs[layer + 1], NaN))
            append!(grouped_y[bin], (ys[layer][source], ys[layer + 1][destination], NaN))
        end

        for bin in 1:bins
            isempty(grouped_x[bin]) && continue
            normalized = 2 * (bin - 1) / (bins - 1) - 1
            magnitude = abs(normalized)
            plot!(p, grouped_x[bin], grouped_y[bin];
                  color=gradient[(normalized + 1) / 2],
                  linewidth=0.45 + 2.25 * magnitude,
                  alpha=0.14 + 0.78 * magnitude, label=false)
        end
    end
    return max_abs
end

function plot_architecture(model_path, output_path)
    isfile(model_path) || error("Trained network not found: $model_path")
    model = JLD2.load(model_path)
    haskey(model, "layer_weights_list") ||
        error("$model_path does not contain layer_weights_list")

    weights = model["layer_weights_list"]
    biases = get(model, "layer_biases_list", nothing)
    sizes = architecture_from_weights(weights)
    layer_count = length(sizes)
    xs = collect(0.0:(layer_count - 1))
    ys = neuron_positions.(sizes)

    # A tall canvas gives all 784 MNIST input neurons distinct positions.
    height = clamp(3 * maximum(sizes), 900, 2600)
    p = plot(; size=(max(1100, 260 * layer_count), height),
             xlims=(-0.45, layer_count - 0.55), ylims=(-1.09, 1.12),
             axis=false, grid=false, legend=false, background_color=:white,
             foreground_color=:black,
             title="Trained Network — Weights and Biases", titlefontsize=18,
             margin=8Plots.mm)

    max_weight = draw_weight_edges!(p, weights, xs, ys)

    # Inputs have no bias. Their small markers are all present, even for MNIST's 784 inputs.
    scatter!(p, fill(xs[1], sizes[1]), ys[1]; markersize=max(sizes[1] > 100 ? 1.2 : 4.0, 1.2),
             markercolor="#475569", markerstrokewidth=0, label=false)

    max_bias = biases === nothing ? 1.0 : maximum(abs, vcat(vec.(biases)...))
    max_bias = max(max_bias, eps(Float32))
    for layer in 2:layer_count
        layer_biases = biases === nothing ? zeros(sizes[layer]) : vec(biases[layer - 1])
        marker_size = sizes[layer] > 64 ? 2.5 : sizes[layer] > 32 ? 4.0 : 7.0
        scatter!(p, fill(xs[layer], sizes[layer]), ys[layer];
                 markersize=marker_size, marker_z=layer_biases,
                 color=cgrad(["#DC2626", "#D1D5DB", "#2563EB"]),
                 clims=(-max_bias, max_bias), colorbar=false,
                 markerstrokecolor="#334155",
                 markerstrokewidth=0.35, label=false)
    end

    for layer in 1:layer_count
        kind = layer == 1 ? "Input" : layer == layer_count ? "Output" : "Hidden $(layer - 1)"
        annotate!(p, xs[layer], 1.075, text("$kind\n$(sizes[layer]) neurons", 10, :black, :center))
    end

    parameter_count = sum(length, weights) +
                      (biases === nothing ? 0 : sum(length, biases))
    annotate!(p, (layer_count - 1) / 2, -1.065,
              text("$parameter_count parameters  •  edges: red − / grey 0 / blue +  •  " *
                   "thicker = larger |weight|  •  neuron fill shows bias: red − / grey 0 / blue +",
                   9, "#334155", :center))

    mkpath(dirname(output_path))
    savefig(p, output_path)
    println("Architecture $(join(sizes, " → "))")
    println("Rendered $(sum(length, weights)) weighted edges and $(sum(sizes)) neurons")
    println("Saved diagram to $output_path")
end


function main(args=ARGS)
    isempty(args) && error(
        "Usage: julia visualisation/plot_network_architecture.jl <trained-network.jld2> [output.png]"
    )
    model_path = abspath(args[1])
    output_path = length(args) >= 2 ? abspath(args[2]) :
                  joinpath(@__DIR__, "network_architecture.png")
    plot_architecture(model_path, output_path)
end

main()
