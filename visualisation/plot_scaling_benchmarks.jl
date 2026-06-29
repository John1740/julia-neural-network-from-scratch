using CSV
using DataFrames
using Plots
using Statistics

function main(args=ARGS)
    input = isempty(args) ? joinpath(@__DIR__, "scaling_results.csv") : abspath(args[1])
    output_dir = length(args) >= 2 ? abspath(args[2]) : @__DIR__
    isfile(input) || error("Benchmark data not found: $input")
    mkpath(output_dir)
    data = CSV.read(input, DataFrame)

    summary = combine(groupby(data, [:sweep, :value, :implementation]),
                      :seconds => median => :median_seconds,
                      :seconds => minimum => :minimum_seconds,
                      :seconds => length => :measurements)
    CSV.write(joinpath(output_dir, "scaling_summary.csv"), summary)

    specs = [
        ("batch_size", "Batch size", "batch_size_scaling.png"),
        ("layer_width", "Width of each of 3 hidden layers", "layer_width_scaling.png"),
        ("depth", "Number of hidden layers (width 256)", "depth_scaling.png"),
        ("epochs", "Training epochs", "epoch_scaling.png"),
    ]
    for (sweep, xlabel_text, filename) in specs
        subset = summary[summary.sweep .== sweep, :]
        isempty(subset) && continue
        p = plot(xlabel=xlabel_text, ylabel="Median training time (seconds)",
                 title="ANN training scaling — " * replace(sweep, '_' => ' '),
                 xscale=:identity, yscale=:log10, marker=:circle, linewidth=2,
                 legend=:topleft, grid=true)
        for implementation in unique(subset.implementation)
            rows = sort(subset[subset.implementation .== implementation, :], :value)
            plot!(p, rows.value, rows.median_seconds, label=implementation)
        end
        savefig(p, joinpath(output_dir, filename))
    end
    println("Saved plots and scaling_summary.csv to $output_dir")
end

main()
