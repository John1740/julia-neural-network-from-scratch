using JLD2
using Random
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "train_network_cpu.jl"))
include(joinpath(ROOT, "src", "train_network_gpu.jl"))
include(joinpath(ROOT, "src", "train_network_gpu_fused.jl"))

const IMPLEMENTATIONS = [
    ("CPU", train_network_cpu),
    ("GPU", train_network_gpu),
    ("GPU fused", train_network_gpu_fused),
]
const VALID_SWEEPS = ["batch_size", "layer_width", "depth", "epochs"]

function canonical_sweep(name)
    aliases = Dict("batch" => "batch_size", "width" => "layer_width",
                   "layer" => "layer_width", "epoch" => "epochs")
    return get(aliases, lowercase(strip(name)), lowercase(strip(name)))
end

function parse_args(args)
    quick = "--quick" in args
    dataset = "mnist"
    output = joinpath(@__DIR__, "scaling_results.csv")
    sweeps = copy(VALID_SWEEPS)
    for arg in args
        startswith(arg, "--dataset=") && (dataset = split(arg, "=", limit=2)[2])
        startswith(arg, "--output=") && (output = abspath(split(arg, "=", limit=2)[2]))
        if startswith(arg, "--sweeps=") || startswith(arg, "--sweep=")
            selection = split(arg, "=", limit=2)[2]
            sweeps = canonical_sweep.(split(selection, ','))
        end
    end
    invalid = setdiff(sweeps, VALID_SWEEPS)
    isempty(invalid) || error("Unknown sweep(s): $(join(invalid, ", ")). Valid sweeps: $(join(VALID_SWEEPS, ", "))")
    isempty(sweeps) && error("At least one sweep must be selected")
    return (; quick, dataset, output, sweeps)
end

function timed_run(train!, x, y, classes, layers, epochs, batch_size, learning_rate)
    GC.gc()
    elapsed = @elapsed train!(x, y, classes, layers, epochs, batch_size, learning_rate, "")
    return elapsed
end

function write_row(io, sweep, value, implementation, repetition, seconds,
                   samples, epochs, batch_size, layers)
    layer_text = join(layers, "x")
    println(io, join((sweep, value, implementation, repetition, seconds,
                      samples, epochs, batch_size, layer_text), ','))
    flush(io) # Preserve completed measurements if a later configuration fails.
end

function main(args=ARGS)
    config = parse_args(args)
    data_path = joinpath(ROOT, "prepared_data", config.dataset * "_train.jld2")
    isfile(data_path) || error("Training data not found: $data_path")
    @load data_path x_train y_train number_of_classes

    # Keep the number of samples fixed so values within each sweep are comparable.
    sample_count = min(size(x_train, 2), config.quick ? 2_048 : 16_384)
    x = x_train[:, 1:sample_count]
    y = ndims(y_train) == 1 ? y_train[1:sample_count] : y_train[:, 1:sample_count]
    base_epochs = 1
    repeats = config.quick ? 1 : 3
    learning_rate = 0.01
    requested_batch_sizes = config.quick ? [64, 256] : [32, 64, 128, 256, 512, 1024]
    batch_sizes = filter(<=(sample_count), requested_batch_sizes)
    layer_widths = config.quick ? [64, 256] : [32, 64, 128, 256, 512, 1024]
    depths = config.quick ? [1, 3] : [1, 2, 3, 4, 6, 8]
    epoch_counts = config.quick ? [1, 2] : [1, 2, 4, 8, 16]
    fixed_layers = [256, 256, 256]
    fixed_width = 256
    fixed_batch = min(256, sample_count)

    mkpath(dirname(config.output))
    open(config.output, "w") do io
        println(io, "sweep,value,implementation,repetition,seconds,samples,epochs,batch_size,layers")

        # Compile each implementation before measuring it. The conversion back from
        # ROCArray inside the GPU functions also ensures queued GPU work is complete.
        warmup_n = min(sample_count, 256)
        warmup_x = x[:, 1:warmup_n]
        warmup_y = ndims(y) == 1 ? y[1:warmup_n] : y[:, 1:warmup_n]
        for (name, train!) in IMPLEMENTATIONS
            println("Warming up $name...")
            train!(warmup_x, warmup_y, number_of_classes, [32], 1,
                   min(64, warmup_n), learning_rate, "")
        end

        if "batch_size" in config.sweeps
          for batch_size in batch_sizes, (name, train!) in IMPLEMENTATIONS, repetition in 1:repeats
            Random.seed!(10_000 + repetition)
            println("batch size=$batch_size, $name, repetition $repetition/$repeats")
            seconds = timed_run(train!, x, y, number_of_classes, fixed_layers,
                                base_epochs, batch_size, learning_rate)
            write_row(io, "batch_size", batch_size, name, repetition, seconds,
                      sample_count, base_epochs, batch_size, fixed_layers)
          end
        end

        if "layer_width" in config.sweeps
          for width in layer_widths, (name, train!) in IMPLEMENTATIONS, repetition in 1:repeats
            layers = [width, width, width]
            Random.seed!(20_000 + repetition)
            println("layer width=$width, $name, repetition $repetition/$repeats")
            seconds = timed_run(train!, x, y, number_of_classes, layers,
                                base_epochs, fixed_batch, learning_rate)
            write_row(io, "layer_width", width, name, repetition, seconds,
                      sample_count, base_epochs, fixed_batch, layers)
          end
        end

        if "depth" in config.sweeps
          for depth in depths, (name, train!) in IMPLEMENTATIONS, repetition in 1:repeats
            layers = fill(fixed_width, depth)
            Random.seed!(30_000 + repetition)
            println("depth=$depth, $name, repetition $repetition/$repeats")
            seconds = timed_run(train!, x, y, number_of_classes, layers,
                                base_epochs, fixed_batch, learning_rate)
            write_row(io, "depth", depth, name, repetition, seconds,
                      sample_count, base_epochs, fixed_batch, layers)
          end
        end

        if "epochs" in config.sweeps
          for epoch_count in epoch_counts, (name, train!) in IMPLEMENTATIONS, repetition in 1:repeats
            Random.seed!(40_000 + repetition)
            println("epochs=$epoch_count, $name, repetition $repetition/$repeats")
            seconds = timed_run(train!, x, y, number_of_classes, fixed_layers,
                                epoch_count, fixed_batch, learning_rate)
            write_row(io, "epochs", epoch_count, name, repetition, seconds,
                      sample_count, epoch_count, fixed_batch, fixed_layers)
          end
        end
    end
    println("Saved $(join(config.sweeps, ", ")) benchmark data to $(config.output)")
end

main()
