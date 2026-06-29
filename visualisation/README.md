# ANN training scaling benchmarks

Run from the repository root:

```powershell
julia visualisation/run_scaling_benchmarks.jl --quick
julia visualisation/plot_scaling_benchmarks.jl
```

Remove `--quick` for the full sweep (three measurements per point). The runner
benchmarks CPU, GPU, and fused GPU training on a fixed subset of the training
data. It measures batch-size scaling, layer-width scaling, depth scaling at a
fixed width of 256, and epoch scaling. Compilation is warmed up before timings.

Run only one sweep, or a comma-separated selection, with `--sweeps`:

```powershell
julia visualisation/run_scaling_benchmarks.jl --sweeps=depth
julia visualisation/run_scaling_benchmarks.jl --sweeps=epochs --quick
julia visualisation/run_scaling_benchmarks.jl --sweeps=batch_size,layer_width
```

Valid names are `batch_size`, `layer_width`, `depth`, and `epochs`; the short
aliases `batch`, `width`, and `epoch` also work. Without `--sweeps`, all four run.

Options:

```powershell
julia visualisation/run_scaling_benchmarks.jl --dataset=mnist --output=visualisation/my_results.csv
julia visualisation/plot_scaling_benchmarks.jl visualisation/my_results.csv visualisation
```

Outputs are `scaling_results.csv`, `scaling_summary.csv`,
`batch_size_scaling.png`, `layer_width_scaling.png`, `depth_scaling.png`, and
`epoch_scaling.png`. Only plots represented in the input CSV are generated. The plotter needs the
`CSV`, `DataFrames`, and `Plots` packages; the training scripts retain their
existing `JLD2`, `AMDGPU`, and other dependencies.

## Network architecture diagram

Create a diagram directly from the weights and biases in any trained network:

```powershell
julia visualisation/plot_network_architecture.jl trained_networks/mnist_layers_512_256_128_epochs_10_batchsize_1024_learningrate_0.01.jld2
```

The default output is `visualisation/network_architecture.png`. An optional
second argument chooses another PNG or SVG output path:

```powershell
julia visualisation/plot_network_architecture.jl trained_networks/my_model.jld2 visualisation/my_architecture.svg
```

Every neuron and weighted edge is rendered. Positive weights are blue, negative
weights are red, and weights near zero are grey; stronger weights are thicker
and more opaque. Non-input neuron colors use the same red-grey-blue scale to
show their learned biases. Large input layers therefore produce deliberately
tall, high-resolution diagnostic figures.
