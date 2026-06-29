using JLD2

include("src/test_network.jl")
include("src/train_network_cpu.jl")
include("src/train_network_gpu.jl")
include("src/train_network_gpu_fused.jl")

# select data
data="mnist"  # "mnist", "cstraining", "german_data_numeric", "default_of_credit_card_clients"

# configure network and training parameters
size_of_layers = [512, 256, 128]
batch_size = 1024
backpropagation_epochs = 10
learning_rate=0.01


overwrite=1 # 0: do not overwrite existing trained network, 1: overwrite existing trained network

# load data
@load "prepared_data/" * data * "_train.jld2" x_train y_train number_of_classes
@load "prepared_data/" * data * "_test.jld2" x_test y_test number_of_classes

prefix=data
filename = "layers_" * join(size_of_layers, "_") *
        "_epochs_$(backpropagation_epochs)_batchsize_$(batch_size)_learningrate_$(learning_rate).jld2"
filename_gpu = "layers_" * join(size_of_layers, "_") *
        "_epochs_$(backpropagation_epochs)_batchsize_$(batch_size)_learningrate_$(learning_rate)_gpu.jld2"
filename_gpu_fused = "layers_" * join(size_of_layers, "_") *
        "_epochs_$(backpropagation_epochs)_batchsize_$(batch_size)_learningrate_$(learning_rate)_gpu_fused.jld2"

# add prefix
if prefix != ""
    filename = prefix * "_" * filename
    filename_gpu = prefix * "_" * filename_gpu
    filename_gpu_fused = prefix * "_" * filename_gpu_fused
end

filepath=joinpath("trained_networks", filename)
filepath_gpu=joinpath("trained_networks", filename_gpu)
filepath_gpu_fused=joinpath("trained_networks", filename_gpu_fused)


if !isfile(filepath) || overwrite!=0 # only train if the file does not exist or overwrite is set to 1

        # run once to to compile the functions and avoid compilation time in the time measurement
        layer_weights_list, layer_biases_list =  train_network_cpu(x_train, y_train, number_of_classes, size_of_layers, 1, batch_size, learning_rate, "trained_networks/warmup.jld2")
        layer_weights_list, layer_biases_list = train_network_gpu(x_train, y_train, number_of_classes, size_of_layers, 1, batch_size, learning_rate, "trained_networks/warmup.jld2")
        layer_weights_list, layer_biases_list = train_network_gpu_fused(x_train, y_train, number_of_classes, size_of_layers, 1, batch_size, learning_rate, "trained_networks/warmup.jld2")

        
        println("")
        println("CPU Training:")
        @time layer_weights_list, layer_biases_list = train_network_cpu(x_train, y_train, number_of_classes, size_of_layers, backpropagation_epochs, batch_size, learning_rate, filepath)
        println("GPU Training:")
        @time layer_weights_list, layer_biases_list = train_network_gpu(x_train, y_train, number_of_classes, size_of_layers, backpropagation_epochs, batch_size, learning_rate, filepath_gpu)
        println("Fused GPU Training:")
        @time layer_weights_list, layer_biases_list = train_network_gpu_fused(x_train, y_train, number_of_classes, size_of_layers, backpropagation_epochs, batch_size, learning_rate, filepath_gpu_fused)
end

#test the trained network
println("")
println("Testing CPU trained network:")
test_network(x_test, y_test, number_of_classes, filepath)
println("Testing GPU trained network:")
test_network(x_test, y_test, number_of_classes, filepath_gpu)
println("Testing Fused GPU trained network:")
test_network(x_test, y_test, number_of_classes, filepath_gpu_fused)