using JLD2

include("src/test_network.jl")
include("src/train_network_cpu.jl")


# select data
data="mnist"  # "mnist", "cstraining", "german_data_numeric", "default_of_credit_card_clients"

# configure network and training parameters
size_of_layers = [32,16]
batch_size = 32
backpropagation_epochs = 25
learning_rate=0.01


overwrite=1 # 0: do not overwrite existing trained network, 1: overwrite existing trained network


@load "prepared_data/" * data * "_train.jld2" x_train y_train number_of_classes
@load "prepared_data/" * data * "_test.jld2" x_test y_test number_of_classes

prefix=data
filename = "layers_" * join(size_of_layers, "_") *
        "_epochs_$(backpropagation_epochs)_batchsize_$(batch_size)_learningrate_$(learning_rate).jld2"
filename_gpu = "layers_" * join(size_of_layers, "_") *
        "_epochs_$(backpropagation_epochs)_batchsize_$(batch_size)_learningrate_$(learning_rate)_gpu.jld2"

if prefix != ""
    filename = prefix * "_" * filename
    filename_gpu = prefix * "_" * filename_gpu
end

filepath=joinpath("trained_networks", filename)
filepath_gpu=joinpath("trained_networks", filename_gpu)

if !isfile(filepath) || overwrite!=0
        # run once to to compile the functions and avoid compilation time in the timing measurement
        layer_weights_list, layer_biases_list = train_network_cpu(x_train, y_train, number_of_classes, size_of_layers, 1, batch_size, learning_rate, "trained_networks/warmup.jld2")

        @time layer_weights_list, layer_biases_list = train_network_cpu(x_train, y_train, number_of_classes, size_of_layers, backpropagation_epochs, batch_size, learning_rate, filepath)
end

test_network(x_test, y_test, number_of_classes, filepath)
