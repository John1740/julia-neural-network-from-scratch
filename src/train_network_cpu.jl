using LinearAlgebra

include("utility_functions.jl")

function train_network_cpu(x_train, y_train, number_of_classes, size_of_layers = [64,34], backpropagation_epochs=10, batch_size = 64, learning_rate=0.01, filepath="")
        
        # x (input_size, number_data_samples)
        # y (number_of_classes,number_data_samples)
        
        x = Float32.(x_train)
        y = y_train
        training_samples=size(x, 2)

        attributes=size(x)
        size_of_input = attributes[1]     # number of attributes

        size_of_output = number_of_classes     # number of possible classes

        number_of_batches=div(training_samples,batch_size) #round down to ignore last batch which is too small when division has remainder
        if size_of_output == 1    # bianary classifiaction
                size_of_output = 1
                output_activation = sigmoid
        else
                output_activation = softmax
        end

        # Preallocation of arrays to avoid repeated allocations during training

        sizes = [size_of_input; size_of_layers; size_of_output]

        layer_weights_list = [kaiming_normal(sizes[i+1], sizes[i]) for i in 1:length(sizes)-1]
        layer_biases_list = [ zeros(Float32, (sizes[i+1], 1)) for i in 1:length(sizes)-1]
        deltas_list= [ zeros(Float32, (sizes[i+1], batch_size)) for i in 1:length(sizes)-1]
        activations_list=[ zeros(Float32, (sizes[i+1], batch_size)) for i in 1:length(sizes)-1]
        gradient_layer_weights_list = [ zeros(Float32, (sizes[i+1], sizes[i])) for i in 1:length(sizes)-1]
        gradient_layer_biases_list = [ zeros(Float32, (sizes[i+1], 1)) for i in 1:length(sizes)-1]

        # Reusable work buffers, to avoid temporary allocations.

        z_list = [ zeros(Float32, (sizes[i+1], batch_size)) for i in 1:length(sizes)-1]
        hidden_delta_tmp_list = [ zeros(Float32, (sizes[i+1], batch_size)) for i in 1:length(sizes)-2]

        x_batch = zeros(Float32, size_of_input, batch_size)
        y_batch_binary = zeros(Float32, 1, batch_size)
        y_batch_class = zeros(eltype(y), batch_size)

        # back propagation

        for i in 1:backpropagation_epochs
                print("\rProgress: $i/$backpropagation_epochs")
                flush(stdout)
                batch_indices = shuffle(1:training_samples)
                batches = [batch_indices[i:min(i+batch_size-1, training_samples)] for i in 1:batch_size:training_samples]
                
                for j in 1:number_of_batches
                        batch = batches[j]
                        @views x_batch .= x[:, batch]
                        
                        # calculate output
                        for k in eachindex(layer_weights_list)
                                inputs = k == 1 ? x_batch : activations_list[k-1]
                                z = z_list[k]

                                mul!(z, layer_weights_list[k], inputs)
                                z .+= layer_biases_list[k]
                                
                                # Activation function
                                if k == length(layer_weights_list)  # output layer
                                        activations_list[k] .= output_activation(z)
                                else  # hidden layers
                                        activations_list[k] .= max.(0f0,z)
                                end
                        end



                        for k in length(layer_weights_list):-1:1
                                if k==length(layer_weights_list)

                                        if size_of_output == 1
                                                @views y_batch_binary .= y[:, batch]
                                                deltas_list[k] .= activations_list[k] .- y_batch_binary
                                        else
                                                @views y_batch_class .= y[batch]
                                                # deltas_list[k] already has right shape; reuse it
                                                deltas = deltas_list[k]
                                                deltas .= activations_list[k]   # copy predictions

                                                @inbounds for (j, cls) in enumerate(y_batch_class)
                                                        deltas[cls, j] -= 1        # subtract 1 at the correct class
                                                end
                                        end

                                else
                                        tmp = hidden_delta_tmp_list[k]
                                        mul!(tmp, layer_weights_list[k+1]', deltas_list[k+1])
                                        deltas_list[k] .= ifelse.(activations_list[k] .> 0f0, tmp, 0f0)
                                end

                                inputs = k == 1 ? x_batch : activations_list[k-1]
                                mul!(gradient_layer_weights_list[k], deltas_list[k], inputs')
                                gradient_layer_weights_list[k] ./= batch_size
                                gradient_layer_biases_list[k] .= sum(deltas_list[k], dims=2)
                                gradient_layer_biases_list[k] ./= batch_size
                        end


                        for k in eachindex(layer_weights_list)
                                layer_weights_list[k] .-= learning_rate .* gradient_layer_weights_list[k]
                        end

                        for k in eachindex(layer_biases_list)
                                layer_biases_list[k] .-= learning_rate .* gradient_layer_biases_list[k]
                        end

                        
                end


        end

        if filepath != ""

                @save filepath layer_weights_list layer_biases_list
        end

        return layer_weights_list, layer_biases_list
end
