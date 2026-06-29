using AMDGPU, LinearAlgebra

include("utility_functions.jl")

function subtract_one_multiclass!(deltas, y_batch, batch_size)
        j = workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x

        if j <= batch_size
                cls = y_batch[j]
                @inbounds deltas[cls, j] -= 1f0
        end

        return
end

function train_network_gpu(x_train, y_train, number_of_classes, size_of_layers = [64,34], backpropagation_epochs=10, batch_size = 64, learning_rate=0.01, filepath="")
        
        # x size (input_size, number_data_samples)
        # y holds classificcation and has size size (number_of_choices,number_data_samples)
        x = ROCArray(Float32.(x_train))
        y = ROCArray(y_train)
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

        # initialize layers

        
        sizes = [size_of_input; size_of_layers; size_of_output]

        layer_weights_list = [
        ROCArray(kaiming_normal(sizes[i+1], sizes[i]))
        for i in 1:length(sizes)-1
        ]

        layer_biases_list = [
        AMDGPU.zeros(Float32, sizes[i+1], 1)
        for i in 1:length(sizes)-1
        ]

        deltas_list = [
        AMDGPU.zeros(Float32, sizes[i+1], batch_size)
        for i in 1:length(sizes)-1
        ]

        activations_list = [
        AMDGPU.zeros(Float32, sizes[i+1], batch_size)
        for i in 1:length(sizes)-1
        ]

        gradient_layer_weights_list = [
        AMDGPU.zeros(Float32, sizes[i+1], sizes[i])
        for i in 1:length(sizes)-1
        ]

        gradient_layer_biases_list = [
        AMDGPU.zeros(Float32, sizes[i+1], 1)
        for i in 1:length(sizes)-1
        ]

        z_list = [
        AMDGPU.zeros(Float32, sizes[i+1], batch_size)
        for i in 1:length(sizes)-1
        ]

        hidden_delta_tmp_list = [
        AMDGPU.zeros(Float32, sizes[i+1], batch_size)
        for i in 1:length(sizes)-2
        ]

        x_batch = AMDGPU.zeros(Float32, size_of_input, batch_size)
        y_batch_binary = AMDGPU.zeros(Float32, 1, batch_size)
        y_batch_class = AMDGPU.zeros(eltype(y), batch_size)

        # back propagation


        for i in 1:backpropagation_epochs
                print("\rProgress: $i/$backpropagation_epochs")
                flush(stdout)
                batch_indices = shuffle(1:training_samples)
                batches = [batch_indices[i:min(i+batch_size-1, training_samples)] for i in 1:batch_size:training_samples]
                
                for j in 1:number_of_batches
                        batch = batches[j]
                        x_batch .= x[:, batch]

                        # calculate output
                        for k in eachindex(layer_weights_list)
                                inputs = k == 1 ? x_batch : activations_list[k-1]
                                z = z_list[k]

                                mul!(z, layer_weights_list[k], inputs)
                                z .+= layer_biases_list[k]

                                if k == length(layer_weights_list)
                                        activations_list[k] .= output_activation(z)
                                else
                                        activations_list[k] .= max.(0f0, z)
                                end
                        end


                        #positive_weight = 0.07  # Anteil der Minoritätsklasse
                        #negative_weight = 1-positive_weight  # Anteil der Majoritätsklasse
                        for k in length(layer_weights_list):-1:1
                                if k==length(layer_weights_list)
                                        #deltas_list[k] .= activations_list[k] - y[:, batch]
                                        if size_of_output == 1
                                                y_batch_binary .= y[:, batch]
                                                deltas_list[k] .= activations_list[k] .- y_batch_binary
                                        else
                                                y_batch_class .= y[batch]
                                                deltas = deltas_list[k]
                                                deltas .= activations_list[k]

                                                groupsize = 256
                                                gridsize = cld(batch_size, groupsize)
                                                @roc groupsize=groupsize gridsize=gridsize subtract_one_multiclass!(deltas, y_batch_class, batch_size)
                                        end

                                        #deltas_list[k] .= (activations_list[k] - y[:, batch]) .* (y[:, batch] .* positive_weight .+ (1 .- y[:, batch]) .* negative_weight)
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
 
        layer_weights_list = Array.(layer_weights_list)
        layer_biases_list  = Array.(layer_biases_list)
        #filename="layers:"+string(size_of_layers)+"_epochs:"+string(backpropagation_epochs)+"_batchsize:"+string(batch_size)+"learningrate"+string(learning_rate)+".jld2"
        if filepath != ""
                @save filepath layer_weights_list layer_biases_list
        end

        return layer_weights_list, layer_biases_list
end