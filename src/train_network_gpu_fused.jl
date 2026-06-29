using AMDGPU, LinearAlgebra

include("utility_functions.jl")

# Fuses:
#     z .+= bias
#     activation .= relu(z)
function add_bias_relu!(activation, z, bias, rows, cols)
        idx = workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x
        total = rows * cols

        if idx <= total
                row = ((idx - 1) % rows) + 1
                @inbounds v = z[idx] + bias[row]
                @inbounds activation[idx] = ifelse(v > 0f0, v, 0f0)
        end

        return
end

# Fuses:
#     z .+= bias
#     activation .= sigmoid(z)
function add_bias_sigmoid!(activation, z, bias, rows, cols)
        idx = workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x
        total = rows * cols

        if idx <= total
                row = ((idx - 1) % rows) + 1
                @inbounds v = z[idx] + bias[row]
                @inbounds activation[idx] = 1f0 / (1f0 + exp(-v))
        end

        return
end

# Fuses multiclass output work:
#     logits = z .+ bias
#     activation = softmax(logits)
#     delta = activation
#     delta[y[j], j] -= 1
function softmax_delta!(activation, delta, z, bias, y_batch, num_classes, batch_size)
        j = workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x

        if j <= batch_size
                max_logit = -Inf32

                # numerical stability: subtract per-column max
                @inbounds for c in 1:num_classes
                        logit = z[c, j] + bias[c]
                        max_logit = max(max_logit, logit)
                end

                sum_exp = 0f0

                @inbounds for c in 1:num_classes
                        e = exp(z[c, j] + bias[c] - max_logit)
                        activation[c, j] = e
                        sum_exp += e
                end

                cls = y_batch[j]

                @inbounds for c in 1:num_classes
                        p = activation[c, j] / sum_exp
                        activation[c, j] = p
                        delta[c, j] = p
                end

                @inbounds delta[cls, j] -= 1f0
        end

        return
end

# Fuses ReLU derivative application:
#     delta .= ifelse.(activation .> 0f0, tmp, 0f0)
function relu_backward!(delta, tmp, activation, n)
        idx = workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x

        if idx <= n
                @inbounds delta[idx] = ifelse(activation[idx] > 0f0, tmp[idx], 0f0)
        end

        return
end

# Replaces:
#     param .-= learning_rate .* grad
function sgd_update!(param, grad, learning_rate, n)
        idx = workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x

        if idx <= n
                @inbounds param[idx] -= learning_rate * grad[idx]
        end

        return
end

function launch_1d!(kernel, args...; n, groupsize=256)
        gridsize = cld(n, groupsize)
        @roc groupsize=groupsize gridsize=gridsize kernel(args...)
end

function train_network_gpu_fused(x_train, y_train, number_of_classes, size_of_layers = [64,34], backpropagation_epochs=10, batch_size = 64, learning_rate=0.01, filepath="")
        x = ROCArray(Float32.(x_train))
        y = ROCArray(y_train)
        training_samples = size(x, 2)

        size_of_input = size(x, 1)
        size_of_output = number_of_classes
        number_of_batches = div(training_samples, batch_size)
        is_binary = size_of_output == 1

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

        groupsize = 256

        for epoch in 1:backpropagation_epochs
                print("\rProgress: $epoch/$backpropagation_epochs")
                flush(stdout)

                batch_indices = shuffle(1:training_samples)
                batches = [batch_indices[i:min(i+batch_size-1, training_samples)] for i in 1:batch_size:training_samples]

                for batch_number in 1:number_of_batches
                        batch = batches[batch_number]
                        x_batch .= x[:, batch]

                        # Forward pass
                        # For multiclass, the output layer also computes deltas_list[end].
                        for k in eachindex(layer_weights_list)
                                inputs = k == 1 ? x_batch : activations_list[k-1]
                                z = z_list[k]

                                mul!(z, layer_weights_list[k], inputs)

                                rows, cols = size(z)
                                n = rows * cols

                                if k == length(layer_weights_list)
                                        if is_binary
                                                launch_1d!(add_bias_sigmoid!, activations_list[k], z, layer_biases_list[k], rows, cols; n=n, groupsize=groupsize)
                                        else
                                                y_batch_class .= y[batch]
                                                launch_1d!(softmax_delta!, activations_list[k], deltas_list[k], z, layer_biases_list[k], y_batch_class, size_of_output, batch_size; n=batch_size, groupsize=groupsize)
                                        end
                                else
                                        launch_1d!(add_bias_relu!, activations_list[k], z, layer_biases_list[k], rows, cols; n=n, groupsize=groupsize)
                                end
                        end

                        # Backward pass
                        for k in length(layer_weights_list):-1:1
                                if k == length(layer_weights_list)
                                        if is_binary
                                                y_batch_binary .= y[:, batch]
                                                deltas_list[k] .= activations_list[k] .- y_batch_binary
                                        else
                                                # Already computed by softmax_delta! during the output forward pass.
                                        end
                                else
                                        tmp = hidden_delta_tmp_list[k]
                                        mul!(tmp, layer_weights_list[k+1]', deltas_list[k+1])

                                        n = length(deltas_list[k])
                                        launch_1d!(relu_backward!, deltas_list[k], tmp, activations_list[k], n; n=n, groupsize=groupsize)
                                end

                                inputs = k == 1 ? x_batch : activations_list[k-1]

                                mul!(gradient_layer_weights_list[k], deltas_list[k], inputs')
                                gradient_layer_weights_list[k] ./= batch_size

                                gradient_layer_biases_list[k] .= sum(deltas_list[k], dims=2)
                                gradient_layer_biases_list[k] ./= batch_size
                        end

                        # Stochastic Gradient Descent update
                        for k in eachindex(layer_weights_list)
                                launch_1d!(sgd_update!, layer_weights_list[k], gradient_layer_weights_list[k], Float32(learning_rate), length(layer_weights_list[k]); n=length(layer_weights_list[k]), groupsize=groupsize)
                                launch_1d!(sgd_update!, layer_biases_list[k], gradient_layer_biases_list[k], Float32(learning_rate), length(layer_biases_list[k]); n=length(layer_biases_list[k]), groupsize=groupsize)
                        end
                end
        end

        layer_weights_list = Array.(layer_weights_list)
        layer_biases_list  = Array.(layer_biases_list)

        if filepath != ""
                @save filepath layer_weights_list layer_biases_list
        end

        return layer_weights_list, layer_biases_list
end
