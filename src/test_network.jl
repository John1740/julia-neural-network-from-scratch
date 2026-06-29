function test_network(x,y,number_of_classes,filename)
        # y is a labedl vector

        @load filename layer_weights_list layer_biases_list


        test_size=size(x, 2)

        size_of_output = number_of_classes

        if size_of_output == 1    # bianary classifiaction
                size_of_output = 1
                output_activation = sigmoid
        else
                output_activation = softmax
        end

        activations = x

        for k in 1:length(layer_weights_list)
                z=layer_weights_list[k] * activations .+ layer_biases_list[k]
                # Activation function
                if k == length(layer_weights_list)  # output layer
                        activations = output_activation(z)
                else  # hidden layers
                        activations = max.(0,z)
                end
        end


        if size_of_output == 1
                y_hat = activations .>= 0.5
        else
                y_hat = argmax(activations, dims=1)
                y_hat =vec( getindex.(y_hat, 1))
        end


        acc = sum(y_hat .== y)/test_size
        println("Accuracy of the model on test data: $acc")
        return nothing

end
