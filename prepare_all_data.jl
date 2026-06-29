include("src/prepare_data.jl")

prepare_mnist()
println("MNIST data prepared")

prepare_default_of_credit_card_clients()
println("Default of credit card clients data prepared")

prepare_cstraining()
prepare_cstraining_balance_reduce()
prepare_cstraining_balance_add()
println("CS-TRAINING data prepared")

prepare_german_data_numeric()
println("German data numeric prepared")