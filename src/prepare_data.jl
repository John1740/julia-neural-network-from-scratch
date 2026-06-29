using MLDatasets, JLD2, CSV, XLSX, DataFrames
include("utility_functions.jl")

function prepare_mnist()
        test_x, test_y = MNIST(split=:test)[:]
        train_x, train_y = MNIST(split=:train)[:]
        training_samples=length(train_y)


        x_train = reshape(train_x, 28*28, :)
        y_train = zeros(Float32, 10, training_samples)
        y_train = train_y.+1

        y_test = test_y.+1
        x_test = reshape(test_x, 28*28, :)

        number_of_classes = 10
        filepath_train="prepared_data/mnist_train.jld2"
        filepath_test="prepared_data/mnist_test.jld2"
        @save filepath_train x_train y_train number_of_classes
        @save filepath_test x_test y_test number_of_classes
end


function prepare_cstraining()
        df = CSV.read("raw_data/cs-training.csv", DataFrame; header=true , missingstring=["NA"], types=Dict(i=>Float32 for i in 1:12))
        dropmissing!(df)
        df = filter(row -> 0.0 ≤ row[2] ≤ 1.0 && 0.0 ≤ row[6] ≤ 1.0, df)
        #print(df[1:20,:])
        data = convert(Array{Float32,2}, Matrix(df))
        
        data[:,4] = standardize(data[:,4])   # age
        data[:,5] = standardize(data[:,5])   # 30–59 days past due
        # data[:,6] bleibt so (DebtRatio)
        data[:,7] = standardize(log1p.(data[:,7]))  # MonthlyIncome
        data[:,8] = standardize(data[:,8])   # OpenCreditLines
        data[:,9] = standardize(data[:,9])   # 90DaysLate
        data[:,10] = standardize(data[:,10]) # RealEstateLoans
        data[:,11] = standardize(data[:,11]) # 60–89 days past due
        data[:,12] = standardize(data[:,12]) # Dependents

        x=data[:,3:end]
        #display(x[1:20,:])
        y=data[:,2]
        idx_good = findall(y .== 1)
        idx_bad  = findall(y .== 0)

        n_test_good = round(Int, length(idx_good) * 0.2)
        n_test_bad  = round(Int, length(idx_bad)  * 0.2)


        test_idx_good = idx_good[randperm(length(idx_good))[1:n_test_good]]
        test_idx_bad  = idx_bad[randperm(length(idx_bad))[1:n_test_bad]]

        test_idx = vcat(test_idx_good, test_idx_bad)
        train_idx = setdiff(1:length(y), test_idx)

        x_train = x[train_idx, :]'
        y_train = y[train_idx, :, 1]'

        x_test  = x[test_idx, :]'
        y_test  = y[test_idx, :, 1]'
        number_of_classes = 1
        filepath_train="prepared_data/cstraining_train.jld2"
        filepath_test="prepared_data/cstraining_test.jld2"
        @save filepath_train x_train y_train number_of_classes
        @save filepath_test x_test y_test number_of_classes

end


function prepare_cstraining_balance_reduce()
        df = CSV.read("raw_data/cs-training.csv", DataFrame; header=true , missingstring=["NA"], types=Dict(i=>Float32 for i in 1:12))
        dropmissing!(df)
        df = filter(row -> 0.0 ≤ row[2] ≤ 1.0 && 0.0 ≤ row[6] ≤ 1.0, df)
        #print(df[1:20,:])
        data = convert(Array{Float32,2}, Matrix(df))
        
        data[:,4] = standardize(data[:,4])   # age
        data[:,5] = standardize(data[:,5])   # 30–59 days past due
        # data[:,6] bleibt so (DebtRatio)
        data[:,7] = standardize(log1p.(data[:,7]))  # MonthlyIncome
        data[:,8] = standardize(data[:,8])   # OpenCreditLines
        data[:,9] = standardize(data[:,9])   # 90DaysLate
        data[:,10] = standardize(data[:,10]) # RealEstateLoans
        data[:,11] = standardize(data[:,11]) # 60–89 days past due
        data[:,12] = standardize(data[:,12]) # Dependents

        x=data[:,3:end]
        #display(x[1:20,:])
        y=data[:,2]
        idx_good = findall(y .== 1)
        idx_bad  = findall(y .== 0)

        n_test_good = round(Int, length(idx_good) * 0.2)
        n_test_bad  = round(Int, length(idx_bad)  * 0.2)


        test_idx_good = idx_good[randperm(length(idx_good))[1:n_test_good]]
        test_idx_bad  = idx_bad[randperm(length(idx_bad))[1:n_test_bad]]

        test_idx = vcat(test_idx_good, test_idx_bad)
        train_idx = setdiff(1:length(y), test_idx)

        x_train = x[train_idx, :]'
        y_train = y[train_idx, :, 1]'


        x_test  = x[test_idx, :]'
        y_test  = y[test_idx, :, 1]'

        y_train_vec = vec(y_train)

        idx_good_train = findall(y_train_vec .== 1)
        idx_bad_train  = findall(y_train_vec .== 0)


        # balance by undersampling majority
        n_minority = min(length(idx_good_train), length(idx_bad_train))

        balanced_good = idx_good_train[randperm(length(idx_good_train))[1:n_minority]]
        balanced_bad  = idx_bad_train[randperm(length(idx_bad_train))[1:n_minority]]

        balanced_idx = vcat(balanced_good, balanced_bad)

        x_train = x_train[:, balanced_idx]
        y_train = reshape(y_train_vec[balanced_idx], 1, :)

        number_of_classes = 1

        filepath_train="prepared_data/cstraining_balance_reduce_train.jld2"
        filepath_test="prepared_data/cstraining_balance_reduce_test.jld2"
        @save filepath_train x_train y_train number_of_classes
        @save filepath_test x_test y_test number_of_classes

end

function prepare_cstraining_balance_add()
        df = CSV.read("raw_data/cs-training.csv", DataFrame; header=true , missingstring=["NA"], types=Dict(i=>Float32 for i in 1:12))
        dropmissing!(df)
        df = filter(row -> 0.0 ≤ row[2] ≤ 1.0 && 0.0 ≤ row[6] ≤ 1.0, df)
        data = convert(Array{Float32,2}, Matrix(df))

        data[:,4] = standardize(data[:,4])   # age
        data[:,5] = standardize(data[:,5])   # 30–59 days past due
        # data[:,6] bleibt so (DebtRatio)
        data[:,7] = standardize(log1p.(data[:,7]))  # MonthlyIncome
        data[:,8] = standardize(data[:,8])   # OpenCreditLines
        data[:,9] = standardize(data[:,9])   # 90DaysLate
        data[:,10] = standardize(data[:,10]) # RealEstateLoans
        data[:,11] = standardize(data[:,11]) # 60–89 days past due
        data[:,12] = standardize(data[:,12]) # Dependents

        x=data[:,3:end]
        #display(x[1:20,:])
        y=data[:,2]

        idx0 = findall(y .== 0.0)
        idx1 = findall(y .== 1.0)

        n_samples = length(idx1)  # Ziel: gleiche Anzahl wie 1
        undersampled_idx0 = rand(idx0, n_samples)

        new_idx = vcat(undersampled_idx0, idx1)
        x_balanced = x[new_idx, :]
        y_balanced = y[new_idx]

        shuffle_idx = randperm(length(y_balanced))
        x = x_balanced[shuffle_idx, :]
        y = y_balanced[shuffle_idx]



        idx_good = findall(y .== 0)
        idx_bad  = findall(y .== 1)

        n_test_good = round(Int, length(idx_good) * 0.2)
        n_test_bad  = round(Int, length(idx_bad)  * 0.2)


        test_idx_good = idx_good[randperm(length(idx_good))[1:n_test_good]]
        test_idx_bad  = idx_bad[randperm(length(idx_bad))[1:n_test_bad]]

        test_idx = vcat(test_idx_good, test_idx_bad)
        train_idx = setdiff(1:length(y), test_idx)

        x_train = x[train_idx, :]'
        y_train = y[train_idx, :, 1]'

        x_test  = x[test_idx, :]'
        y_test  = y[test_idx, :, 1]'

        number_of_classes = 1

        filepath_train="prepared_data/cstraining_balance_add_train.jld2"
        filepath_test="prepared_data/cstraining_balance_add_test.jld2"
        @save filepath_train x_train y_train number_of_classes
        @save filepath_test x_test y_test number_of_classes
end


function prepare_german_data_numeric()
        input_file= "raw_data/german.data-numeric"

        data_lines = readlines(input_file)
        data = [parse.(Float32, split(line)) for line in data_lines]
        data = hcat(data...)'  # transponieren, weil hcat spaltenweise zusammenfügt
        y = data[:, end]
        y = map(x -> x == 1 ? 1 : 0, y)  # 1 = Good, 0 = Bad
        x = data[:, 1:end-1]
        for i in 1:size(x)[2]
                x[:,i]=standardize(x[:,i])
        end

        idx_good = findall(y .== 1)
        idx_bad  = findall(y .== 0)

        n_test_good = round(Int, length(idx_good) * 0.2)
        n_test_bad  = round(Int, length(idx_bad)  * 0.2)


        test_idx_good = idx_good[randperm(length(idx_good))[1:n_test_good]]
        test_idx_bad  = idx_bad[randperm(length(idx_bad))[1:n_test_bad]]

        test_idx = vcat(test_idx_good, test_idx_bad)
        train_idx = setdiff(1:length(y), test_idx)

        x_train = x[train_idx, :]'
        y_train = y[train_idx, :, 1]'

        x_test  = x[test_idx, :]'
        y_test  = y[test_idx, :, 1]'
        number_of_classes = 1

        filepath_train="prepared_data/german_data_numeric_train.jld2"
        filepath_test="Prepared_data/german_data_numeric_test.jld2"
        @save filepath_train x_train y_train number_of_classes
        @save filepath_test x_test y_test number_of_classes

end

function prepare_default_of_credit_card_clients()

        input_file= "raw_data/default_of_credit_card_clients.xlsx"

        df = XLSX.readtable(input_file, "Data")
        df = DataFrame(df)
        data = Matrix{Float64}(df)
        y = data[:, end]
        x = data[:, 1:end-1]
        for i in 1:size(x)[2]
                x[:,i]=standardize(x[:,i])
        end

        idx_good = findall(y .== 1)
        idx_bad  = findall(y .== 0)

        n_test_good = round(Int, length(idx_good) * 0.2)
        n_test_bad  = round(Int, length(idx_bad)  * 0.2)


        test_idx_good = idx_good[randperm(length(idx_good))[1:n_test_good]]
        test_idx_bad  = idx_bad[randperm(length(idx_bad))[1:n_test_bad]]

        test_idx = vcat(test_idx_good, test_idx_bad)
        train_idx = setdiff(1:length(y), test_idx)

        x_train = x[train_idx, :]'
        y_train = y[train_idx, :, 1]'

        x_test  = x[test_idx, :]'
        y_test  = y[test_idx, :, 1]'

        number_of_classes = 1
        
        filepath_train="prepared_data/default_of_credit_card_clients_train.jld2"
        filepath_test="prepared_data/default_of_credit_card_clients_test.jld2"
        @save filepath_train x_train y_train number_of_classes
        @save filepath_test x_test y_test number_of_classes

end