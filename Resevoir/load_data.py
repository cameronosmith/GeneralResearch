import numpy as np

data        = np.loadtxt( "MackeyGlass_t17.txt" )
train_len   = data.shape[0]//2

train_data_input  = data[:train_len]
train_data_output = data[1:train_len+1]
test_data_input   = data[train_len:-1]
test_data_output  = data[train_len+1:]

#method to get our training data
#@return : data_input, data_output in 1xlen
def get_training_data() :
    return train_data_input, train_data_output

#method to get our testing data
#@return : data_input, data_output in 1xlen
def get_testing_data() :
    return test_data_input, test_data_output
