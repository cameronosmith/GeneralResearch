#file to run the network with our data
import load_data
import err
from pyESN import pyESN
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#network/data consts

train_len   = 3000
test_len    = 1000

train_idx   = np.index_exp[:train_len]
test_idx    = np.index_exp[:test_len]

#get/format our data

data         = load_data.get_data()
input_data   = data['input']
output_data  = data['output']

train_in        = input_data[train_idx]
train_out_truth = output_data[train_idx]
test_in         = input_data[test_idx]
test_out_truth  = output_data[test_idx]

#make our network and run dat dynamical boi

esn = pyESN.ESN(  n_inputs  = train_in.shape[1],
                  n_outputs = train_out_truth.shape[1],
                  n_reservoir = 700,
                  spectral_radius = .8,
                  noise = .08,
                  silent = False,
                  sparsity = .92,
                  random_state = 42 )

print( "training network..." )
machine_training_out = esn.fit(train_in, train_out_truth, inspect=False)
print( "done training network." )

for pred_col in range( 1 ):#train_out_truth.shape[1] ) :

    plt.title("training results for col "+str(pred_col))
    plt.plot( machine_training_out[:,pred_col], label="machine pred" )
    plt.plot( train_out_truth[:,pred_col], label="truth" )
    plt.legend( loc='best' )
    plt.show()

#compute our err on trained output (should be small)

print( "computing training error..." )
training_err = err.get_err( train_out_truth, machine_training_out )
print( "err on trained is ", training_err, " . " )

#try running with test data

print( "running on test data... " )
machine_pred = esn.predict( test_in )

print( "computing training error..." )
test_err = err.get_err( machine_pred, test_out_truth )
print( "err on trained is ", test_err, " . " )

for pred_col in range( test_out_truth.shape[1] ) :

    plt.title("testing results for col "+str(pred_col))
    plt.plot( machine_pred[:,pred_col], label="machine pred" )
    plt.plot( test_out_truth[:,pred_col], label="truth" )
    plt.legend( loc='best' )
    plt.show()
