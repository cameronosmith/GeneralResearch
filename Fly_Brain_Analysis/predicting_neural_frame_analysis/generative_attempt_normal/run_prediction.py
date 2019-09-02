#file to run the network with our data
import load_data
import err
from pyESN import pyESN
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#network/data consts

generative  = True #run prediction from last true pt or from last pred pt

train_len   = 5900
test_len    = train_len

train_idx   = np.index_exp[:train_len]
test_idx    = np.index_exp[:train_len]
pred_cols   = np.index_exp[:,0:1]

#get/format our data

"""
data        = load_data.get_data()
neuron_data = data['neuron_data']
fut_brain   = data['future_neuron_data']

train_in        = neuron_data[train_idx][:,0:1]
train_out_truth = fut_brain[train_idx][:,0:1]
test_in         = neuron_data[test_idx][:,0:1]
test_out_truth  = fut_brain[test_idx][:,0:1]
"""

temp_1_col      = pd.read_csv("../../data/temp_1_col.csv").to_numpy()
temp_1_col_fut  = temp_1_col[1:,]
train_in        = temp_1_col[train_idx][:,0:1]
train_out_truth = temp_1_col_fut[train_idx][:,0:1]
test_in         = temp_1_col[test_idx][:,0:1]
test_out_truth  = temp_1_col_fut[test_idx][:,0:1]

#make our network and run dat dynamical boi

esn = pyESN.ESN(  n_inputs  = train_in.shape[1],
                  n_outputs = train_out_truth.shape[1],
                  n_reservoir = 100,
                  spectral_radius = .9,
                  noise = .01,
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

#get predictions on either true prev points or predicted last point

if not generative :

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

else :

    #stored prediction points. we start with the true point. 

    predictions = np.zeros( (test_len, test_out_truth.shape[1]) )

    predictions[ 0 ] = test_out_truth[ 0:1, ]
    
    #run the prediction machine

    for pred_idx in range( 1, test_len ) : 

        last_pred_point = predictions[ (pred_idx-1):pred_idx, ] 
        prediction      = esn.predict( last_pred_point ) 
        #prediction      = np.clip( prediction, data_min, data_max )
        predictions [ pred_idx ] = prediction

    print(predictions[1:20])
    print(predictions[:-20])

    test_err = err.get_err( predictions, test_out_truth )
    print( "err on trained is ", test_err, " . " )

    bias = predictions[101] - test_out_truth[101]
    print("bias is ",bias)

    new_max_val = np.amax( test_out_truth )
    new_min_val = np.amin( test_out_truth )
    old_max_val = np.amax( predictions )
    old_min_val = np.amin( predictions )
    
    print("followare maxes")
    print(np.amax(test_out_truth))
    print(np.amax(predictions))
    for idx in range( predictions.shape[0] ) : 
        tmp = (predictions[idx]-old_min_val)/(old_max_val-old_min_val)
        predictions[idx] = (new_max_val-new_min_val) * tmp * new_min_val

    print("followare maxes")
    print(np.amax(test_out_truth))
    print(np.amax(predictions))

    for pred_col in range( test_out_truth.shape[1] ) :

        plt.title("testing results for col "+str(pred_col))
        plt.plot( (predictions[100:,pred_col]-bias), label="machine pred" )
        plt.plot( test_out_truth[100:,pred_col], label="truth" )
        plt.legend( loc='best' )
        plt.show()
