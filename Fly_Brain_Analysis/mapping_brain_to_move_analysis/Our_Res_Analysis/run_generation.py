#file to run the network with our data

import load_data
import err

import sys, os
sys.path.insert(0, os.getcwd()+"/efficient_esn/")
import EchoStateNetwork 

import numpy as np
from matplotlib import pyplot as plt

vec_tanh    = np.vectorize( lambda x : np.tanh( x ) )

#network/data consts

train_len   = 4000
test_len    = train_len

train_idx   = np.index_exp[:train_len]
test_idx    = np.index_exp[:train_len]

#get/format our data

data        = load_data.get_data()
move_data   = data['move_data'][:,0:1]
neuron_data = data['neuron_data'][:,0:1]
next_frame_neuron_data = neuron_data[1:,0:1]

train_in        = neuron_data[train_idx]
train_out_truth = move_data[train_idx]
test_in         = neuron_data[test_idx]
test_out_truth  = move_data[test_idx]

#make our network and run dat dynamical boi

esn = EchoStateNetwork.EchoStateNetwork( train_in[0].shape[0], 
                                         train_out_truth[0].shape[0], 
                                         res_size = 100,
                                         leak_rate = .9,
                                         spectral_radius = .8  )

print( "training network..." )
esn.train( train_in, train_out_truth )
print( "done training network." )

machine_out, err = esn.test( test_in, test_out_truth )
print("error is: ",err)

plt.plot(machine_out[:,0], label="machine outputs")
plt.plot(test_out_truth, label="truth")
plt.legend( loc='best' )
plt.show()


#stored prediction points. we start with the true point. 

predictions = np.zeros( (test_len, test_out_truth.shape[1]) )

predictions[ 0 ] = test_out_truth[ 0:1, ]

#run the prediction machine

for pred_idx in range( 1, test_len ) : 

    last_pred_point = predictions[ (pred_idx-1):pred_idx, ] 
    prediction, _   = esn.test( last_pred_point, last_pred_point ) 
    #prediction      = np.clip( prediction, data_min, data_max )
    predictions [ pred_idx ] = prediction

print(predictions[1:20])

#test_err = err.get_err( predictions, test_out_truth )
#print( "err on trained is ", test_err, " . " )

for pred_col in range( test_out_truth.shape[1] ) :

    plt.title("testing results for col "+str(pred_col))
    plt.plot( predictions[:,pred_col], label="machine pred" )
    plt.plot( test_out_truth[:,pred_col], label="truth" )
    plt.legend( loc='best' )
    plt.show()




