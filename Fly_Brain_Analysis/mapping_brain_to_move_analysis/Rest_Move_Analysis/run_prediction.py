#file to run the network with our data

import load_data
import err
from pyESN import pyESN
import numpy as np
from matplotlib import pyplot as plt

#network/data consts

train_len   = 2000
test_len    = 2000

train_idx   = np.index_exp[:train_len]
test_idx    = np.index_exp[train_len:train_len+test_len]

#get/format our data

data                 = load_data.get_data()
move_data            = data['move_data'][:,0:1]
neuron_data          = data['neuron_data']
classified_move_data = data['classified_move_data'][:,1:2]

train_in        = neuron_data[train_idx]
train_out_truth = classified_move_data[train_idx]
test_in         = neuron_data[test_idx]
test_out_truth  = classified_move_data[test_idx]

#make our network and run dat dynamical boi

esn = pyESN.ESN(  n_inputs  = train_in.shape[1],
                  n_outputs = train_out_truth.shape[1],
                  n_reservoir = 1000,
                  spectral_radius = .5,
                  teacher_forcing = False,
                  noise = .001,
                  sparsity = .42,
                  random_state = 42 )

print( "training network..." )
machine_training_out = esn.fit(train_in, train_out_truth, inspect=False)
print( "done training network." )

plt.title("training results")
plt.plot( machine_training_out[:,0], label="machine pred" )
plt.plot( train_out_truth[:,0], label="truth" )
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

plt.title("testing results")
plt.plot( machine_pred[:,0], label="machine pred" )
plt.plot( test_out_truth[:,0], label="truth" )
plt.legend( loc='best' )
plt.show()
