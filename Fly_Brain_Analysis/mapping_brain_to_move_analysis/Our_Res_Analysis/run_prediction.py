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

train_len   = 3000
test_len    = 3000

train_idx   = np.index_exp[:train_len]
test_idx    = np.index_exp[train_len:train_len+test_len]

#get/format our data

data        = load_data.get_data()
move_data   = data['move_data'][:,0:1]
neuron_data = data['neuron_data'][:,0:20]

train_in        = neuron_data[train_idx]
train_out_truth = move_data[train_idx]
test_in         = neuron_data[test_idx]
test_out_truth  = move_data[test_idx]

#make our network and run dat dynamical boi

esn = EchoStateNetwork.EchoStateNetwork( train_in[0].shape[0], 
                                         train_out_truth[0].shape[0], 
                                         res_size = 1000,
                                         leak_rate = .3,
                                         spectral_radius = .9  )

esn.train( train_in, train_out_truth )

print( "training network..." )
machine_out, err = esn.test( test_in, test_out_truth )
print( "done training network." )
print("error is: ",err)

plt.plot(machine_out[:,0], label="machine outputs")
plt.plot(test_out_truth, label="truth")
plt.legend( loc='best' )
plt.show()


