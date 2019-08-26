#method to test our RNN on some data
from RNN import RNN
import numpy as np

###########get_data_section######################
##########run_machine_section####################
#train
word_dim = 4
def one_hot ( idx ) :
    output = np.zeros( word_dim )
    output[ idx ] = 1
    return output

input_len, output_len   =   4, 4
num_training_steps      =   10000

machine = RNN( input_len, output_len )

input_indices   = [0,1,2,2]
output_indices  = [1,2,2,3]
input_vec       = [one_hot(idx) for idx in input_indices ]
output_vec      = [one_hot(idx) for idx in output_indices]

for _ in range(num_training_steps):
    machine.train( input_vec, output_vec ) 
