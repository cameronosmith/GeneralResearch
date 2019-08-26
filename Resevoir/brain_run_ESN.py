#file to load our brain data and corresponding fly movement
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import distance
from EchoStateNetwork import EchoStateNetwork

#consts
num_brain_cols = 80

#load in our data

scaled_data = pd.read_csv('data/Smaller_Scaled_Fly80XY.csv').to_numpy()
brain_data  = scaled_data[:,0:num_brain_cols]
move_data   = scaled_data[:,num_brain_cols:]

train_len           = 1000
test_len            = 100
train_data_input    = brain_data[:train_len]
train_data_output   = move_data[:train_len]
test_data_input     = brain_data[train_len:train_len+test_len]
test_data_output    = move_data[train_len:train_len+test_len]

#create our ESN and run it
machine = EchoStateNetwork( train_data_input.shape[1], 
                            train_data_output.shape[1],
                            200 )
machine.train( train_data_input, train_data_output )
outputs, err = machine.test( test_data_input, test_data_output )

print("error is: ",err)
print("note: not mse, the difference x100 since small data points range")

#plot the outputs of our machine against the truth
time_seq = list(range(test_len))
plt.plot(time_seq, test_data_output[:,0], label="truth")
plt.legend(loc='best')
plt.figure()
plt.plot(time_seq, outputs[:,0], label="machine outputs")
plt.legend(loc='best')
plt.figure()
plt.show()
