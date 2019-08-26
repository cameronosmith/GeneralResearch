import numpy as np
import load_data
import scipy
import scipy.linalg as linalg

#random consts and getters
range_shift     = .5
rand_weights    = lambda x,y : np.random.rand( x,y ) - range_shift
max_eigen       = 1
num_init_batches= 0
lucky_num       = 42

np.random.seed( lucky_num )

#load our data
train_input, train_truth = load_data.get_training_data()
test_input, test_truth   = load_data.get_training_data()

#properties of our ESN
input_size  = 1
output_size = 1
res_size    = 10
leak_rate   = .3

#init our random weights. input, node weights
W_in        = rand_weights( res_size, input_size )
W_nodes     = rand_weights( res_size, res_size )
W_out       = rand_weights( output_size, res_size )
res_states  = np.zeros( (res_size, 1) )

#limit the spectral radius of our node weights
max_eigen   = max( abs( linalg.eig( W_nodes )[0] ) ) 
W_nodes    /= max_eigen

#helper method to run the resevoir with an input data
#@param res_state   : the current activated hidden states
#@param data        : the data to run the resevoir on
#@return            : the new hidden states, and output of system (prediction)
def run_res ( res_states, data ):

    #get update and apply it to the current states
    hidden_update   = np.dot( W_in, data ) + np.dot( W_nodes, res_states )
    hidden_update   = np.tanh( hidden_update )
    new_hidden      = (1-leak_rate)*res_states + leak_rate*hidden_update
    #get output of system
    output          = np.dot( W_out, new_hidden )

    return new_hidden, output

#run the data on the testing data to collect our state matrix
states_matrix = np.zeros( (res_size, len(train_input)) )
for t, data_t in enumerate( train_input ) :
    res_states, _ = run_res( res_states, data_t )
    states_matrix[:,t] = res_states[:,0]

#train the output layer
reg = 1e-8
X   = states_matrix
X_T = X.T
Yt  = np.array(train_truth).reshape(1,len(train_truth))
term1 = np.dot(Yt,X_T)
term2 = linalg.inv( np.dot(X,X_T) + reg*scipy.eye(res_size) )
W_out = np.dot( term1, term2 )

#get system output and compute error
collected_outputs = []
for data_t in test_input :
    res_states, output = run_res( res_states, data_t )
    collected_outputs.append( output )
error = [ (x-y)**2 for x,y in zip( collected_outputs, test_truth ) ] 
mse   = np.sum( error ) / len( collected_outputs )
print( "mse is : ",mse )
