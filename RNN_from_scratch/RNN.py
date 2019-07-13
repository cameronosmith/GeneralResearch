#simple modular implementation of RNN
from RNN_Node import RNN_Node
import numpy as np
import math
from ActivationAndOutput import Softmax

#translation on weights to the classic rnn diagram:
#U = weights_in
#W = weights_hidden
#V = weights_out

#x is the dimension of the random matrix to generate
get_rand = lambda x : np.random.uniform( -1/np.sqrt(x[0]),-1/np.sqrt(x[1]), x )

#the actual RNN class and all it's behavior
class RNN:
            
    #how fast or slow the gradient descent deriv should be applied
    learning_rate = .005
    #num of hidden cells in our layer, and dim of hidden state
    num_hidden = 4
    hidden_state_dim = 2

    #rnn constructor
    #@param input_dim       :       input should be 1Xinput_dim
    #@param output_dim      :       output is 1Xoutput_dim
    def __init__( self, input_dim, output_dim ):
        self.input_dim, self.output_dim = input_dim, output_dim
        #initialize the initial weights of our network
        self.weights_IN         = get_rand( (self.hidden_state_dim, input_dim) )
        self.node_weights       = get_rand( (self.hidden_state_dim, \
                                             self.hidden_state_dim) )
        self.weights_OUT        = get_rand( (output_dim, self.hidden_state_dim) )
        #setup nodes
        self.nodes = []
        for _ in range( self.num_hidden ):
                prev_node = self.nodes[-1] if len(self.nodes)!=0 else None
                curr_node = RNN_Node( prev_node, (self.hidden_state_dim,1) )
                self.nodes.append( curr_node )

    #method to run the rnn on some input and get the output
    #@param inputs  :   list of sequential inputs
    #@return        :   the raw outputs from the machine (not softmaxed)
    def run( self, inputs ):
        #reshape the inputs to by 1xlen instead of 1d vectors
        inputs = [np.array(input_data).reshape(1,len(input_data)) for\
            input_data in inputs]
        #run the machine on every node layer to collect outputs
        outputs = []
        for time_idx in range( len(inputs) ):
            node_at_t   = self.nodes[ time_idx ]
            output      = node_at_t.run(
                                input_data          =inputs[time_idx], 
                                global_node_weights =self.node_weights,
                                weights_in          =self.weights_IN,
                                weights_out         =self.weights_OUT )
            outputs.append( output )
        return outputs

    #method to train the rnn on some single input (series of timesteps)
    #@param inputs          : one hot vectors as the inputs to the machine
    #@param truth_outputs   : one hot vectors as the truth outputs
    def train ( self, inputs, truth_outputs ):
        #the total cross entropy error over all timesteps
        total_xent_err = 0
        #run and collect errors from machine.
        machine_outputs = self.run( inputs )
        total_dv = np.zeros( self.weights_OUT.shape )
        #iterate time steps computing output and err for each then bptt
        for time_idx in range( len(inputs) ):
            machine_output  = machine_outputs[ time_idx ]
            truth_output    = truth_outputs[ time_idx ]
            total_xent_err  += Softmax.cross_entropy(
                                machine_output[0],truth_output)
            curr_err        = Softmax.raw_err(machine_output, truth_output)
            #iterate nodes backwards from this timestep to bptt
            d_err_out = np.outer(self.nodes[time_idx].hidden_state, curr_err).T
            total_dv += d_err_out
            """
            for node_t in reversed( self.nodes[:time_idx+1] ) : 
                node_t.back_prop(error               =curr_err, 
                                 global_node_weights =self.node_weights,
                                 weights_in          =self.weights_IN,
                                 weights_out         =self.weights_OUT )
            """
        self.weights_OUT -=  total_dv * .001
                
        print("total err was ",total_xent_err)
