#file just has the rnn node/layer class
import numpy as np
from ActivationAndOutput import Softmax, Tanh

#the individual node/cell of the RNN
class RNN_Node: 
        
    #rnn node constructor
    #@param prev_node           :   the node before this one
    #@param hidden_state_shape  :   the shape of the hidden state
    def __init__( self, prev_node, hidden_state_shape ):
        self.hidden_state   =   np.zeros( hidden_state_shape )
        self.prev_node      =   prev_node

    #backprop an error thru time on prev layers
    #@param error       :  the error of the output from this node
    #@param ___weights  :  the global weights of the rnn
    def back_prop( self, error, global_node_weights, weights_in, weights_out ):
        #get deriv err with respect to output weights( dE/dV )
        dV = np.outer( error, self.hidden_state )
        #update the weights with our deltas 
        #weights_out -= dV * .01
        return dV

    #method to get output from this node
    #@param input_data    :  data to run the computation on. should be 1xlen
    #@param ___weights    :  the global weights of the rnn
    #@param output        :  the output vec (raw, not softmaxed)
    def run(self, input_data, global_node_weights, weights_in, weights_out):
        #update hidden state: hidden*nodes_weight + input weight*input
        prev_hidden         = self.prev_node.hidden_state if self.prev_node\
                              is not None else np.zeros(self.hidden_state.shape)
        tmp_hidden          = np.dot(global_node_weights,prev_hidden) +\
                                     np.dot(weights_in, input_data.T)
        self.hidden_state   = Tanh.Tanh(tmp_hidden)
        #compute output for this node
        output              = np.dot(weights_out,self.hidden_state).T
        return output

