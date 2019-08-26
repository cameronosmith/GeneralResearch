#file just has the rnn node/layer class
import numpy as np
from ActivationAndOutput import Tanh

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
    #return             :  deltas in order of d_nodes, d_out, d_in
    def back_prop( self, error, node_weights, weights_in, weights_out ):

        #used for derivatives below
        d_tanh  = Tanh.deriv( self.pre_act_hidden )

        #deriv of w_out.(dE/dV). doesn't depend on prev terms
        d_out   = np.dot(self.hidden_state, error).T 

        #deriv of w_nodes is dE/dy(=err) * dy/dst(=V) * st/st-1.. * st-k/W
        #first term 
        d_nodes = np.dot( error, weights_out )
        #iterate all middle states (not last) to get the derivs of hiddens
        tmp_node= self
        while tmp_node.prev_node is not None :
            pass
            #tmp_node = tmp_node.prev_node
        #d_nodes     = np.dot(tmp_node.first_node_prev_h, d_nodes )
        print("pre is ",d_nodes.shape)
        
        last_deriv  = np.dot( Tanh.deriv(tmp_node.pre_act_hidden).T,
                                tmp_node.first_node_prev_h)
        print("deriv is ",last_deriv.shape)
                              
        return d_nodes, d_out, np.zeros( weights_out.shape )

    #method to get output from this node
    #@param input_data    :  data to run the computation on. should be 1xlen
    #@param ___weights    :  the global weights of the rnn
    #@param output        :  the output vec (raw, not softmaxed)
    def run(self, input_data, node_weights, weights_in, weights_out):
        if self.prev_node is None:
            self.first_node_prev_h = np.ones(self.hidden_state.shape)
        #update hidden state: hidden*nodes_weight + input weight*input
        prev_hidden         = self.prev_node.hidden_state if self.prev_node\
                              is not None else self.first_node_prev_h
        self.pre_act_hidden = np.dot(node_weights,prev_hidden) +\
                                     np.dot(weights_in, input_data.T)
        #store the pre-activation hidden since we need it in backprop
        self.hidden_state   = Tanh.Tanh(self.pre_act_hidden)

        #compute output for this node
        output              = np.dot(weights_out,self.hidden_state).T

        return output

