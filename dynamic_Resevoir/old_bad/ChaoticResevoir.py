#file contains the implementation of the actual resevoir (middle nodes stuff)
#also contains a mini node class which is the base unit in the resevoir
import numpy as np
import scipy.linalg as linalg
import matplotlib.pylab as plt

range_shift = .3

#num nodes in each state
num_states_in_node = 3

#helper method to scale our data between a and b so lorenz isn't too big
#@param dataset : the dataset
#@return        : the scaled dataset
def scale_input(dataset):
    min_val = min( dataset ) 
    max_val = max( dataset ) 
    top     = 40
    bot     = -40
    scaled  = []
    for data in dataset : 
        scaled.append( bot + (top-bot)*(data-min_val)/(max_val-min_val) )
    return scaled

#method to simulate the lorenz system forward
#@params x,y,z          : the x,y,z coord of curr state
#@params sigma,beta,rho : 3 initial params of the lorenz system
#@param  time_step      : how much to move in this direction
#@return                : the new x,y,z state as numpy array
def lorenz_step ( x,y,z, sigma=10,beta=8/3,rho=28, time_step=.5 ) :

    x,y,z=x[0],y[0],z[0]

    #calculate differentials and apply as time step
    dx,dy,dz 	= sigma*(y - x), x*(rho - z) - y, x*y - beta*z	
    deltas      = [diff * time_step for diff in [dx,dy,dz]]
    new_point 	= [round(sum(i),2) for i in zip(deltas, [x,y,z])]

    return np.array( new_point )

class Resevoir : 

    #class to be the basic math unit of the resevoir
    #each unit is basically 3 nodes (since we have 3 observers in lorenz)
    class ChaoticUnit : 
        
        #constructor
        #resevoir is the resevoir we're attached to
        def __init__( self, resevoir ):

            self.resevoir = resevoir

            #the hidden states for this node ( 3 observers in lorenz )
            self.state    = np.ones( num_states_in_node )
            self.lorenz_init_conditions = np.array( [10,8/3,28] )

        #method to update the hidden state for this node
        #@param weighted_data: the weights_in * data since not unique to node
        #@param adj_nodes    : list of weights (row of adj matrix)
        #@return             : none, just updates this node's hidden state
        def run ( self, weighted_data, weights_row ):
            
            #the collected aggregate of val*weight of hidden inputs from adj's 
            #adj_sum = np.zeros( num_states_in_node )

            state = lorenz_step( *weighted_data )

            #get the inputs from all of the nodes that feed into here
            #for node_idx, weight in enumerate( weights_row ):

            #    adj_node = self.resevoir.res_nodes[ node_idx ]
            #    adj_sum += adj_node.state * weight

            #perform update to our hidden 
            #hidden_update   = weighted_data + adj_sum
            self.state      = (1-self.resevoir.leak_rate)*self.state + \
                                self.resevoir.leak_rate*hidden_update
            #self.state = lorenz_step( *self.state, *self.lorenz_init_conditions )
    
    #constructor
    #@param res_size     : the number of hidden nodes to use    
    #@param connectivity : the connectivity of the nodes as float
    #@param leak_rate    : how much the nodes states should flow into next
    def __init__( self, res_size, connectivity,  leak_rate ):

        #we need at least 3 nodes and has to be mult of 3
        if res_size < num_states_in_node : 
            res_size = num_states_in_node
        mod = res_size % num_states_in_node
        if mod != 0 :
            res_size += num_states_in_node - mod

        self.res_size  = res_size 
        self.leak_rate = leak_rate
        self.num_units = res_size // num_states_in_node #num chaotic units

        #create our weight matrix and res nodes
        self.res_nodes    = [self.ChaoticUnit(self) 
                                    for _ in range(self.num_units)]
        self.res_weights  = self.init_res_weights(self.res_nodes, connectivity)
        
        #limit the spectral radius of our node weights
        print("computing max eigen val...")
        max_eigen         = max( abs( linalg.eig( self.res_weights )[0] ) ) 
        self.res_weights /= max_eigen
        print("done limiting spectral radius (max eigen)")
    
    #helper method to setup adjacency matrix of the resevoir aka node weights
    #@param res_nodes   : the nodes in the resevoir
    #@param connectivity: float - the percentage of nodes a node should be conn.
    #@param plot        : true if we want to show plot of the adjacency matrix
    #@return            : the adj. matrix of nodes.connection is [0,1)
    def init_res_weights( self, res_nodes, connectivity, plot=False ):

        #setup weights from default to [0,1) to [.5,1.5] then to [.5,1]
        weights = np.random.rand( len(res_nodes),len(res_nodes) ) + range_shift
        weights = np.clip( weights, None, 1 )

        #pick the rand indices each node will drop (limiting connectivity)
        num_nodes_dropped = round( len(res_nodes) * (1-connectivity) )
        for row in weights : 
            drop_indices = np.random.choice( len(res_nodes), \
                                        num_nodes_dropped, replace = False )
            row[ drop_indices ] = 0

        #plot adj matrix for illustration if desired
        if plot :
            plt.imshow(weights)
            plt.colorbar()
            plt.show()

        return weights
    
    #method to run the resevoir (throw data into the resevoir) 
    #@param weights_in  : the input weights
    #@param data_t      : the data to pass through the resevoir on 
    #@return            : none 
    def run (self, weights_in, data_t ):
        
        #compute here so every node doesn't have to compute it
        weighted_data = np.dot( weights_in, data_t )

        #run each node to get new activation states
        for row_i, node_i in enumerate( self.res_nodes ) :
            node_i.run( weighted_data, self.res_weights[ row_i, : ] )

    #getter for the array of hidden states
    #@return : the (res_sizeX1) matrix of hidden states
    def get_hidden_states( self ):

        #format hidden states as np array
        raw_states = []
        for node in self.res_nodes:
            for observer in node.state:
                raw_states.append( observer )

        formatted = np.array( raw_states ).reshape( \
                                    (self.res_size, 1) )

        return formatted


