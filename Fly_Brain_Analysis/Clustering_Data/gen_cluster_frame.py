# file to take the cluster indices for each neuron node and gen activity frame

import make_cluster
import numpy as np
from matplotlib import pyplot as plt 

# get cluster arrangements. separate each cluster into its own array

cluster_map = np.array( make_cluster.cluster_map )

sorted_indices = cluster_map[ cluster_map[:,1].argsort() ]

# method to take in 80 neurons activity and format in order of given nodes
def format_neurons ( activity, make_image=False ) :

    formatted_activity = activity[ sorted_indices[:,0] ]

    return formatted_activity

# append a -1 so we can have 9x9 grid with 80 neurons. reshape to 2d grid.

format_neurons( np.arange(80) )

# optional plot of the cluster groupings

if False :

    cluster_grids = np.vstack( [sorted_indices, [-1,-1]] )
    cluster_grids = cluster_grids[:,1].reshape( -1, 9 )
    print(cluster_grids)

    #concatenate and plot
    plt.imshow( cluster_grids )
    plt.colorbar()
    plt.show()
