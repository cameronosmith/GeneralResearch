#file to take the CCM (convergent cross mapping) results and construt a map of 
#the neurons where causal variables are closer to each other

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
import scipy.cluster.hierarchy as h_cluster
import scipy.spatial.distance as dist


# get our ccm data ( matrix of causality between the 80 neurons )

ccm_matrix       = pd.read_csv("../data/Fly80_CCM_Rho_tau200.csv").to_numpy()
condensed_matrix = ccm_matrix[np.triu_indices(ccm_matrix.shape[0],1)]

#since we are giving linkage a condensed distance matrix, and we have rho vals
#subtract by 1 so high rho values are now low distances

condensed_matrix[condensed_matrix < 0] = 0
condensed_matrix = 1-condensed_matrix

# raw ccm matrix conditional plot.

if False :

    plt.imshow( ccm_matrix )
    plt.colorbar()
    plt.show()

# clusters dists ( not the actual clusters yet )

cluster_info = h_cluster.linkage( condensed_matrix, method="ward" ) 

# calculate the error of the linkage

cluster_accuracy, _ = h_cluster.cophenet( cluster_info, condensed_matrix )
print("cluster accuracy is " , cluster_accuracy)

# dendrogram plot conditional (used to determine dist for clusters cutoff)

if False :

    h_cluster.dendrogram( cluster_info, leaf_rotation=90., leaf_font_size=8. )
    plt.show()

# identify actual cluster indices of the data (which cluster idx for each point)

cluster_indices = h_cluster.fcluster( cluster_info, 1.3, "distance" )
print("cluster indices: ",cluster_indices)
max_cluster = max( cluster_indices )
print("num clusters is ", max_cluster)

if False :

    plt.scatter( list(range(ccm_matrix.shape[0])), cluster_indices )
    plt.show()

# separate clusters for plotting 

cluster_map = []

for node_idx, cluster_idx in enumerate( cluster_indices ) :

    cluster_map.append( [ node_idx, cluster_idx ] )
