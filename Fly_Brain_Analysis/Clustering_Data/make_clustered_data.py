# file to actually turn the data into formatted data

import numpy as np
import pandas as pd
import gen_cluster_frame

#consts
num_brain_cols = 80

#load in our data
scaled_data = pd.read_csv('../data/Smaller_Scaled_Fly80XY.csv').to_numpy()
brain_data  = scaled_data[:,0:num_brain_cols]

# format the brain data as clustered
for row_idx in range( brain_data.shape[0] ) :

    activity    = brain_data[ row_idx ]
    formatted   = gen_cluster_frame.format_neurons( activity )
    brain_data[ row_idx ] = formatted

clustered_df = pd.DataFrame( brain_data )
clustered_df.to_csv("../data/Clustered_Scaled_Fly80XY.csv", index=False)
