#file to load our brain data and corresponding fly movement

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import distance

#load in our data
input_cols  = pd.read_csv('../../data/manually_selected_in_out_cols/input_df_neuron_cols.csv')
output_cols = pd.read_csv('../../data/manually_selected_in_out_cols/output_df_neuron_cols.csv')

#getter for our data
def get_data() : 
    return { 'input':input_cols.to_numpy(),
             'output':output_cols.to_numpy() }
