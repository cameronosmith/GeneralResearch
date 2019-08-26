#file to load our brain data and corresponding fly movement
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import distance

#consts
num_brain_cols = 80

#load in our data
scaled_data          = pd.read_csv('../../data/Smaller_Scaled_Fly80XY.csv')
classified_move_data = pd.read_csv('../../data/ClassifiedMoveRest.csv')
brain_data           = scaled_data.iloc[:,0:4]
move_data            = scaled_data.iloc[:,num_brain_cols:]

#function to visualize our data
def visualize_data() :
    print("plotting brain activity data. patience, as we are working with \n"+
            "dataset of 10,000 time points * 80 neuron sections. \n each subplot "+
            "is the time series of one single neuron cluster.")
    half_cols = brain_data.shape[1]//2
    #brain_data.iloc[:,:half_cols].plot(subplots=True)
    plt.title("first half of neurons",loc="right",pad=40)
    brain_data.iloc[:,half_cols:].plot()#subplots=True)
    plt.title("second half of neurons",loc="right",pad=40)
    plt.show()
    print("plotting fly movement (differentials).\n top is left-right movement"+ 
            " and bottom is movement up.")
    move_data.plot()
    plt.show()

    print("just plotting rest vs active in this predictor")
    classified_move_data.iloc[:,0].plot()
    move_data.iloc[:,0].plot()
    plt.show()
    
#getter for our data
def get_data() : 
    return {'neuron_data':brain_data.to_numpy(),
            'classified_move_data':classified_move_data.to_numpy(),
            'move_data':move_data.to_numpy()}

if __name__ == "__main__":
    visualize_data()
