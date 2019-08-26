import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import matplotlib.animation as animation 


# function to animate a row of brain data at a time

def activity_animation ( brain_data ) :
    
    imgs = []

    for activity in brain_data :

        activity = np.append( activity, activity[1] ).reshape( 9, 9 )
        img = plt.imshow( activity, animated=True ) 
        imgs.append( [ img ] )

    plt.colorbar()
    ani = animation.ArtistAnimation(plt.figure(), imgs, interval=50, blit=True )
    plt.show()


#consts

num_brain_cols = 80

#load in our data

unclustered_brain = pd.read_csv('../data/Smaller_Scaled_Fly80XY.csv').to_numpy()
unclustered_brain = unclustered_brain[:,0:num_brain_cols]
clustered_brain   = pd.read_csv('../data/Clustered_Scaled_Fly80XY.csv').to_numpy()

activity_animation( unclustered_brain[:150] )
activity_animation( clustered_brain  [:150] )
