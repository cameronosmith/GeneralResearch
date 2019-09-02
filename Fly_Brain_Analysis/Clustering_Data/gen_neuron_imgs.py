import os, tempfile, shutil
import numpy as np, pandas as pd
from matplotlib import pyplot as plt 
import matplotlib.animation as animation 

# function to animate a row of brain data at a time
# returns a collection of plt images

def activity_animation ( brain_data, animate = False, write_path = "" ) :
    
    imgs = []

    for idx, activity in enumerate( brain_data ) :

        plt.axis('off')
        activity = np.append( activity, activity[1] ).reshape( 9, 9 )
        img = plt.imshow( activity, animated=animate ) 
        imgs.append( [ img ] )
        
        if write_path is not "" :
            print(idx) 
            plt.savefig( write_path+str(idx), transparent=True )
            
        plt.close()        

    if animate :

        plt.colorbar()
        ani = animation.ArtistAnimation(plt.figure(), imgs, interval=50, blit=True )
        plt.show()
    
#consts

num_brain_cols = 80

#load in our data

unclustered_brain = pd.read_csv('../data/Smaller_Scaled_Fly80XY.csv').to_numpy()
unclustered_brain = unclustered_brain[:,0:num_brain_cols]
clustered_brain   = pd.read_csv('../data/Clustered_Scaled_Fly80XY.csv').to_numpy()

unclustered_frames= activity_animation( unclustered_brain[:10], animate = False )
clustered_frames  = activity_animation( clustered_brain[:10]  , animate = False )

#write the data or not
write_data = True

if write_data :

    image_set_names = [ "clustered_neurons_frames","unclustered_neurons_frames" ]
    data_sets       = [ unclustered_brain, clustered_brain ]

    for image_set_name, data_set in zip( image_set_names, data_sets ) :

        # make imgs directories. remove if already exists. tmp needed for speed

        image_dir_path = "images/" + image_set_name
        
        if os.path.exists( image_dir_path ) : 

            tmp = tempfile.mktemp(dir=os.path.dirname( image_dir_path ))
            shutil.move( image_dir_path, tmp)
            shutil.rmtree(tmp)

        os.makedirs( image_dir_path )
        
        # write the actual frames
        activity_animation ( data_set[:5], 
                             write_path = image_dir_path+"/" )
    

