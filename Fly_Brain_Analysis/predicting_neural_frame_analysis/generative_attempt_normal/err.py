from scipy.spatial import distance
from matplotlib import pyplot as plt

#method to compute error between numpy arrays with each row as tuple
def get_err( true_out, pred_out, plot=False ):
    if true_out.shape!=pred_out.shape :
        print("bad input dimensions to compute err: ",true_out.shape,pred_out.shape)

    aggreg_dist = 0

    for row_idx in range( len(true_out) ):
        true_point  = tuple( true_out[ row_idx ] )
        pred_point  = tuple( pred_out[ row_idx ] )
        curr_dist   = distance.euclidean( true_point, pred_point )
        aggreg_dist += curr_dist if curr_dist > 0 else -curr_dist

    if plot:
        plt.plot( true_out[:,0],color="red",label="true" )
        plt.plot( pred_out[:,0],color="blue",label="pred" )
        plt.legend(loc='best')
        plt.show()
        #plt.show()
        #plt.plot( true_out[1] ) 
        #plt.plot( pred_out[1] ) 
        #plt.show()
    
    return aggreg_dist / len( true_out )

