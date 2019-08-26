#file for some auxilliary functions used by the algorithms
import scipy.spatial.distance as distance
import copy
import numpy as np

#nearest neighbor search - current computes dist to every point in df
#@params point			: point to base location
#@params num_neighbors	: k - number of neighbors to find
#@params data			: library to search in. should be in form of (idx,val)
#@return				: list of knn indices
def FindKNN( point, num_neighbors, data ) :
	#get neighbors into form of (idx,val,dist)
	new_lib_form = []
	for neighb in data :
		dist = abs(distance.euclidean(point,neighb[1]))
		new_lib_form.append( (neighb[0],neighb[1],dist) )
	nearest_points 	= sorted( new_lib_form, key=lambda x:x[2] )
	#sort based on dist column
	return np.array(nearest_points[0:num_neighbors])
