#file with the simplex algorithm
import AuxFuncs
import scipy.spatial.distance as dist
import math
import numpy as np, numpy.linalg as linalg

dist_col = 2

#SMap algorithm to estimate the next point 
#finds knn and estimates based on their next point
#@param df 	:	pandas df, the library to search in
#@param knn :	num neighbors to search for
#@param lib :	(start,end) for range of lib
#@param pred:	(start,end) for range of pred
#@param tp	:	the predict ahead amount for each point
#@return 	:	estimation of next point right after lib
def SMap ( data, knn, theta, lib, pred, tp=1 ) :
	lib_data = data[ lib[0] : lib[1] ]
	#create copy of lib in form of (idx, val)
	tmp = []
	for lib_idx, lib_val in enumerate(lib_data):
		tmp.append( (lib_idx, lib_val) )
	lib_data 	= tmp
	#predict for each point in pred
	predictions = []
	for pred_idx in range(pred[0],pred[1]):
		query_point 	= data[ pred_idx ]
		#get k nearest neighbors ( as indices )
		nearest_neighbs	= AuxFuncs.FindKNN( query_point, knn, lib_data )
		#get sum of distances		
		dist_sum	= sum(neighb[dist_col] for neighb in nearest_neighbs)
		#compute parallel weights for each neighb (where theta is used )
		weights 	= []
		for neighb in nearest_neighbs:
			weight 	= math.exp( (-theta)*neighb[dist_col]/dist_sum )
			weights.append( weight )
		#get weights as a diagonal matrix for mult
		weights 	= np.identity(len(weights))*weights
		#matrix of 1,Xn,Xn-1,..n where n=dimension for each neigb
		#matrix 		= np.array([np.concatenate(([1],neighb[1])) 
				#for neighb in nearest_neighbs])
		matrix 		= np.array([(neighb[1])
				for neighb in nearest_neighbs])
		weighted_matrix	= np.dot(weights, matrix)
		#get next points for each neighb into vec. then weight it.
		knn_nexts 	= np.array([lib_data[neighb[0]+tp][1] 
				for neighb in nearest_neighbs])
		weighted_nexts	= np.dot( weights, knn_nexts )
		#solve Ax=b, b is response vec of next pts. 
		sol			= linalg.lstsq( weighted_matrix, weighted_nexts )[0]
		#iterate linear model to get our prediction. c0+sig(ci*yi)
		prediction = 0
		print("initial is ",prediction)
		for row_idx in range(0,sol.shape[0]):
			new_val = (np.multiply(sol[row_idx],nearest_neighbs[row_idx][1]))
			prediction += new_val
		print("true next pt is ",data[pred_idx+tp])
		print("prediction is ",prediction)
		
	
#Simplex algorithm to estimate the next point 
#finds knn and estimates based on their next point
#@param df 	:	pandas df, the library to search in
#@param knn :	num neighbors to search for
#@param lib :	(start,end) for range of lib
#@param pred:	(start,end) for range of pred
#@param tp	:	the predict ahead amount for each point
#@return 	:	estimation of next point right after lib
def Simplex ( data, knn, lib, pred, tp=1 ) :
	lib_data = data[ lib[0] : lib[1] ]
	#create copy of lib in form of (idx, val)
	tmp = []
	for lib_idx, lib_val in enumerate(lib_data):
		tmp.append( (lib_idx, lib_val) )
	lib_data = tmp
	#predict for each point in pred
	predictions = []
	for pred_idx in range(pred[0],pred[1]):
		query_point 	= data[ pred_idx ]
		#get k nearest neighbors ( as indices )
		nearest_neighbs	= AuxFuncs.FindKNN( query_point, knn, lib_data )
		#set distance scale (from closest point)
		dist_scale 		= nearest_neighbs[0][2]
		#get next pt of each neighb mult by its weight and add to sum
		weights_sum 	= 0
		next_pt_sum 	= 0
		for neighb in nearest_neighbs:
			neighb_dist	= neighb[2]
			weight 		= math.exp( -neighb_dist/dist_scale )
			neighb_next = lib_data[neighb[0]+tp][1]
			weights_sum += ( weight )
			next_pt_sum += ( neighb_next * weight )
		#now we have the average of predictions = our prediction
		prediction	 	= next_pt_sum / weights_sum
		predictions.append( prediction )	
	return predictions
