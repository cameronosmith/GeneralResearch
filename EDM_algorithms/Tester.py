#file to run the algos
from EDM_algos import Simplex, SMap
from AuxFuncs import FindKNN
import LorenzGen
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#get our test data
lorenz_data = LorenzGen.sim_lorenz()
query_point = lorenz_data[0]
qp_x, qp_y, qp_z = [query_point[0]],[query_point[1]],[query_point[2]]
nn = FindKNN( query_point,10,lorenz_data )
neighb_x,neighb_y,neighb_z = nn[:,0],nn[:,1],nn[:,2]

#plot raw lorenz data for study 
if ( False ):
	x,y,z 	= lorenz_data[:,0],lorenz_data[:,1],lorenz_data[:,2]
	fig 	= plt.figure()
	ax 		= fig.add_subplot(111, projection='3d')
	ax.plot3D(x, y, z, c='r', marker='o')
	ax.plot3D(neighb_x, neighb_y, neighb_z, c='g', marker='o')
	ax.plot3D(qp_x, qp_y, qp_z, c='b', marker='o')
	plt.show()
#plot nn
lib = (0,500)
pred= (501,502)
#Simplex(lorenz_data, 3, lib, pred )
SMap(lorenz_data, 4, 1, lib, pred )
