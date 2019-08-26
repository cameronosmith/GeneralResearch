#file to generate the lorenz data we're looking at
import numpy as np

#takes in curr point and moves in lorenz direction
def lorenz_step ( curr_point, time_step, sigma=10, beta=8/3, rho=28 ) :
	x,y,z 		= curr_point
	dx,dy,dz 	= sigma*(y - x), x*(rho - z) - y, x*y - beta*z	
	deltas 		= [diff * time_step for diff in [dx,dy,dz]]
	new_point 	= [round(sum(x),2) for x in zip(deltas, curr_point)]
	return	new_point
#function to get the output of the series one step ahead
#@return : the 2d list with the lorenz data
def sim_lorenz ( start_state = [2,3,4], time_step = .01, num_steps = 2000):
	lorenz_data = [ start_state ]
	for idx in range(1,num_steps):
		prev_state 	= lorenz_data[ idx-1 ]
		new_state	= lorenz_step( prev_state, time_step )
		lorenz_data.append( new_state )
	return np.array(lorenz_data)
