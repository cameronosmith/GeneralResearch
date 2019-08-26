#define the system consts
sigma 		<- 10
beta  		<- 8/3
rho 		<- 28

#function to get the output of the series one step ahead
#@param curr_state 	: current x,y,z state as list
#@return 			: the next state of the system (x,y,z)
lorenz_step <- function ( curr_state, time_step ) { 
	x <- curr_state[1]; y <- curr_state[2]; z<- curr_state[3];
	dx <- sigma*(y - x); dy <- x*(rho - z) - y; dz <- x*y - beta*z;
	diffs <- c(dx,dy,dz)
	return (diffs*time_step+curr_state)
}

#function to get the output of the series one step ahead
#@return : the dataframe with the lorenz data
sim_lorenz <- function( start_state = c(2,3,4), time_step = .01, num_steps = 2000) {
	lorenz_data 		 	<- matrix(0, ncol = length(start_state), nrow = num_steps)
	colnames(lorenz_data) 	<- c("x","y","z")
	lorenz_data[1,] 		<- start_state
	for ( row_idx in 2:nrow(lorenz_data) ) {
		prev_state 				<- lorenz_data[row_idx-1,]	
		new_state				<- lorenz_step( prev_state, time_step )
		lorenz_data[row_idx,]	<- new_state
	}
	return ( as.data.frame(lorenz_data) )
}
