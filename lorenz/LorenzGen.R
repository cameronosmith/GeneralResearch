#define the system consts
sigma 		<- 10
beta  		<- 8/3
rho 		<- 28
start_state <- c(x = 2, y = 3, z = 4)
#non-system simulation consts
time_step 	<- .01
num_steps 	<- 2500

#function to get the output of the series one step ahead
#@param curr_state 	: current x,y,z state as list
#@return 			: the next state of the system (x,y,z)
lorenz_step <- function ( curr_state ) { 
	x <- curr_state[1]; y <- curr_state[2]; z<- curr_state[3];
	dx <- sigma*(y - x); dy <- x*(rho - z) - y; dz <- x*y - beta*z;
	diffs <- c(dx,dy,dz)
	return (diffs*time_step+curr_state)
}

#function to get the output of the series one step ahead
#@return : the dataframe with the lorenz data
sim_lorenz <- function() {
	lorenz_data 		 	<- matrix(0, ncol = length(start_state), nrow = num_steps)
	colnames(lorenz_data) 	<- c("x","y","z")
	lorenz_data[1,] 		<- start_state
	for ( row_idx in 2:nrow(lorenz_data) ) {
		prev_state 				<- lorenz_data[row_idx-1,]	
		new_state				<- lorenz_step( prev_state )
		lorenz_data[row_idx,]	<- new_state
	}
	return ( as.data.frame(lorenz_data) )
}

lorenz_data <- sim_lorenz()
library(rgl)
plot3d(lorenz_data$x,lorenz_data$y,lorenz_data$z)
snapshot3d("lorenz_full_system.png")
