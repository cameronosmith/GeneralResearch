#file to visualize the ccm data matrix 

library(lattice)

#plot the ccm matrix data of each neuron section cross predicting each other

ccm_matrix_data <- data.matrix( read.csv("../data/Fly80_CCM_Rho_tau200.csv") )
ccm_matrix_data <- ccm_matrix_data[ , (2:ncol(ccm_matrix_data)) ]
plot_col_names  <- 1:ncol(ccm_matrix_data)
dimnames( ccm_matrix_data ) <- list( plot_col_names, plot_col_names )

levelplot( ccm_matrix_data )
