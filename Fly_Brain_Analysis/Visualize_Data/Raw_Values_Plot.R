#file to visualize the raw data frames

library(lattice)

scaled_data <- read.csv("../data/Scaled_Fly80XY.csv")
raw_data    <- read.csv("../data/Fly80XY.csv")

fly_coord   <- scaled_data[,c('Left_Right','FWD')]

neurons     <- read.csv("../data/Diffed_Scaled_Fly80XY.csv")[,1:80]
neurons     <- neurons[1:50,]

neurons     <- 100*data.frame( diff( as.matrix( neurons ) ) )

for ( row_idx in 1:20 ) {

    trellis.device(device="png", filename=paste("imgs/",
                sprintf("%06d", row_idx),".png",sep='') )
    #format neuron activity as 9x9 grid (missing last cell) 
    neuron_row  <- as.numeric( neurons[ row_idx, ] )
    neuron_row[length(neuron_row)+1] <- 0
    neuron_mat  <- matrix( neuron_row, nrow = 9, byrow = TRUE )
    heat_plot   <- levelplot( neuron_mat )
    print( heat_plot )

    #barplot( as.numeric( neurons[row_idx,] ),ylim=c(min(neurons),max(neurons)))
    #dev.off()

    #plot( curr_xy_points$Left_Right, curr_xy_points$FWD,
    #        xlim = x_range, ylim = y_range )
    
}

