library( rgl )
library( nonlinearTseries )
source("NonlinearGen.R")

num_steps    <- 500
lorenz_data  <- sim_lorenz( num_steps = num_steps )
rossler_data <- data.frame(nonlinearTseries::rossler())
rossler_data <- rossler_data[-1]

lorenz_data$color <- "red"
rossler_data$color <- "blue"

#setup data with shapes and colors for difference visual
agg_data <- rbind( lorenz_data, rossler_data )

plot3d( agg_data$x, agg_data$y, agg_data$z,
              col=agg_data$color )
