source("LorenzGen.R")
data <- sim_lorenz(num_steps=2000)


library(rgl)
num_steps_animate <- 2000
png_files <- c()
for ( i in 1:num_steps_animate ) {
    plot3d( data$x[1:i],data$y[1:i],data$z[1:i] )
    filename <- paste("animation/",i,".png",sep="")
    png_files <- c(png_files, filename)
    rgl.snapshot(filename=filename)
}
png_files <- paste(png_files, collapse=" ")
system(paste("convert -delay 3 -loop 0 ",png_files," animated.gif" ))
