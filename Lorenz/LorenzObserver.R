library(rEDM)
library(rEDMNew)
source("LorenzGen.R")

lorenz_full <- sim_lorenz(time_step=.02,num_steps=2000)
observer    <- subset( lorenz_full, select = -c(y, z) )$x

#no lag ( x, time )
plot(observer)

#tmp <- s_map(observer,E=1)
#print(tmp)

if ( FALSE ) {
    library(rgl)
    plot3d(lorenz_full$x,lorenz_full$y,lorenz_full$z,col="red")
    snapshot3d("output.png")
}
