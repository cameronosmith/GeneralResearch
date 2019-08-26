library(rEDM)

run_stock_machine <- function ( data ) {

    #the data ranges to run on
    E=30
    len     <- length( data )
    pred    <- c(len-E, len+1)
    lib     <- c(1, len-1-E)

    #get our model prediction
    simplex_out <- simplex(data, stats_only=FALSE, lib=lib,pred=pred,E=E)
    predictions <- simplex_out[[1]]$model_output

    prediction<- predictions$pred[ len ]
    return (prediction)
}

