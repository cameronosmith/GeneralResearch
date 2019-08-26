
#get our data. reverse so that we are predicting the end = more recent

raw_data <- rev( read.csv("data/nasdaq.csv")$X )
data     <- diff (raw_data)


#get our model prediction
library(rEDM)

len     <- length( data )
pred_len<- 100
pred    <- c(len-pred_len, len)
lib     <- c(1, len-pred_len-3)

simplex_out <- simplex(data, stats_only=FALSE, lib=lib,pred=pred,E=30)
predictions <- simplex_out[[1]]$model_output

real_data <- predictions$obs [(pred[1]+1):(pred[2]+1)]
prediction<- predictions$pred[ pred[1]   : pred[2]]

plot( prediction,  col="blue", type="b" )
lines(real_data,   col="red",  type="b" )
legend(1,30,legend=c("obs", "pred"), col=c("red", "blue"), lty=1:2, cex=.8)

#note. for some reason shifting the obs +1 forward lines up better w real


#now we have our predictions. calculate how good they were for up and down
good = 0
bad = 0

for ( pred_time in pred[1]:(pred[2]) ) {
    true_diff <- raw_data[ pred_time+1 ] - raw_data[ pred_time ]
    pred_diff <- predictions$pred[ pred_time-1 ]
    if ( is.na(pred_diff) || is.na(true_diff)) next
    if ( sign(true_diff)[1] == sign(pred_diff)[1] ){
        good <- good+1
    }
    else{
        bad <- bad+1
    }
}
print(paste("good is ",good))
print(paste("bad is ",bad))



