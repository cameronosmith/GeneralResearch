source("stock_machine.R")

raw_data  <- rev( read.csv("data/nasdaq.csv")$X )
diff_data <- diff(raw_data)

days_back <- 50

total_pred <- (0)

#see how we'd do for a number of days back
for ( num_days_back in 1:days_back ) {
    data        <- raw_data[ 1:(length(raw_data)-num_days_back) ]
    #the true results of whether pince went up or down 
    true_diff   <- data[length(data)] - data[length(data)-1]
    #get our prediction
    pred        <- run_stock_machine( diff( data ) )
    total_pred[length(total_pred)+1] <- pred
}

plot( total_pred,type="o",col="red" )


#now we have our predictions. calculate how good they were for up and down
good = 0
bad = 0

if ( FALSE)  {
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
}
