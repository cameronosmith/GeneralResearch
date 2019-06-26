parameters <- c(s = 10, r = 28, b = 8/3)
state <- c(X = 0, Y = 1, Z = 1)
 
Lorenz <- function(t, state, parameters) {
    with(as.list(c(state, parameters)), {
        dX <- s * (Y - X)
        dY <- X * (r - Z) - Y
        dZ <- X * Y - b * Z
        list(c(dX, dY, dZ))
    })
}
