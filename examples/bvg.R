# Analysis of BMRestore samples from a bivariate Gaussian distribution
# with covariance matrix
#   1.2, 0.4
#   0.4, 0.8
# using a Gaussian regeneration distribution with identity covariance
# matrix
library(mvtnorm)
library(RColorBrewer)

# Target moments
mo1 <- c(0,0)
mo2 <- c(1.2, 0.8)

# True target covariance matrix
targ_cov <- matrix(c(1.2, 0.4, 0.4, 0.8), nrow=2)
targ_prec <- solve(targ_cov)

dtarg <- function(x){
  exp(-0.5 * as.vector(x %*% targ_prec %*% x))
}

# Log density of the target
ldtarg <- function(x){
  -0.5 * as.vector(x %*% targ_prec %*% x)
}

# Partial regeneration rate
kappa_partial <- function(x){
  0.5 * (sum((targ_prec %*% x)^2) - sum(diag(targ_prec)))
}

# Plot the target distribution, regeneration distribution and
# partial regeneration rate
n_grid <- 100
x_seq <- seq(from = -2.5, to = 2.5, length.out = n_grid)
y_seq <- x_seq
targ_dens <- matrix(0, nrow=n_grid, ncol=n_grid)
regen_dens <- matrix(0, nrow=n_grid, ncol=n_grid)
partial_regen_rate <- matrix(0, nrow=n_grid, ncol=n_grid)
for (i in 1:n_grid){
  for (j in 1:n_grid){
    xij <- c(x_seq[i],y_seq[j])
    targ_dens[i,j] <- dtarg(xij)
    regen_dens[i,j] <- dmvnorm(xij)
    partial_regen_rate[i,j] <- kappa_partial(xij)
  }
}

# Plot of the target distribution
contour(x_seq, y_seq, targ_dens, main='Target Distribution',
        xlab='x1', ylab='x2')

# Plot of the target and regeneration distributions
contour(x_seq, y_seq, targ_dens, main='Target and Regeneration Distributions',
        xlab='x1', ylab='x2')
contour(x_seq, y_seq, regen_dens, add=TRUE, col ='green')

# Plot of the target and regeneration distributions,
# as well as the partial regeneration rate
contour(x_seq, y_seq, targ_dens, xlab='x1', ylab='x2',
        main='Target and Regeneration Distributions, Partial Regeneration Rate')
contour(x_seq, y_seq, regen_dens, add=TRUE, col ='green')
contour(x_seq, y_seq, partial_regen_rate, add=TRUE, col = 'red')

# Partial regeneration rate at the origin
kappa_partial(c(0,0))

# Full regeneration rate
logC <- 2.07
kappa <- function(x){
  kappa_partial(x) + exp(logC + mvtnorm::dmvnorm(x, log=TRUE) - ldtarg(x))
}
optim(c(0,0), kappa)

# Plot contours
regen_rate <- matrix(0, nrow=n_grid, ncol=n_grid)
for (i in 1:n_grid){
  for (j in 1:n_grid){
    regen_rate[i,j] <- kappa(c(x_seq[i],y_seq[j]))
  }
}

# Full regeneration rate
contour(x_seq, y_seq, regen_rate, col = 'red', xlab='x1', ylab='x2',
        main='Full Regeneration Rate', levels=c(1, 4, 10, 20, 50, 100))

# Target distribution and full regeneration rate
contour(x_seq, y_seq, targ_dens, main='Target density and regeneration rate',
        xlab='x1', ylab='x2')
contour(x_seq, y_seq, regen_rate, col = 'red',
        levels=c(1, 4, 10, 20, 50, 100), add =TRUE)

#####################
# Short, detailed run
#####################

x <- read.table("bmrstr_x1.txt",
                quote="\"", comment.char="")
ts <- read.table("bmrstr_ts1.txt",
                quote="\"", comment.char="")
ts <- ts$V1
tours <- read.table("bmrstr_tours1.txt",
                quote="\"", comment.char="")
tours <- tours$V1

# 2d trace plots
ntours <- 10

plot(x[tours <= ntours,], type = 'n', xlab='X1', ylab='X2')
contour(x_seq, y_seq, targ_dens, col='gray', add=TRUE)
for (tour in 0:10){
  lines(x[tours==tour,])
}

# Same 2d plot, with colours
qual_col_pals <- brewer.pal.info[brewer.pal.info$category == 'qual',]
col_vector <- unlist(mapply(brewer.pal, qual_col_pals$maxcolors,
                            rownames(qual_col_pals)))
col_vector <- sample(col_vector)

plot(x[tours <= ntours,], type = 'n', xlab='X1', ylab='X2')
contour(x_seq, y_seq, targ_dens, col='gray', add=TRUE, nlevels = 5)
for (tour in 0:20){
  lines(x[tours==tour,], col=col_vector[tour], cex=2)
}

# Trace plots
for (i in 1:2){
  plot(ts, x[,i], type='n', xlab='t', ylab=paste0('X',i))
  for (tour in 0:(ntours-1)){
    lines(ts[tours==tour], x[tours==tour,i])
  }
}

# Trace plots: zoomed in
for (i in 1:2){
  plot(ts, x[,i], type='n', xlab='t', ylab=paste0('X',i),
       xlim=c(0,5))
  for (tour in 0:(ntours-1)){
    lines(ts[tours==tour], x[tours==tour,i])
  }
}

##############################
# Load samples from longer run
##############################
x2 <- read.table("bmrstr_x2.txt", quote="\"", comment.char="")
dim(x2)

# Marginals
for (i in 1:2){
  plot(density(x2[,i]), col='blue',
       main=paste0('Density Estimate of X', i))
  curve(dnorm(x, sd=sqrt(targ_cov[i,i])), add=TRUE)
}

# Analyse Monte Carlo samples
mo1_est <- colMeans(x2)
mo2_est <- colMeans(x2^2)

# RMSE
sqrt(mean((mo1 - mo1_est)^2))
sqrt(mean((mo2 - mo2_est)^2))
