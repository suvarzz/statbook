# Estimate model precision for one continious dependent variable (Y)
# Source: Course 'Math modeling' practical work, State University of Management, Moscow
# link: https://sites.google.com/a/kiber-guu.ru/r-practice/home

### GENERATE DATA
# Generate data for given function
y.func <- function(x) {4 - 2e-02*x + 5.5e-03*x^2 - 4.9e-05*x^3}

### Parameters for data generation
my.seed <- 123         # set seed
n.all <- 100           # number of samples 
train.percent <- 0.85  # percent of train data
res.sd <- 1            # standard deviation of noise 

x.min <- 5             # minimal value
x.max <- 104           # maximal value

# generate random x values
set.seed(my.seed)
x <- runif(x.min, x.max, n = n.all)

# set normally distributed noise for each value
set.seed(my.seed)
res <- rnorm(mean = 0, sd = res.sd, n = n.all)

# calculate y values using funcion and add noise
y <- y.func(x) + res

# vector for subseting train data (contains numbers of selected elements in x)
set.seed(my.seed)
inTrain <- sample(seq_along(x), size = train.percent*n.all)
inTrain

# Subset vectors with train and test data subsets
# Train data
x.train <- x[inTrain]
y.train <- y[inTrain]

# Test data
x.test <- x[-inTrain]
y.test <- y[-inTrain]

# x and y to display given function 'y.func'
x.line <- seq(x.min, x.max, length = n.all)
y.line <- y.func(x.line)

x.line
y.line

### PLOT PRIMARY DATA
#------------------------------------------------------------------------------
# Plot parameters
x.lim <- c(x.min, x.max)
y.lim <- c(min(y), max(y))

# Plot generated train data
plot(x.train, y.train, 
     col = grey(0.2), bg = grey(0.2), pch = 21,
     xlab = 'X', ylab = 'Y', 
     xlim = x.lim, ylim = y.lim, 
     cex = 1.2, cex.lab = 1.2, cex.axis = 1.2)

# Header
mtext('Initial train data and real function', side = 3)

# test values
points(x.test, y.test, col = 'red', bg = 'red', pch = 21)

# true function
lines(x.line, y.line, lwd = 2, lty = 2)

# legend
legend('bottomright', legend = c('train data', 'test data', 'f(X)=4 - 2e-02*x + 5.5e-03*x^2 - 4.9e-05*x^3'),
       pch = c(16, 16, NA), 
       col = c(grey(0.2), 'red', 'black'),  
       lty = c(0, 0, 2), lwd = c(1, 1, 2), cex = 1.2)

#------------------------------------------------------------------------------

### MODEL
# For modeling spline with degree of freedoms from 2 (straight line) to 50 (1/2 of N)
# To demonstrate spline with df = 6
mod <- smooth.spline(x = x.train, y = y.train, df = 6)

# Model data for error estimation
y.model.train <- predict(mod, data.frame(x = x.train))$y[, 1]
y.model.test <- predict(mod, data.frame(x = x.test))$y[, 1]

# Calculate mean squared errors for test and model data
MSE <- c(sum((y.train - y.model.train)^2) / length(x.train),
         sum((y.test - y.model.test)^2) / length(x.test))
names(MSE) <- c('train', 'test')
round(MSE, 2)

# Now let's make splines for all degrees of freedoms (2-50)
# Max degrees of fredom for spline model
max.df <- 50

tbl <- data.frame(df = 2:max.df)   # Create result table for recording calculated MSE 
tbl$MSE.train <- 0                 # Column for train data
tbl$MSE.test <- 0                  # Column for test data

# For all degrees of freedoms
for (i in 2:max.df) {
    # build model
    mod <- smooth.spline(x = x.train, y = y.train, df = i)
    
    # model values from calculaton of errors
    y.model.train <- predict(mod, data.frame(x = x.train))$y[, 1]
    y.model.test <- predict(mod, data.frame(x = x.test))$y[, 1]
    
    # Calculate MSE for predicted and train data sets
    MSE <- c(sum((y.train - y.model.train)^2) / length(x.train),
             sum((y.test - y.model.test)^2) / length(x.test))
    
    # record errors into the table
    tbl[tbl$df == i, c('MSE.train', 'MSE.test')] <- MSE
}

head(tbl)

### PLOT - Dependency of MSEs from model flexibility (degree of freedoms)
#------------------------------------------------------------------------------
plot(x = tbl$df, y = tbl$MSE.test, 
     type = 'l', col = 'red', lwd = 2,
     xlab = 'Degree of freedoms', ylab = 'MSE',
     ylim = c(min(tbl$MSE.train, tbl$MSE.test), 
              max(tbl$MSE.train, tbl$MSE.test)),
     cex = 1.2, cex.lab = 1.2, cex.axis = 1.2)

# Header
mtext('MSE dependency from degree of freedoms', side = 3)

points(x = tbl$df, y = tbl$MSE.test,
       pch = 21, col = 'red', bg = 'red')
lines(x = tbl$df, y = tbl$MSE.train, col = grey(0.3), lwd = 2)
# unrecoverable error
abline(h = res.sd, lty = 2, col = grey(0.4), lwd = 2)

# legend
legend('topright', legend = c('train', 'test'),
       pch = c(NA, 16), 
       col = c(grey(0.2), 'red'),  
       lty = c(1, 1), lwd = c(2, 2), cex = 1.2)

#------------------------------------------------------------------------------

# Detect degree of freedom with minimal error for test data
min.MSE.test <- min(tbl$MSE.test)
df.min.MSE.test <- tbl[tbl$MSE.test == min.MSE.test, 'df']
df.min.MSE.test

# Compromize for flexibility and pricision of model
df.my.MSE.test <- df.min.MSE.test

my.MSE.test <- tbl[tbl$df == df.my.MSE.test, 'MSE.test']
my.MSE.test

# Display optimal df
abline(v = df.my.MSE.test, lty = 2, lwd = 2)
points(x = df.my.MSE.test, y = my.MSE.test, pch = 15, col = 'blue')
mtext(df.my.MSE.test, side = 1, line = -1, at = df.my.MSE.test, col = 'blue', cex = 1.2)

### PLOT OPTIMAL MODEL
mod.MSE.test <- smooth.spline(x = x.train, y = y.train, df = df.my.MSE.test)

# for smoothed curves of model
x.model.plot <- seq(x.min, x.max, length = 250)
y.model.plot <- predict(mod.MSE.test, data.frame(x = x.model.plot))$y[, 1]

x.lim <- c(x.min, x.max)
y.lim <- c(min(y), max(y))

# Plot train data with noise
#------------------------------------------------------------------------------
plot(x.train, y.train, 
     col = grey(0.2), bg = grey(0.2), pch = 21,
     xlab = 'X', ylab = 'Y', 
     xlim = x.lim, ylim = y.lim, 
     cex = 1.2, cex.lab = 1.2, cex.axis = 1.2)

# Header
mtext('Given data and optimal model', side = 3)

# Test data
points(x.test, y.test, col = 'red', bg = 'red', pch = 21)

# True function
lines(x.line, y.line, lwd = 2, lty = 2)

# Model
lines(x.model.plot, y.model.plot, lwd = 2, col = 'blue')

# legend
legend('topleft', legend = c('train', 'test', 'f(X)', 'model'),
       pch = c(16, 16, NA, NA), 
       col = c(grey(0.2), 'red', 'black', 'blue'),  
       lty = c(0, 0, 2, 1), lwd = c(1, 1, 2, 2), cex = 1.2)
#------------------------------------------------------------------------------
