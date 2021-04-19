# Estimate model precision for one descrete dependent variable (Y)
# Source: Course 'Math modeling' practical work, State University of Management, Moscow
# link: https://sites.google.com/a/kiber-guu.ru/r-practice/home

# in this example descrete data can be split by line

library('mlbench')
library('class')
library('car')
library('class')
library('e1071')
library('MASS')

# parameters
my.seed <- 123
n <- 100               # size of sample
train.percent <- 0.85

# x's are normaly distributed, two dimensional variables
set.seed(my.seed)
class.0 <- mvrnorm(45, mu = c(25, 49), Sigma = matrix(c(6, 0, 0, 3.4), 2, 2, byrow = T))

set.seed(my.seed + 1)
class.1 <- mvrnorm(65, mu = c(15, 51), Sigma = matrix(c(2, 0, 0, 6), 2, 2, byrow = T))

# Combine class.0 and class.1 into vectors
x1 <- c(class.0[, 1], class.1[, 1])
x2 <- c(class.0[, 2], class.1[, 2])
# фактические классы Y
y <- c(rep(0, nrow(class.0)), rep(1, nrow(class.1)))
# классы для наблюдений сетки
rules <- function(x1, x2){ ifelse(x2 < 1.6*x1 + 19, 0, 1) }


# Subset test data
set.seed(my.seed)
inTrain <- sample(seq_along(x1), train.percent*n)
x1.train <- x1[inTrain]
x2.train <- x2[inTrain]
x1.test <- x1[-inTrain]
x2.test <- x2[-inTrain]
# используем истинные правила, чтобы присвоить фактические классы
y.train <- y[inTrain]
y.test <- y[-inTrain]
# фрейм с обучающей выборкой
df.train.1 <- data.frame(x1 = x1.train, x2 = x2.train, y = y.train)
# фрейм с тестовой выборкой
df.test.1 <- data.frame(x1 = x1.test, x2 = x2.test)

# Generate data set
set.seed(my.seed)
x1 <- rnorm(20, 3.7, n = n)

set.seed(my.seed + 1)
x2 <- rnorm(50, 3.3, n = n)

# Vector to subset train and test data
set.seed(my.seed)
inTrain <- sample(seq_along(x1), train.percent*n)

x1.train <- x1[inTrain]
x2.train <- x2[inTrain]

x1.test <- x1[-inTrain]
x2.test <- x2[-inTrain]

# Create descrete parameter according to given function
y.train <- rules(x1.train, x2.train)
y.test <- rules(x1.test, x2.test)
# Table with train data
df.train.1 <- data.frame(x1 = x1.train, x2 = x2.train, y = y.train)
df.train.1
# Table with test data
df.test.1 <- data.frame(x1 = x1.test, x2 = x2.test)
df.test.1

### PLOT
# Рисуем обучающую выборку графике ---------------------------------------------
# для сетки (истинных областей классов): целочисленные значения x1, x2
x1.grid <- rep(seq(floor(min(x1)), ceiling(max(x1)), by = 1),
               ceiling(max(x2)) - floor(min(x2)) + 1)
x2.grid <- rep(seq(floor(min(x2)), ceiling(max(x2)), by = 1),
               each = ceiling(max(x1)) - floor(min(x1)) + 1)
# классы для наблюдений сетки
y.grid <- rules(x1.grid, x2.grid)
# фрейм для сетки
df.grid.1 <- data.frame(x1 = x1.grid, x2 = x2.grid, y = y.grid)
# цвета для графиков
cls <- c('blue', 'orange')
cls.t <- c(rgb(0, 0, 1, alpha = 0.5), rgb(1,0.5,0, alpha = 0.5))
# график истинных классов
plot(df.grid.1$x1, df.grid.1$x2, 
     pch = '·', col = cls[df.grid.1[, 'y'] + 1],
     xlab = 'X1', ylab = 'Y1',
     main = 'Обучающая выборка, факт')
# точки фактических наблюдений
points(df.train.1$x1, df.train.1$x2,
       pch = 21, bg = cls.t[df.train.1[, 'y'] + 1], 
       col = cls.t[df.train.1[, 'y'] + 1])

### 1. Build model using naive Bayes method
# Байесовский классификатор ----------------------------------------------------
#  наивный байес: непрерывные объясняющие переменные
# строим модель
nb <- naiveBayes(y ~ ., data = df.train.1)
# получаем модельные значения на обучающей выборке как классы
y.nb.train <- ifelse(predict(nb, df.train.1[, -3], 
                             type = "raw")[, 2] > 0.5, 1, 0)
# график истинных классов
plot(df.grid.1$x1, df.grid.1$x2, 
     pch = '·',  col = cls[df.grid.1[, 'y'] + 1], 
     xlab = 'X1', ylab = 'Y1',
     main = 'Обучающая выборка, модель naiveBayes')
# точки наблюдений, предсказанных по модели
points(df.train.1$x1, df.train.1$x2, 
       pch = 21, bg = cls.t[y.nb.train + 1], 
       col = cls.t[y.nb.train + 1])

# матрица неточностей на обучающей выборке
tbl <- table(y.train, y.nb.train)
tbl

# точность, или верность (Accuracy)
Acc <- sum(diag(tbl)) / sum(tbl)
Acc

# As we can see Naive Bayes can ideally split classes in this case
# Lets forcast test samples
# прогноз на тестовую выборку
y.nb.test <- ifelse(predict(nb, df.test.1, type = "raw")[, 2] > 0.5, 1, 0)
# матрица неточностей на тестовой выборке
tbl <- table(y.test, y.nb.test)
tbl
# точность, или верность (Accuracy)
Acc <- sum(diag(tbl)) / sum(tbl)
Acc

### 2. Build kNN model for k=3
# Метод kNN --------------------------------------------------------------------
#  k = 3
# строим модель и делаем прогноз
y.knn.train <- knn(train = scale(df.train.1[, -3]), 
                   test = scale(df.train.1[, -3]),
                   cl = df.train.1$y, k = 3)
# график истинных классов
plot(df.grid.1$x1, df.grid.1$x2, 
     pch = '·', col = cls[df.grid.1[, 'y'] + 1],
     xlab = 'X1', ylab = 'Y1',
     main = 'Обучающая выборка, модель kNN')
# точки наблюдений, предсказанных по модели
points(df.train.1$x1, df.train.1$x2, 
       pch = 21, bg = cls.t[as.numeric(y.knn.train)], 
       col = cls.t[as.numeric(y.knn.train)])
# матрица неточностей на обучающей выборке
tbl <- table(y.train, y.knn.train)
tbl
# точность (Accuracy)
Acc <- sum(diag(tbl)) / sum(tbl)
Acc

# прогноз на тестовую выборку
y.knn.test <- knn(train = scale(df.train.1[, -3]), 
                  test = scale(df.test.1[, -3]),
                  cl = df.train.1$y, k = 3)
# матрица неточностей на тестовой выборке
tbl <- table(y.test, y.knn.test)
tbl

# точность (Accuracy)
Acc <- sum(diag(tbl)) / sum(tbl)
Acc
