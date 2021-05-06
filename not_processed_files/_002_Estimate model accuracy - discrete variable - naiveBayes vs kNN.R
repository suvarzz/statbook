# Estimate model precision for one descrete dependent variable (Y)
# Source: Course 'Math modeling' practical work, State University of Management, Moscow
# link: https://sites.google.com/a/kiber-guu.ru/r-practice/home

library('mlbench')
library('class')
library('car')
library('class')
library('e1071')
library('MASS')

# discrete function
rules <- function(x1, x2){ ifelse((x1 > 20 & x2 < 50) | (x1 < 18 & x2 > 52), 1, 0) }

# parameters
my.seed <- 123
n <- 100
train.percent <- 0.85

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

### PLOT TEST DATA
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

### 1. Build a model using Naive Bayess classifier

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

Acc <- sum(diag(tbl)) / sum(tbl)
Acc

cat('Как можно видеть на графике, ту часть жёлтого класса, которая расположена в левой верхней области пространства координат, модель классифицирует неверно. Таким образом, байесовская решающая граница не моделирует разрыв жёлтого класса синим. Это происходит потому, что в непрерывном случае наивный байесовский метод исходит из допущения о линейной разделимости двух классов и нормальности распределения объясняющих переменных в них. Однако в этом примере это допущение не выполняется.
Сделаем прогноз классов Y на тестовую выборку и оценим точность модели. Как можно убедиться, точность на тестовой оказывается ниже, чем на обучающей выборке. Учитывая, как ведёт себя классификатор на обучающей выборке, такой модели доверять не стоит.')

# прогноз на тестовую выборку
y.nb.test <- ifelse(predict(nb, df.test.1, type = "raw")[, 2] > 0.5, 1, 0)
# матрица неточностей на тестовой выборке
tbl <- table(y.test, y.nb.test)
tbl

# точность, или верность (Accuracy)
Acc <- sum(diag(tbl)) / sum(tbl)
Acc

### 2. Build kNN model for k=3
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

# As we can see, kNN method is more accurate than Naive Bayes for this case.

