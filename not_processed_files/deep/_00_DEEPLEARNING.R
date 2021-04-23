library(h2o)
cl <- h2o.init(max_mem_size = "32G", nthreads = 16)

d <- as.h2o(droplevels(iris))
d

h2o.levels(d)

d <- h2o.importFile(path = "http://www.web/file.csv")
d <- h2o.importFile(path = "file.csv")

## NNET package
# Neural Network
digits.train <- read.csv("~/kaggle/train.csv")
dim(digits.train)
head(colnames(digits.train), 4)
tail(colnames(digits.train), 4)
head(digits.train[, 1:4])
## convert to factor
digits.train$label <- factor(digits.train$label, levels = 0:9)
i <- 1:5000
digits.X <- digits.train[i, -1]
digits.y <- digits.train[i, 1]

# digits distibution
barplot(table(digits.y))

set.seed(1234)
digits.m1 <- train(x = digits.X, y = digits.y,
                   method = "nnet",
                   tuneGrid = expand.grid(
                    .size = c(5),
                    .decay = 0.1),
                   trControl = trainControl(method = "none"),
                   MaxNWts = 10000,
                   maxit = 100)

digits.yhat1 <- predict(digits.m1)
barplot(table(digits.yhat1))

caret::confusionMatrix(xtabs(~digits.yhat1 + digits.y))

# 10 hidden neurons
set.seed(1234)
digits.m2 <- train(digits.X, digits.y,
                   method = "nnet",
                   tuneGrid = expand.grid(
                     .size = c(10),
                     .decay = 0.1),
                   trControl = trainControl(method = "none"),
                   MaxNWts = 50000,
                   maxit = 100)

digits.yhat2 <- predict(digits.m2)
barplot(table(digits.yhat2))
caret::confusionMatrix(xtabs(~digits.yhat2 + digits.y))
# Increasing from 5 to 10 hidden neurons improved our in-sample performance from an overall accuracy of 44.3% to 57.6%

# 40 hidden neurons
set.seed(1234)
digits.m3 <- train(digits.X, digits.y,
                   method = "nnet",
                   tuneGrid = expand.grid(
                     .size = c(40),
                     .decay = 0.1),
                   trControl = trainControl(method = "none"),
                   MaxNWts = 50000,
                   maxit = 100)
digits.yhat3 <- predict(digits.m3)
barplot(table(digits.yhat3))
caret::confusionMatrix(xtabs(~digits.yhat3 + digits.y))
Confusion Matrix and Statistics

# Stuttgart Neural Network Simulator (SNNS)
set.seed(1234)
digits.m4 <- mlp(as.matrix(digits.X),
                 decodeClassLabels(digits.y),
                 size = 40,
                 learnFunc = "Rprop",
                 shufflePatterns = FALSE,
                 maxit = 60)

digits.yhat4 <- fitted.values(digits.m4)
digits.yhat4 <- encodeClassLabels(digits.yhat4)
barplot(table(digits.yhat4))
caret::confusionMatrix(xtabs(~ I(digits.yhat4 - 1) + digits.y))
