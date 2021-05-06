data <- iris[,-5]
fit <- randomForest(iris[,5] ~ ., data=data, ntree=200)
fit


require(randomForest)
require(MASS)
attach(Boston)

head(Boston)
?Boston
str(Boston)

library(mlbench)
data(PimaIndiansDiabetes)
str(PimaIndiansDiabetes)
df <- PimaIndiansDiabetes
inTrain = sample(nrow(df), nrow(df)*0.8)

# split data
train = df[inTrain, ]
test = df[-inTrain, ]


fit <- randomForest(diabetes ~ . , data=train, ntree=100, proximity=TRUE)
fit
table(predict(fit), train$diabetes)
plot(fit)
importance(fit)
varImpPlot(fit)
pred <- predict(fit, newdata=test)
table(pred, test$diabetes)

plot(margin(fit, test$Species))

# Tune Random Forest
tune.rf <- tuneRF(iris[,-5], iris[,5], stepFactor=.5)
print(tune.rf)
