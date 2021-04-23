# Random Forest

# Methods of improving classifiers:
# 1. Stacking
# 2. Bagging (bootstrap aggregation)
# 3. Boosting

# Random Forest = Bootstrap aggregation
# N samples, M characteristics

install.packages("randomForest")
library(randomForest)


# Read data
zzz <- read.table("~/DataAnalysis/R_data_analysis/DATA/Wine.txt", header=T, sep="", dec=".")
zzz
names(zzz) <- c("Alcohol", "Malic_acid", "Ash", "Alcalinity_of_ash", "Magnesium",
                "Total_phenols", "Flavanoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity",
                "Hue", "OD280_OD315_of_diluted_wines", "Proline", "Wine_type")
zzz
# Predictors
x <- zzz[,1:13]

# Responce
y <- zzz[,14]
y.1 <- as.factor(y)
table(y)

set.seed(123)
ntree.1 <- 500     # number of trees in the forest
nodesize.1 <- 1    # minimum size of terminal nodes
keep.forest.1 <- TRUE  # results include forest

rf.res <- randomForest(x, y=y.1,
                       ntree=ntree.1,
                       mtry=floor(sqrt(ncol(zzz))),
                       replace=FALSE,
                       nodesize = nodesize.1,
                       importance=TRUE,
                       localImp=FALSE,
                       proximity=FALSE,
                       norm.votes=TRUE,
                       do.trace=ntree.1/10,
                       keep.forest=keep.forest.1,
                       corr.bias=FALSE,
                       keep.inbag=FALSE)

# Detect number of trees
zzz.predict <- predict(rf.res, newdata = x)
# Предпочтительнее вариант
# zzz.predict <- predict(rf.res, newdata = x, type = "prob")
# Оценим качество результата
table(y, zzz.predict)
# zzz.predict
# y 0 1 2
# 0 59 0 0
# 1 0 48 0
# 2 0 0 71
# import.wine <- importance(rf.res, type=NULL, class=1, scale=TRUE)
# import.wine.2 <- as.data.frame(import.wine)
varImpPlot(rf.res, sort=F)
varUsed(rf.res, by.tree=FALSE, count=TRUE)
