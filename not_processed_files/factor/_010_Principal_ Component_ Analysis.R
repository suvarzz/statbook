### Principal Component Analysis

# Source: https://www.datacamp.com/community/tutorials/pca-analysis-r

# PCA allows to visualize datasets with many variables
# PCA is a linear transformation which simplify many variables into a smaller number of "Principal Components".

### Example 1 "mtcars"
mtcars.pca <- prcomp(mtcars[,c(1:7,10,11)], center = TRUE,scale. = TRUE)

summary(mtcars.pca)
str(mtcars.pca)

library(devtools)
install_github("vqv/ggbiplot")
library(ggbiplot)

ggbiplot(mtcars.pca)
ggbiplot(mtcars.pca, labels=rownames(mtcars))

ggbiplot(mtcars.pca,ellipse=TRUE,choices=c(3,4),   labels=rownames(mtcars), groups=mtcars.country)
ggbiplot(mtcars.pca,ellipse=TRUE,circle=TRUE, labels=rownames(mtcars), groups=mtcars.country)
ggbiplot(mtcars.pca,ellipse=TRUE,obs.scale = 1, var.scale = 1,  labels=rownames(mtcars), groups=mtcars.country)
ggbiplot(mtcars.pca,ellipse=TRUE,obs.scale = 1, var.scale = 1,var.axes=FALSE,   labels=rownames(mtcars), groups=mtcars.country)

### Example 2 "USArrests"

?USArrests               # Violent Crime Rates by US State
dim(USArrests)
dimnames(USArrests)

apply(USArrests,2,mean)   # finding mean of all 
# big variance between samples, need to standartize
pca.out<-prcomp(USArrests,scale=TRUE)
pca.out
summary(pca.out)
names(pca.out)
biplot(pca.out,scale = 0, cex=0.65)
