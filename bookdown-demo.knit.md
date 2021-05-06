--- 
title: "R statistics"
author: "Mark Goldberg"
date: "2021-05-06"
site: bookdown::bookdown_site
output: bookdown::gitbook
documentclass: book
bibliography: [book.bib, packages.bib]
biblio-style: apalike
link-citations: yes
github-repo: rstudio/bookdown-demo
description: "This is a minimal example of using the bookdown package to write a book. The output format for this example is bookdown::gitbook."
---

# Introduction

This is the collection of statistical methods in R.  

<!--chapter:end:index.Rmd-->

# Statistics R functions reference

## Get data

## Data inspection

```r
head(df)
typeof(df)  # data type
dim(df)     # dimention
nrow(df)    # number of rows
ncol(df)    # number of columns
str(df)     # data structure
summary(df) # data summary
names(df)   # names of columns
colnames(df) # also column names

table(df)	# frequency of categorical data
```

## Plots

```r
plot(x ~ y)
barplot(df)
boxplot(v)
hist(x)
pie(groupsize, labels, col, ...)
```

## Analysis of the distribution

```r
# mode
getmode <- function(v) {
   uniqv <- unique(v)
   uniqv[which.max(tabulate(match(v, uniqv)))]
}

getmode(v)        # mode
mean(v)           # mean
mean(v, trim=0.1) # trimmed mean
median(v)         # median

min(v)
max(v)
range(v)          # c(mun(v), max(v))
max(v)-min(v)   # range

sort(v)
rank(v)
sample(v)        # subsample
sample(v, size)

var(v)            # variance
sd(v)             # standard deviation
cor(v)            # correlation
cov(v)            # covariation
scale(v)          # z-scores

quantile(v)
IQR(v)            # interquantile range: IQR = Q3 – Q1

qqnorm(v)         # normal probability plot
qqline(v)         # adds a line to a normal probability plot passing through 1Q and 3Q
```

## Distributions

**d** - probability density  
**p** - cumulative distribution  
**q** - quantile function (inverse cumulative distribution)  
**r** - random variables from distribution  

```r
# Normal distribution
dnorm(x, mean, sd)     # Probability Density Function (PDF)
pnorm(q, mean, sd)     # Cumulative Distribution Function (CDF)
qnorm(p, mean, sd)     # quantile function - inverse of pnorm
rnorm(n, mean, sd)     # random numbers from normal distribution

# Binomial
dbinom(x)     # Probability Density Function (PDF)
pbinom(q)     # Cumulative Distribution Function (CDF)
qbinom(p)     # quantile function - inverse of pnorm
rbinom(n)     # random numbers from normal distribution

# Poisson
dpois()
ppois()
qpois()
rpois()

# Exponential
dexp()
pexp()
qexp()
rexp()

# Chi-squared distribution

dchisq(v, df)     # density of the interval
pchisq()          # distribution of the interval (AUC)
qchisq()          # quantiles
rchisq()          # random deviates
```

## t-Test

```r
t.test(x, mu = 0, alternative = c("two.sided", "less", "greater"), 
       paired = FALSE, var.equal = FALSE, conf.level = 0.95)
t.test(v, mu)     # one-sample t-test, mu - null hypothesized value
t.test(v1, v2)    # two-sample t-test
t.test(v1, v2, var.equal=T)
t.test(var1, var2, paired=T)
wilcox.test(v1, v2, paired=T)
```

## ANOVA

```r
# One way ANOVA
oneway.test(x ~ f)
aov(x ~ f)

anova(m1, m2)      # compair two models
```


<!--chapter:end:02_stat_fun_ref.Rmd-->

# Basic Statistics

## Definitions
**population** - all existing samples  
**sample** - subset of statistical population  
**simple random sample** - random subset  
**stratified sample** - fist clustering, than random sample from  
**cluster sample** - random choosing from several existing clusters  
**variables** - discret, continuous, ordinal (ранговая)  

## Probability
A standard French-suited deck of playing cards contains 52 cards; 13 each of hearts (♥), spades (♠), clubs (♦), and diamonds (♣). Assuming that you have a well-shuffled deck in front of you, the probability of drawing any given card is 1/52 ≈ 1.92%.  
Calculate the probability of drawing any of the four aces! That is, calculate the probability of drawing 🂡 or 🂱 or 🃁 or 🃑 using the sum rule and assign it to prob_to_draw_ace.  


```r
# Calculate the probability of drawing any of the four aces
prob_to_draw_ace <- 1/52 + 1/52 + 1/52 + 1/52
```
Cards and the product rule
Again, assuming that you have a well-shuffled deck in front of you, the probability of drawing any given card is 1/52 ≈ 1.92% . The probability of drawing any of the four aces is 1/52 + 1/52 + 1/52 + 1/52 = 4/52. Once an ace has been drawn, the probability of picking any of the remaining three is 3/51. If another ace is drawn the probability of picking any of the remaining two is 2/50, and so on.  
Use the product rule to calculate the probability of picking the four aces in a row from the top of a well-shuffled deck and assign it to prob_to_draw_four_aces.  

```r
# Calculate the probability of picking four aces in a row
prob_to_draw_four_aces <- 4/52 * 3/51 * 2/50 * 1/49
```

## Analysis of sample distribution
### Histogram

```r
# sample of random integers
v <- round(rnorm(n=50, sd=5, mean=100))

par(mfrow=c(1,2))
stripchart(v, method = "stack", pch=19, cex=2, offset=.5, at=.15,
           main = "Dotplot of random value", xlab = "Random value")

hist(v)

# add density
x <- density(v)$x
y <- (10/max(density(v)$y))*density(v)$y  # scale y to plot with histogram
lines(x, y, col="red", lwd=2)
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-9-1.png" width="672" />

### Outliers
Outliers are rare values that appear far away from the majority of the data. 
Outliers can bias the results and potentially lead to incorrect conclusions if not handled properly. 
One method for dealing with outliers is to simply remove them. 
However, removing data points can introduce other types of bias into the results, and potentially result in losing critical information. 
If outliers seem to have a lot of influence on the results, a nonparametric test such as the **Wilcoxon Signed Rank Test** may be appropriate to use instead. 
Outliers can be identified visually using a boxplot.  

### Normality
It is possible to use histogram to estimate normality of the distribution.  


```r
# QQ-plot - fit normal distibution
qqnorm(v); qqline(v)
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-10-1.png" width="672" />

```r
var(v)     # variance: sd = sqrt(var)
```

```
## [1] 23.94653
```

```r
sd(v)      # standard deviation
```

```
## [1] 4.893519
```

```r
sd(v)/sqrt(length(v))  # standard error sd/sqrt(n)
```

```
## [1] 0.6920481
```

```r
# Z-score (standartization)
# transform distribution to mean=0, variance=1
# z = (x - mean(n))/sd
vs <- scale(v)[,1]
vs
```

```
##  [1] -0.64983907 -1.26289480 -0.44548716  0.57627238 -1.67159862  1.39368002  0.37192047  0.16756857 -0.24113525  1.59803193 -0.85419098  0.37192047 -0.03678334 -0.44548716
## [15]  1.59803193 -0.44548716  0.57627238 -1.05854289 -1.46724671  1.59803193  0.16756857  0.37192047 -0.03678334 -2.48900625 -0.03678334 -0.03678334  1.39368002 -0.85419098
## [29] -1.05854289  0.78062429 -0.03678334  1.18932811  0.98497620 -1.26289480  0.16756857  0.37192047  1.59803193  0.16756857  1.80238384 -0.64983907 -0.03678334 -1.67159862
## [43] -1.26289480 -0.44548716  0.16756857  1.18932811 -0.03678334  0.16756857 -1.05854289  0.78062429
```

```r
par(mfrow=c(1,2))
hist(v, breaks=10)
hist(vs, breaks=10)
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-10-2.png" width="672" />

## Confidence interval

```r
# sample of random integers
x <- round(rnorm(n=50, sd=5, mean=100))

# Confidence interval for normal distribution with p=0.95
m <- mean(x)
s <- sd(x)
n <- length(x)
error <- qnorm(0.95)*s/sqrt(n)
confidence <- c(m-error, m+error)
confidence
```

```
## [1]  98.58989 100.97011
```

```r
# Confidence interval for t-distribution with p=0.95
a <- 5
s <- 2
n <- 20
error <- qt(0.975,df=n-1)*s/sqrt(n)
# confidence interval
c(left=a-error, right=a+error)
```

```
##     left    right 
## 4.063971 5.936029
```

<!--chapter:end:03_basic_statistics.Rmd-->

# Primary data analysis - Case studies


Abbakumov, 2016, lectures  


```r
# GET DATA
df <- read.table("./DATA/Swiss_Bank_Notes.csv", header=T, sep=" ", dec=",")
head(df)

# Data explanation: parameters of Swiss Banknotes
# Size of data: 200 (100 are real, 100 are false)
# Length - length
# H_l - height left
# H_r - height right
# dist_l - border left
# dist_up - border up
# Diag - diagonal
# ? Find false banknotes

# 1. Let's add a column with 100 filled 0 and 100 filled with 1.
origin <- 0
df <- data.frame(df, origin)
df$origin[1:100] <-1
# Set origin as factor - binary data (0,1)
df$origin <- as.factor(df$origin)
is.factor(df$origin)
```


```r
# 2. HISTORGRAM
par(mfrow=c(length(colnames(df))/2,2))
for (i in 1:length(colnames(df))) { hist(df[,i], main = paste(colnames(df)[i])) }

# Histogram for Diagonals
par(mfrow=c(1,1))
hist(df$Diag, breaks=18, probability=TRUE)

# Barplot
barplot(VADeaths, beside=TRUE, legend=TRUE, ylim=c(0, 100),
        ylab="Deaths per 1000", main="Death rates in Virginia")

# Pieplot
groupsize <- c(18, 30, 32, 10, 10)
labels <- c("A", "B", "C", "D", "F")
pie(groupsize, labels, col=c("red", "white", "grey", "black", "blue"))

# All pairs of data scatter plot
plot(df)

# Length ~ Dial
plot(df$Length, df$Diag)
# true notes
points(df$Length[df$origin==1],
       df$Length[df$origin==0], pch=3, col="green")
# false notes
points(df$Length[df$origin==1],
       df$Length[df$origin==0], pch=1, col="red")

# If factors are given, plot makes boxplot
plot(df$origin, df$Diag)
title("Swiss Bank Notes")

# GET DATA - TOWNS
town <- read.table("DATA/town_1959_2.csv", header=T, sep="\t", dec=".")
town
summary(town)
# Median is more stable to outliers
summary(town[,3])
# lets remove 2 first towns from the data
summary(town[3:1004,3])
hist(town[,3])

# log scale allows us to see outliers better
hist(log(town[,3]), breaks=50)

# Truncated mean is better than mean
mean(town[,3], trim=0.05)

# GET DATA - BASKETBALL
bb <- read.table("DATA/basketball.csv", header=F, sep=";", dec=".")
bb
# NBA Player characteristics:
# percent of positives vs:
# SF - light forvard
# PF - heavy forvard
# C - center
# G - defender

summary(bb[,1])

par(mfrow=c(1,1))
plot(bb[,1]~bb[,2])

par(mfrow=c(2,2))
for (i in 1:4) { hist(bb[,1], breaks=5*i, main=paste("Breaks", 5*i), ylab="") }

for (i in unique(bb[,2])) {
     hist(bb[bb[,2]==i ,1],  breaks=6, 
     xlim=c(min(bb[,1])-5, max(bb[,1]+5)), col="white", main=i, ylab="")
}
# Conclusion: for several groups of data boxplots may be more informative than histograms
```

<!--chapter:end:04_primary_analysis_cases.Rmd-->

# Primary data analysis

## Handling missing data

* **Ignore**: Discard samples with missing values.  
* **Impute**: 'Fill in' the missing values with other values.  
* **Accept**: Apply methods that are unaffected by the missing data.  


```r
library(naniar)
mammographic <- read.csv('./DATA/mammographic.data')
any_na(mammographic)
# Replace ? with NAs: bands
mammographic <- replace_with_na_all(mammographic, ~.x == '?')
any_na(mammographic)
miss_var_summary(mammographic)
```

**Vizualysing missing data**  

```r
library(ggpubr)

a <- vis_miss(mammographic)
# comulative
b <- vis_miss(mammographic, cluster=TRUE)
c <- gg_miss_case(mammographic)

ggarrange(a, b, c + rremove("x.text"), 
          labels = c("frame view", "cumulative", "missing"),
          ncol = 3, nrow = 1)
```

**Missing data types**  
- **MCAR**: Missing Completely At Random  
- **MAR**: Missing At Random  
- **MNAR**: Missing Not At Random  

| **Type** | **Imputation**    | **Deletion**          | **Visual cues**                                                              |
|----------|-------------------|-----------------------|------------------------------------------------------------------------------|
| **MCAR** | Recommended       | Will not lead to bias | Random or noisy patterns in missingness clusters                             |
| **MAR**  | Recommended       | May lead to bias      | Well-defined missingness clusters when arrangin for a particular variable(s) |
| **MNAR** | Will lead to bias | Will lead to bias     | Neither visual pattern above holds                                           |

It can be difficult to ascertain the missingness type using visual inspection!  

**Internal evaluation**  
Compair distributions with/without imputed values:  

* Mean  
* Variance  
* Scale  

**Exterlan evaluation**
Build ML models with/without imputated values and evaluate impact of imputation method on ML model performance:  

* Classification  
* Regression  
* Clustering  
* etc.  

Ideally imputation should not bring big differences.  

**Mean and linear imputations**  

```r
library(naniar)
library(simputation)

# Impute with the mean
imp_mean <- bands %>%
    bind_shadow(only_miss = TRUE) %>% 
    add_label_shadow() %>% 
    impute_mean_all()

# Impute with lm
imp_lm <- bands %>%
    bind_shadow(only_miss = TRUE) %>%
    add_label_shadow() %>%
    impute_lm(Blade_pressure ~ Ink_temperature) %>%
    impute_lm(Roughness ~ Ink_temperature) %>%
    impute_lm(Ink_pct ~ Ink_temperature)
```

**Combining multiple imputation models**  

```r
# Aggregate the imputation models
imp_models <- bind_rows(mean = imp_mean,
                        lm = lmp_lm,
                        .id = "imp_model")
head(imp_models)
```

## Dealing with outliers

* The 3-sigma rule (for normally distributed data)  
* The 1.5*IQR rule (more general)  

Outliers:  

+ any value lower that $Q1 - 1.5 x IQR$  
+ or any higher than $Q3 + 1.5 x IQR$

**Multivariate methods**  
* **Distance-based**: K-nearest neighbors (kNN) distance  
* **Density-based**: Local outlier factor (LOF)  

**1.5*IQR rule**  
Outliers:  

+ any value lower that $Q1 - 1.5 x IQR$  
+ or any higher than $Q3 + 1.5 x IQR$


**Distance-based methods**  

* Average distance to the K-nearest neighbors  

**Density-based methods**  

* Number of the neighboring points within a certain distance  

Assumption: outliers often lie far from their neighbors  

Local Outlier Factor (LOF)  
* Measures the local deviation of a data point with respect to its neighbors.  
* Outliers are observations with substantially lower density than their neighbors.  
`get.knn()` from FNN package  

* Each observation $x$ has an associated score LOF($x$)  
LOF($x$) $\approx$ 1 similar density to its neighbors  
LOF($x$) < 1 higher density than neighbors (inlier)  
LOF($x$) > 1 lower density than neighbors (outlier)   

`lof()` function from `dbscan` package

**What to do with outlier observations?**  
1. **Retention**: Keep them in your dataset and, if possible, use algorithms that are robust to outliers.
- e.g. K nearest-neighbors (kNN), tree-based methods (decision tree, random forest)  
2. **Imputation**: Use an imputation method to replace their value with a less extreme observation.  
- e.g. mode imputation, linear imputation, kNN imputation.  
3. **Capping**: Replace them with the value of the 5-th percentile (lower limit) or 95-th percentile (upper limit).  
4. **Exclusion**: Not recommended, especially in small datasets or those where a normal distribution cannot be assumed.  


```r
cars <- read.csv('./DATA/cars.csv')
cars <- cars[,1:3]
head(cars)
```

```
##   distance consume speed
## 1       28       5    26
## 2       12      42    30
## 3      112      55    38
## 4      129      39    36
## 5      185      45    46
## 6       83      64    50
```

```r
boxplot(cars$consume)
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-18-1.png" width="672" />

```r
consume_quartiles <- quantile(cars$consume)
consume_quartiles
```

```
##   0%  25%  50%  75% 100% 
##    4   41   46   52  122
```

```r
# Scale data and create scatterplot: cars_scaled
require(dplyr)
glimpse(cars)
```

```
## Rows: 388
## Columns: 3
## $ distance <int> 28, 12, 112, 129, 185, 83, 78, 123, 49, 119, 124, 118, 123, 247, 124, 173, 334, 118, 259, 118, 253, 142, 179, 118, 123, 124, 184, 184, 183, 184, 123, 118,…
## $ consume  <int> 5, 42, 55, 39, 45, 64, 44, 5, 64, 53, 56, 46, 59, 51, 47, 51, 56, 51, 49, 47, 55, 59, 57, 47, 59, 41, 57, 58, 55, 57, 53, 5, 56, 48, 43, 57, 52, 74, 48, 6…
## $ speed    <int> 26, 30, 38, 36, 46, 50, 43, 40, 26, 30, 42, 38, 59, 58, 46, 24, 36, 32, 39, 40, 32, 38, 37, 36, 62, 57, 21, 28, 29, 35, 51, 29, 58, 40, 36, 36, 37, 26, 23…
```

```r
cars_scaled <- as.data.frame(scale(cars))
plot(distance ~ consume, data = cars_scaled, 
     main = 'Fuel consumption vs. distance')
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-18-2.png" width="672" />

```r
# Calculate upper threshold: upper_th
upper_th <- consume_quartiles[4] + 
    1.5 * (consume_quartiles[4] - consume_quartiles[2])
upper_th
```

```
##  75% 
## 68.5
```

```r
# Print the sorted vector of distinct potential outliers
sort(unique(cars$consume[cars$consume > upper_th]))
```

```
##  [1]  69  71  74  79  81  87  99 108 115 122
```

```r
library(FNN)
# Compute KNN score
cars_knn <- get.knn(data = cars_scaled, k = 7)
cars1 <- cars
cars1$knn_score <- rowMeans(cars_knn$nn.dist)

# Print top 5 KNN scores and data point indices: top5_knn
(top5_knn <- order(cars1$knn_score, decreasing = TRUE)[1:5])
```

```
## [1] 320 107 335  56 190
```

```r
print(cars1$knn_score[top5_knn])
```

```
## [1] 4.472813 2.385471 2.246905 2.122242 1.740575
```

```r
# Plot variables using KNN score as size of points
plot(distance ~ consume, data = cars1, cex = knn_score, pch = 20)
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-18-3.png" width="672" />

```r
library(dbscan)
# Add lof_score column to cars1
cars1 <- cars
cars1$lof_score <- lof(cars_scaled, minPts = 7)

# Print top 5 LOF scores and data point indices: top5_lof
(top5_lof <- order(cars1$lof_score, decreasing = TRUE)[1:5])
```

```
## [1] 165 186 161 228  52
```

```r
print(cars1$lof_score[top5_lof])
```

```
## [1] 2.606151 2.574906 2.328733 2.252610 2.027528
```

```r
# Plot variables using LOF score as size of points
plot(distance ~ consume, data = cars1, 
     cex = lof_score, pch = 20)
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-18-4.png" width="672" />

# Data normalization

Data normalization (feature scaling) is not always needed for e.g. decision-tree-based models.  

Data normalization is beneficial for:  

* Support Vector Machines, K-narest neighbors, Logistic Regression
* Neural networks  
* Clustering algorithms (K-means, K-medoids, DBSCAN, etc.)  
* Feature extraction (Principal Component Analysis, Linear Discriminant Analysis, etc)  

**Min-max scaling**  

* Maps a numerical value $x$ to the [0,1] interval  

$$x' = \frac{x - min}{max - min}$$

* Ensures that all features will share the exact same scale.  
* Does not cope well with outliers.  

**Standardization (Z-score normalization)**  

* Maps a numerical value to $x$ to a new distribution with mean $\mu = 0$ and standard deviation $\sigma = 1$  

$$x' = \frac{x - \mu}{\sigma}$$

* More robust to outliers then min-max normalization.  
* Normalized data may be on different scales.  

**Example**  

```r
library(dplyr)

# generate data with different ranges
df <- data.frame(
    a = sample(seq(0, 2, length.out=20)),
    b = sample(seq(100, 500, length.out=20)))
# view data
glimpse(df)
```

```
## Rows: 20
## Columns: 2
## $ a <dbl> 0.0000000, 1.8947368, 0.4210526, 1.4736842, 1.3684211, 0.5263158, 0.1052632, 0.8421053, 1.5789474, 1.1578947, 1.7894737, 1.0526316, 0.7368421, 1.2631579, 0.63157…
## $ b <dbl> 226.3158, 478.9474, 163.1579, 415.7895, 457.8947, 352.6316, 205.2632, 289.4737, 500.0000, 310.5263, 436.8421, 268.4211, 331.5789, 142.1053, 184.2105, 100.0000, 1…
```

```r
# data ranges to see if normalization is needed
sapply(df, range)
```

```
##      a   b
## [1,] 0 100
## [2,] 2 500
```

```r
# ranges are different => normalize
# Apply min-max and standardization
nrm <- df %>% 
    mutate(a_MinMax = (a - min(a)) / (max(a) - min(a)), 
           b_MinMax = (b - min(b)) / (max(b) - min(b)),
           a_ZScore = (a - mean(a)) / sd(a),
           b_ZScore = (b - mean(b)) / sd(b)
    )

# ranges for normalized data
sapply(nrm, range)
```

```
##      a   b a_MinMax b_MinMax  a_ZScore  b_ZScore
## [1,] 0 100        0        0 -1.605793 -1.605793
## [2,] 2 500        1        1  1.605793  1.605793
```

```r
# plots
par(mfrow=c(1,3))
boxplot(nrm[, 1:2], main = 'Original')
boxplot(nrm[, 3:4], main = 'Min-Max')
boxplot(nrm[, 5:6], main = 'Z-Score')
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-19-1.png" width="672" />

**Sources**  
[Practicing Machine Learning Interview Questions in R on DataCamp](https://learn.datacamp.com/courses/practicing-machine-learning-interview-questions-in-r)  

<!--chapter:end:04_primary_data_analysis.Rmd-->

# Statistical distributions
## Normal Distribution


```r
x <- seq(-6, 6, length=100)
y1 <- dnorm(x, sd=1)
y2 <- dnorm(x, sd=0.5)
y3 <- dnorm(x, sd=2)
plot(x, y1, 
     xlim=c(-6,6), ylim=c(0,0.8),
     type="l", lwd=2,
     xlab="x value", ylab="Density",
     main="Normal distribution")
lines(x, y2, col="red")
lines(x, y3, col="green")
legend("topright", legend=c("sd = 1", "sd = 0.5", "sd = 2"),
       col=c("black", "red", "green"), 
       lty=c(1,1,1), lwd=2)
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-20-1.png" width="672" />

```r
# Area under the curve
# For normal distribution auc = 1 (probability for all)
library(DescTools)
AUC(x, y1)
```

```
## [1] 1
```

```r
AUC(x,y2)
```

```
## [1] 1
```

```r
AUC(x,y3) # to broad distribution, some samples out of the area
```

```
## [1] 0.9972921
```

## Bernoulli Distribution
**Bernoulli distribution** is the discrete probability distribution of a random variable which takes the value 1 with probability p and the value 0 with probability q=1-p [wiki](https://en.wikipedia.org/wiki/Bernoulli_distribution).  
Less formally, it can be thought of as a model for the set of possible outcomes of any single experiment that asks a yes–no question.  
The Bernoulli distribution is a special case of the binomial distribution with  n=1.  
Bernoulli process, a random process consisting of a sequence of independent Bernoulli trials.  

## Binomial Distribution
It describes the outcome of n independent trials in an experiment. Each trial is assumed to have only two outcomes, either success or failure. If the probability of a successful trial is p, then the probability of having x successful outcomes in an experiment of n independent trials is as follows.
$$f(x) = p^x(1-p)^{(n-x)}$$
where x = 0,1,2,3,4,.... 
The binomial distribution is the basis for the popular binomial test of statistical significance.  

Rules:
1. Must be a fixed number of trials.  
2. Trials must be independent (the outcome of one rial does not affect the others).  
3. Each trial has only two outcomes: success or failure.  
4. The probability of success remains the same in all trials.  

$n$ - number of trials  
$p$ - probability of successful outcome for each trial  
$q$ - probability of failure outcome for each trial  
$x$ - number of successes  
$P(x)$ - probability of the number of successes  

$q = 1 - p$  
$p = 1 - q$  

**Example**  

Suppose there are twelve multiple choice questions in an English class quiz. Each question has five possible answers, and only one of them is correct. Find the probability of having four or less correct answers if a student attempts to answer every question at random.

Solution  
Since only one out of five possible answers is correct, the probability of answering a question correctly by random is 1/5=0.2. We can find the probability of having exactly 4 correct answers by random attempts as follows.  


```r
dbinom(4, size=12, prob=0.2) 
```

```
## [1] 0.1328756
```
To find the probability of having four or less correct answers by random attempts, we apply the function dbinom with x = 0,…,4.  

```r
dbinom(0, size=12, prob=0.2) + 
dbinom(1, size=12, prob=0.2) + 
dbinom(2, size=12, prob=0.2) + 
dbinom(3, size=12, prob=0.2) + 
dbinom(4, size=12, prob=0.2)
```

```
## [1] 0.9274445
```
Alternatively, we can use the cumulative probability function for binomial distribution pbinom.  

```r
pbinom(4, size=12, prob=0.2)
```

```
## [1] 0.9274445
```

**Tasks**  
60% of people who purchase sports cars are men. If 10 sports car owners are randomly selected, find the probability that exactly 7 are men (A: 0.215).  

**Sources**  

[Binomial Distribution](http://www.r-tutor.com/elementary-statistics/probability-distributions/binomial-distribution)

[Statistics Lecture 5.3: A study of Binomial Probability Distributions](https://youtu.be/iGKSxMGX0Do)

Youtube 'Binomial distributions - Probabilities of probabilities' at 3Blue1Brown channel [part 1](https://youtu.be/8idr1WZ1A7Q)[part 2](https://youtu.be/ZA4JkHKZM50)  

## Beta distribution

The **beta distribution** is a family of continuous probability distributions defined on 
the interval [0, 1] parameterized by two positive **shape parameters**, denoted by $\alpha$ and $\beta$. 
[wiki](https://en.wikipedia.org/wiki/Beta_distribution).  

## Geometric Distribution
In probability theory and statistics, the geometric distribution is either one of two discrete probability distributions:

- The probability distribution of the number X of Bernoulli trials needed to get one success, supported on the set { 1, 2, 3, ... }  
- The probability distribution of the number Y = X − 1 of failures before the first success, supported on the set { 0, 1, 2, 3, ... }  
[wiki](https://en.wikipedia.org/wiki/Geometric_distribution)

$$(1 - p)^{k-1} for k trials where k \in {1,2,3,...}$$
$$(1 - p)^kp for k failures where k \in {0,1,2,...}$$

## Uniform Distributions
**Continuous uniform distribution**  is a family of symmetric probability distributions. The distribution describes an experiment where there is an arbitrary outcome that lies between certain bounds.  
The bounds are defined by the parameters, $a$ and $b$, which are the minimum and maximum values. The interval can either be closed $[a, b]$ or open $(a, b)$.  
Therefore, the distribution is often abbreviated $U(a, b)$, where $U$ stands for uniform distribution [wiki](https://en.wikipedia.org/wiki/Continuous_uniform_distribution).  

Probability density function:  
$$ \begin{cases} \frac{1}{b - a} & \text{for} \, x \in [a,b] \\ 1 & \text{for} \, x < a \, \text{or} \, x\, > b \end{cases}$$

## Poisson Distribution
**Poisson distribution** is a discrete probability distribution that expresses the probability of **a given number of events occurring in a 
fixed interval of time or space** if these events occur with a known constant mean rate and independently of the time since the last event. [wiki](https://en.wikipedia.org/wiki/Poisson_distribution)
Also knows as **Poisson Process**.  

$$P(k) = e^{\lambda}\frac{\lambda^k}{k!}$$

The Poisson distribution is also the limit of a binomial distribution, for which the probability of success for each trial equals λ divided by the number of trials, as the number of trials approaches infinity.  

According to the Central Limit Theorem for big $\lambda$ the Normal distribution will be the approximation for the Poisson distribution.  

Criteria of the Poisson process:  
1. Events are independent of each other. The occurrence of one event does not affect the probability another event will occur.
2. The average rate (events per time period) is constant.
3. Two events cannot occur at the same time.

Models:  
- Number of calls per minute in a call center.  
- Number of decay events that occur from a radioactive source in a given observation period.  
- The number of meteorites greater than 1 meter diameter that strike Earth in a year.  
- The number of patients arriving in an emergency room between 10 and 11 pm.  
- The number of laser photons hitting a detector in a particular time interval.

Example:  
12 cars cross the bridge in 1 min in average. What is the probability that 17 or more cars will cross the bridge?  

```r
ppois(16, lambda=12, lower=FALSE)
```

```
## [1] 0.101291
```

```r
# simulate Poisson
x <- rpois(n = 10000, lambda = 3)
hist(x)
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-24-1.png" width="672" />

```r
# Calculate the probability of break-even
mean(x >= 15)
```

```
## [1] 0
```

## Exponential Distribution
**Exponential distribution** is the probability distribution of the time between events in a Poisson point process, i.e., 
a process in which events occur continuously and independently at a constant average rate. [wiki](https://en.wikipedia.org/wiki/Exponential_distribution)

$$P(x) = \lambda e ^{ - \lambda x }$$

It is a particular case of the gamma distribution.

Suppose the mean checkout time of a supermarket cashier is 3 minutes. Find the probability of a customer checkout being completed by the cashier in less than 2 minutes.  


```r
pexp(2, rate=1/3)
```

```
## [1] 0.4865829
```

## Chi-squared Distribution
In probability theory and statistics, the $\{chi}^2$ distribution with k degrees of freedom is the distribution of a sum of the squares of 
$k$ independent standard normal random variables. [wiki](https://en.wikipedia.org/wiki/Chi-square_distribution)
If $X_1,X_2,…,X_m$ are $m$ independent random variables having the standard normal distribution, then the following quantity follows a **Chi-Squared distribution** with $m$ degrees of freedom. Its mean is $m$, and its variance is $2m$.

$$V = X_1^2 + X_2^2 + ... + X_m^{2} \tilde{} X_{(m)}^2$$

Summ of squares of standard normal, independen random variables are distributed according to the chi-square distribution with k degrees of freedom, where k is the number of random variables being summed.  

Chi-squared distribution applications:  
* Chi-square test of independence in contingency tables  
* Chi-square test of goodness of fit of observed data to hypothetical distributions  
* Likelihood-ratio test for nested models  
* Log-rank test in survival analysis  
* Cochran–Mantel–Haenszel test for stratified contingency tables  

```r
x = seq(0,25, length=100)
v = dchisq(x, df=7)
plot(x,v, type='l')
# quantiles of chi-squared
q = c(.25, .5, .75, .999)
qchi = qchisq(q, df = 7, lower.tail = TRUE)
abline(v=qchi, lty=2, col='red')
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-26-1.png" width="672" />

```r
qchi
```

```
## [1]  4.254852  6.345811  9.037148 24.321886
```

**Sources**  
[Seven Must-Know Statistical Distributions and Their Simulations for Data Science](https://towardsdatascience.com/seven-must-know-statistical-distributions-and-their-simulations-for-data-science-681c5ac41e32#:~:text=A%20statistical%20distribution%20is%20a,the%20random%20value%20it%20models.)
[Probability distributions](http://www.r-tutor.com/elementary-statistics/probability-distributions)
Hartmann, K., Krois, J., Waske, B. (2018): E-Learning Project SOGA: Statistics and Geospatial Data Analysis. Department of Earth Sciences, Freie Universitaet Berlin.  

<!--chapter:end:05_statistical_distributions.Rmd-->

# Hypothesis testing
## Hypothesis testing theory

Null hypothesis (H<sub>0</sub>):  
1. H<sub>0</sub>: m = μ  
2. H<sub>0</sub>: m \(\leq\) μ  
3. H<sub>0</sub>: m \(\geq\) μ  

Alternative hypotheses (H<sub>a</sub>):  
1. H<sub>a</sub>:m ≠ μ (different)  
2. H<sub>a</sub>:m > μ (greater)  
3. H<sub>a</sub>:m < μ (less)  

*Note*: Hypothesis 1. are called **two-tailed tests** and hypotheses 2. & 3. are called **one-tailed tests**.  

The p-value is the probability that the observed data could happen, under the condition that the null hypothesis is true.  

*Note*: p-value is not the probability that the null hypothesis is true.  
*Note*: Absence of evidence ⧧ evidence of absence.  

Cutoffs for hypothesis testing \*p < 0.05, \**p < 0.01, \***p < 0.001.  
If p value is less than significance level alpha (0.05), the hull hypothesies is rejected.  

|                     | not rejected ('negative')      | rejected ('positive')         |
|---------------------|--------------------------------|-------------------------------|
| H<sub>0</sub> true  | True negative (specificity)    | False Positive (Type I error) |
| H<sub>0</sub> false | False Negative (Type II error) | True positive (sensitivity)   |

Type II errors are usually more dangerous.  

Significance level $/alpha$ is the probability of mistakenly rejecting the null hypothesis, if the null hypothesis is true. This is also called false positive and type I error.

## Hypothesis test (Practice)

1. Check distribution (is normal?): shapiro.test()  
2. Distribution uniformity (are both normal?)  
3. Dependence of variable (are x and y dependent?)  
4. Difference of distribution parameters (are means different?)  


```r
###. 1 Is distribution normal?
town <- read.table("~/DataAnalysis/DATA/town_1959_2.csv", header=T, sep="\t", dec=".")
town
# histogram
hist(town[,3])

# log scale
hist(log(town[,3]), breaks=50)

# ? Is log scaled number of sitizens per town is normally distributed?
# test for normal distribution of dataset
data <- log(town[,3])
shapiro.test(data)

# H0: data is normally distributed
# W = 0.97467 - 
# p-value = 3.15e-12 -> p < alpha (0.01) - H0 is incorrect
# Our distripution is not normal

# Different tests for normality in package "nortest"
install.packages("nortest")
library(nortest)
ad.test(data)   # Anderson-Darling test
lillie.test(data) # Lilliefors (Kolmogorov-Smirnov) test

# Emperical rule: if n < 2000 -> shapiro.test, else -> lillie.test

# In theory it is possible to use method for normally distributed data, 
# even if they are not normal, but almost normal:
# 1. No outlayers
# 2. Symmetry of histogram
# 3. Bell-shaped form of histogram

# Method to make normal from not-normal distribution
# 1. Remove outlayers
# 2. Transform data to get symmetry: log, Boxcox
# 3. Bimodality: split samples into groups

# Boxcox transformation
install.packages("AID")
library(AID)
bctr = boxcoxnc(data)
bctr$results

### 2. Compair centers of two distributions
# Emperical rule: if notmal -> mean, else -> median
# if median -> Mann–Whitney-Wilcoxon (MWW) test
# if mean -> one of Student's tests

# Student t-test
# t.test(x,y, alternative="two.sided", paired=FALSE, var.equal=FALSE)

# How to compair if dispersions are the same?
# Fligner-Killeen test
# Brown-Forsythe test
#fligner.test(x~g, data=data.table)

# Example
x <- read.table ("~/DataAnalysis/R_data_analysis/DATA/Albuquerque Home Prices_data.txt", header=T)

names(x)
summary(x)

# Change -9999 to NA
x$AGE[x$AGE==-9999] <- NA
x$TAX[x$TAX==-9999] <- NA

fligner.test(PRICE~COR, data=x)
# Dispersion is the same for both distribution
# p > alpha (0.01)

t.test(x$PRICE[x$COR==1], x$PRICE[x$COR==0],
       alternative="less",
       paired=FALSE,
       var.equal=TRUE)
# p-value = 0.1977 (H0 is true)

# Mood's median test
names(x)
x1 <- x[x$NE==1,1]
x2 <- x[x$NE==0,1]
m <- median(c(x1,x2))
f11 <- sum(x1>m)
f12 <- sum(x2>m)
f21 <- sum(x1<=m)
f22 <- sum(x2<=m)
table <- matrix(c(f11,f12, f21, f22), nrow=2,ncol=2)
chisq.test(table)
# p=0.47 > alpha=0.05: H0 is true
# medians are the same

## Mann–Whitney-Wilcoxon (MWW) test
# Check if medians are the same
# U1 = n1*n2+{n1*(n1+1)/2 - T1}
# U1 = n1*n2+{n2*(n2+1)/2 - T2}
# U=min(U1,U2)
# Ti - сумма рангов в объединенной выборке наблюдений из выборки i
# n1, n2 - размеры выборок

wilcox.test(x,y,
            alternative="two.sided",
            paired=FALSE,
            exact=TRUE,
            correct=FALSE)

###. 3. Dependency of variables
# H0: x, y independent
# Correlation analysis
# Correlation (pearson) mesure linear dependency!
# Correlation k depends on outlayers (must be removed)

plot(x$SQFT, x$TAX)

cor(x$SQFT, x$TAX, use = "complete.obs")

cor(x$SQFT, x$TAX, use = "complete.obs", method = "spearman")

cor(x, use = "complete.obs", method = "spearman")

cor(x, use = "complete.obs", method = "pearson")

cor.test(x$SQFT, x$TAX, use = "complete.obs")
```

<!--chapter:end:06_hypothesis_testing.Rmd-->

# t-Procedures
## t-test and normal distribution
t-distribution assumes that the observations are **independent** and that they follow a **normal distribution**.
If the data are **dependent**, then p-values will likely be totally wrong (e.g., for positive correlation, too optimistic). Type II errors?  
It is good to test if observations are normally distributed. Otherwise we assume that data is normally distributed.  
Independence of observations is usually not testable, but can be reasonably assumed if the data collection process was random without replacement.  

FIXME: I do not understand this. Deviation data from normalyty will lead to type-I errors. I data is deviated from normal distribution, use **Wilcoxon test** or **permutation tests**.  

## One-sample t-test
One-sample t-test is used to compare the mean of one sample to a known standard (or theoretical/hypothetical) mean (μ).  
t-statistics:
\(t = \frac{m - \mu}{s/\sqrt{n}}\), where  
**m** is the sample **mean**  
**n** is the sample **size**  
**s** is the sample **standard deviation** with n−1 **degrees of freedom**  
**μ** is the **theoretical value**  


Q: And what should I do with this t-statistics?  
Q: What is the difference between t-test and ANOVA?  
Q: What is the smallest sample size which can be tested by t-test?  
Q: Show diagrams explaining why p-value of one-sided is smaller than two-sided tests.

## Practical example: t-test in R  

We want to test if **N** is different from given mean **μ**=0:  

```r
N = c(-0.01, 0.65, -0.17, 1.77, 0.76, -0.16, 0.88, 1.09, 0.96, 0.25)
t.test(N, mu = 0, alternative = "less")
```

```
## 
## 	One Sample t-test
## 
## data:  N
## t = 3.0483, df = 9, p-value = 0.9931
## alternative hypothesis: true mean is less than 0
## 95 percent confidence interval:
##      -Inf 0.964019
## sample estimates:
## mean of x 
##     0.602
```

```r
t.test(N, mu = 0, alternative = "two.sided")
```

```
## 
## 	One Sample t-test
## 
## data:  N
## t = 3.0483, df = 9, p-value = 0.01383
## alternative hypothesis: true mean is not equal to 0
## 95 percent confidence interval:
##  0.1552496 1.0487504
## sample estimates:
## mean of x 
##     0.602
```

```r
t.test(N, mu = 0, alternative = "greater")
```

```
## 
## 	One Sample t-test
## 
## data:  N
## t = 3.0483, df = 9, p-value = 0.006916
## alternative hypothesis: true mean is greater than 0
## 95 percent confidence interval:
##  0.239981      Inf
## sample estimates:
## mean of x 
##     0.602
```
FIXME: why it accepts all alternatives at the same time (less and greater?)  

## Two samples t-test
Do two different samples have the same mean?  
H<sub>0</sub>:  
1. H<sub>0</sub>: m<sub>1</sub> - m<sub>2</sub> = 0  
2. H<sub>0</sub>: m<sub>1</sub> - m<sub>2</sub> \(\leq\) 0  
3. H<sub>0</sub>: m<sub>1</sub> - m<sub>2</sub> \(\geq\) 0  

H<sub>a</sub>:  
1. H<sub>a</sub>: m<sub>1</sub> - m<sub>2</sub> ≠ 0 (different)  
2. H<sub>a</sub>: m<sub>1</sub> - m<sub>2</sub> > 0 (greater)  
3. H<sub>a</sub>: m<sub>1</sub> - m<sub>2</sub> < 0 (less)  

The paired sample t-test has four main assumptions:

1. The dependent variable must be **continuous** (interval/ratio).  
2. The observations are **independent** of one another.  
3. The dependent variable should be approximately **normally distributed**.  
4. The dependent variable should not contain any **outliers**.  

Continuous data can take on any value within a range (income, height, weight, etc.). The opposite of continuous data is discrete data, which can only take on a few values (Low, Medium, High, etc.). Occasionally, discrete data can be used to approximate a continuous scale, such as with **Likert-type scales**.  

t-statistics:
\(t=\frac{y - x}{SE}\), where
y and x are the samples means.
SE is the standard error for the difference.
If H<sub>0</sub> is correct, test statistic follows a t-distribution with n+m-2 degrees of freedom (n, m the number of observations in each sample).  

To apply t-test samples must be tested if they have equal variance:  
equal variance (homoscedastic). Type 3 means two samples, unequal variance (heteroscedastic).  

## Compare Student's t and normal distributions


```r
x <- seq(-4, 4, length=100)
hx <- dnorm(x)

degf <- c(1, 3, 8, 30)
colors <- c("red", "blue", "darkgreen", "gold", "black")
labels <- c("df=1", "df=3", "df=8", "df=30", "normal")

plot(x, hx, type="l", lty=2, xlab="x value",
     ylab="Density", main="Comparison of t Distributions")

for (i in 1:4){
    lines(x, dt(x,degf[i]), lwd=2, col=colors[i])
}

legend("topright", inset=.05, title="Distributions",
       labels, lwd=2, lty=c(1, 1, 1, 1, 2), col=colors)
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-29-1.png" width="672" />


To generate data with known mean and sd:  

```r
rnorm2 <- function(n,mean,sd) { mean+sd*scale(rnorm(n)) }
r <- rnorm2(100,4,1)
```


```r
### t-test
a = c(175, 168, 168, 190, 156, 181, 182, 175, 174, 179)
b = c(185, 169, 173, 173, 188, 186, 175, 174, 179, 180)

# test homogeneity of variances using Fisher’s F-test
var.test(a,b)
```

```
## 
## 	F test to compare two variances
## 
## data:  a and b
## F = 2.1028, num df = 9, denom df = 9, p-value = 0.2834
## alternative hypothesis: true ratio of variances is not equal to 1
## 95 percent confidence interval:
##  0.5223017 8.4657950
## sample estimates:
## ratio of variances 
##           2.102784
```

```r
# variance is homogene (can use var.equal=T in t.test)

# t-test
t.test(a,b, 
       var.equal=TRUE,   # variance is homogene (tested by var.test(a,b)) 
       paired=FALSE)     # samples are independent
```

```
## 
## 	Two Sample t-test
## 
## data:  a and b
## t = -0.94737, df = 18, p-value = 0.356
## alternative hypothesis: true difference in means is not equal to 0
## 95 percent confidence interval:
##  -10.93994   4.13994
## sample estimates:
## mean of x mean of y 
##     174.8     178.2
```

## Non-parametric tests
## Mann-Whitney U Rank Sum Test
1. The dependent variable is ordinal or continuous.  
2. The data consist of a randomly selected sample of independent observations from two independent groups.  
3. The dependent variables for the two independent groups share a similar shape.  

## Wilcoxon test
The **Wilcoxon** is a **non-parametric test** which works on normal and non-normal data. However, we usually prefer not to use it if we can assume that the data is normally distributed. The non-parametric test comes with less statistical power, this is a price that one has to pay for more flexible assumptions.  

# Tests for categorical variables
**Categorical variable** can take fixed number of possible values, assigning each individual or other unit of observation to a particular group or nominal category on the basis of some qualitative property.

## Chi-squared tests
The chi-squared test is most suited to large datasets. As a general rule, the chi-squared test is appropriate if at least 80% of the cells have an expected frequency of 5 or greater. In addition, none of the cells should have an expected frequency less than 1. If the expected values are very small, categories may be combined (if it makes sense to do so) to create fewer larger categories. Alternatively, Fisher’s exact test can be used.  


```r
data = rbind(c(83,35), c(92,43))
data
```

```
##      [,1] [,2]
## [1,]   83   35
## [2,]   92   43
```

```r
chisq.test(data, correct=F)
```

```
## 
## 	Pearson's Chi-squared test
## 
## data:  data
## X-squared = 0.14172, df = 1, p-value = 0.7066
```
 chisq.test(testor,correct=F)
## Fisher’s Exact test
R Example:  

| Group       | TumourShrinkage-No | TumourShrinkage-Yes | Total |
|-------------|--------------------|---------------------|-------|
| 1 Treatment | 8                  | 3                   | 11    |
| 2 Placebo   | 9                  | 4                   | 13    |
| 3 Total     | 17                 | 7                   | 24    |

The **null hypothesis** is that there is **no association** between treatment and tumour shrinkage.  
The **alternative hypothesis** is that there is **some association** between treatment group and tumour shrinkage.  


```r
data = rbind(c(8,3), c(9,4))
data
```

```
##      [,1] [,2]
## [1,]    8    3
## [2,]    9    4
```

```r
fisher.test(data)
```

```
## 
## 	Fisher's Exact Test for Count Data
## 
## data:  data
## p-value = 1
## alternative hypothesis: true odds ratio is not equal to 1
## 95 percent confidence interval:
##   0.1456912 10.6433317
## sample estimates:
## odds ratio 
##   1.176844
```
The output Fisher’s exact test tells us that the probability of observing such an extreme combination of frequencies is high, our p-value is 1.000 which is clearly greater than 0.05. In this case, there is **no evidence of an association** between treatment group and tumour shrinkage.  

# Multiple testing
When performing a large number of tests, the type I error is inflated: for α=0.05 and performing n tests, the probability of no false positive result is: 
0.095 x 0.95 x ... (n-times) <<< 0.095  
The larger the number of tests performed, the higher the probability of a false rejection!  
Many data analysis approaches in genomics rely on itemby-item (i.e. multiple) testing:  
Microarray or RNA-Seq expression profiles of “normal” vs “perturbed” samples: gene-by-gene  
ChIP-chip: locus-by-locus  
RNAi and chemical compound screens  
Genome-wide association studies: marker-by-marker  
QTL analysis: marker-by-marker and trait-by-trait  

**False positive rate** (FPR) - the proportion of false positives among all resulst.  

**False discovery rate** (FDR) - the proportion of false positives among all significant results.  

Example: 20,000 genes, 100 hits, 10 of them wrong.  
FPR: 0.05%  
FDR: 10%  

## The Bonferroni correction
The Bonferroni correction sets the significance cut-off at α/n.  

# Sources
[One-Sample T-test in R](http://www.sthda.com/english/wiki/one-sample-t-test-in-r)

## t-test

The data set shows energy expend in two groups of women: stature

```r
library(ISwR)
data(energy)
attach(energy)
```

```
## The following objects are masked from energy (pos = 4):
## 
##     expend, stature
```

```
## The following objects are masked from energy (pos = 9):
## 
##     expend, stature
```

```
## The following objects are masked from energy (pos = 11):
## 
##     expend, stature
```

```
## The following objects are masked from energy (pos = 30):
## 
##     expend, stature
```

```r
head(energy)
```

```
##   expend stature
## 1   9.21   obese
## 2   7.53    lean
## 3   7.48    lean
## 4   8.08    lean
## 5   8.09    lean
## 6  10.15    lean
```

```r
tapply(expend, stature, mean)
```

```
##      lean     obese 
##  8.066154 10.297778
```

H~0~: there is no difference in averages between lean and obese.  

```r
t.test(expend ~ stature)
```

```
## 
## 	Welch Two Sample t-test
## 
## data:  expend by stature
## t = -3.8555, df = 15.919, p-value = 0.001411
## alternative hypothesis: true difference in means is not equal to 0
## 95 percent confidence interval:
##  -3.459167 -1.004081
## sample estimates:
##  mean in group lean mean in group obese 
##            8.066154           10.297778
```
Alternative hypothesis is true - means are different.  
Mean difference is in between -3.5 and 1.0 with a probability 95%.  
The risk of error is 0.15%

### Two-tailed test
Compair two sets of variables.

```r
data(intake) # from package ISwR
attach(intake)
```

```
## The following objects are masked from intake (pos = 4):
## 
##     post, pre
```

```
## The following objects are masked from intake (pos = 9):
## 
##     post, pre
```

```
## The following objects are masked from intake (pos = 11):
## 
##     post, pre
```

```
## The following objects are masked from intake (pos = 30):
## 
##     post, pre
```

```r
head(intake)
```

```
##    pre post
## 1 5260 3910
## 2 5470 4220
## 3 5640 3885
## 4 6180 5160
## 5 6390 5645
## 6 6515 4680
```

```r
mean(post - pre)
```

```
## [1] -1320.455
```
Is difference of means significant?  

```r
t.test(pre, post, paired=TRUE)
```

```
## 
## 	Paired t-test
## 
## data:  pre and post
## t = 11.941, df = 10, p-value = 3.059e-07
## alternative hypothesis: true difference in means is not equal to 0
## 95 percent confidence interval:
##  1074.072 1566.838
## sample estimates:
## mean of the differences 
##                1320.455
```
The difference is significant with a probability 95%.  
The difference is in between 1074.1 and 1566.8 kJ/day  

<!--chapter:end:06_t_tests.Rmd-->

# Analysis of Variance (ANOVA)
## One-way ANOVA
**variance** = SS/df, where SS - sum of squares and df - degree of freedom  
\(SS = \displaystyle\sum_{i=1}^{n}{(x_i - \mu)^2}\), where  
\(\mu\) is the sample **mean**  
**n** is the sample **size** 

\(var(x) = \frac{1}{n}{\displaystyle\sum_{i=1}^{n}{(x_i - \mu)^2}}\)  



SST = SSE + SSC = W + B, where  
SST - Total Sum of Squares  
SSE - Error Sum of Squares - within (W)  
SSC - Sum of Squares Columns (treatmens) - between (B)

C - columns (treatments)  
N - total number of observations  

Mean squared of columns - MSC = SSC/df_columns, where df_columns = C-1  
Mean squared of error - MSE = SSE/df_error, where df_error = N-C  
Sum of squares (total) - SST, where df_total = N-1
F-statistics - F = MSC/MSE

Let's calculate degree of freedom for our example:  
df_columns = 3-1 = 2, MSC = SSC/2  
df_error = 21-3 = 18, MSE = SSE/18  
df_total = 21-1 = 20


```r
# 3 groups of students with scores (1-100):
a = c(82,93,61,74,69,70,53)
b = c(71,62,85,94,78,66,71)
c = c(64,73,87,91,56,78,87)

sq = function(x) { sum((x - mean(x))^2) }

sq(a)
```

```
## [1] 1039.429
```

```r
sq(b)
```

```
## [1] 751.4286
```

```r
sq(c)
```

```
## [1] 1021.714
```

Using R packages:

```r
# data
# Number of calories consumed by month:
may <- c(2166, 1568, 2233, 1882, 2019)
sep <- c(2279, 2075, 2131, 2009, 1793)
dec <- c(2226, 2154, 2583, 2010, 2190)

d <- stack(list(may=may, sep=sep, dec=dec))
d
```

```
##    values ind
## 1    2166 may
## 2    1568 may
## 3    2233 may
## 4    1882 may
## 5    2019 may
## 6    2279 sep
## 7    2075 sep
## 8    2131 sep
## 9    2009 sep
## 10   1793 sep
## 11   2226 dec
## 12   2154 dec
## 13   2583 dec
## 14   2010 dec
## 15   2190 dec
```

```r
names(d)
```

```
## [1] "values" "ind"
```

```r
oneway.test(values ~ ind, data=d, var.equal=TRUE)
```

```
## 
## 	One-way analysis of means
## 
## data:  values and ind
## F = 1.7862, num df = 2, denom df = 12, p-value = 0.2094
```


```r
# alternative using aov
res <- aov(values ~ ind, data = d)
res
```

```
## Call:
##    aov(formula = values ~ ind, data = d)
## 
## Terms:
##                      ind Residuals
## Sum of Squares  174664.1  586719.6
## Deg. of Freedom        2        12
## 
## Residual standard error: 221.1183
## Estimated effects may be unbalanced
```

```r
summary(res)
```

```
##             Df Sum Sq Mean Sq F value Pr(>F)
## ind          2 174664   87332   1.786  0.209
## Residuals   12 586720   48893
```

## Sources
Example for one-way ANOVA: [youtube](https://www.youtube.com/playlist?list=PLIeGtxpvyG-KA-BLkL391X__r0kU4_hm5) by Brandon Foltz  

<!--chapter:end:07_anova.Rmd-->

# t-test ANOVA difference
The t-test and ANOVA examine whether group means differ from one another. The t-test compares two groups, while ANOVA can do more than two groups.
The t-test ANOVA have three assumptions: independence assumption (the elements of one sample are not related to those of the other sample), normality assumption (samples are randomly drawn from the normally distributed populstions with unknown population means; otherwise the means are no longer best measures of central tendency, thus test will not be valid), and equal variance assumption (the population variances of the two groups are equal)
ANCOVA (analysis of covariance) includes covariates, interval independent variables, in the right-hand side to control their impacts. MANOVA (multivariate analysis of variance) has more than one left-hand side variable.  

![t-test and ANOVA usage.](./_bookdown_files/images/anova.gif)


<!--chapter:end:07_ttest_anova_dif.Rmd-->

# Chi-squared test


## Multinomial Goodness of Fit
A population is called multinomial if its data is categorical and belongs to a collection of discrete non-overlapping classes.

The null hypothesis for goodness of fit test for multinomial distribution is that the observed frequency fi is equal to an expected count $$e_i$$ in each category. 
It is to be rejected if the p-value of the following **Chi-squared test** statistics is less than a given significance level α.

Example
Survey response about the student’s smoking habit: "Heavy", "Regul" (regularly), "Occas" (occasionally) and "Never". 
The Smoke data is multinomial.


```r
library(MASS)
levels(survey$Smoke) 
```

```
## [1] "Heavy" "Never" "Occas" "Regul"
```

```r
smoke_freq = table(survey$Smoke) 
smoke_freq
```

```
## 
## Heavy Never Occas Regul 
##    11   189    19    17
```

```r
# estimated probabilities
smoke_prob = c(heavy = .045, 
               never = .795, 
               occas = .085, 
               regul = .075)
```

Determine whether the sample data in *smoke_freq* supports estimated probabilities in *smoke_prob* at .05 significance level.


```r
chisq.test(smoke_freq, p=smoke_prob)
```

```
## 
## 	Chi-squared test for given probabilities
## 
## data:  smoke_freq
## X-squared = 0.10744, df = 3, p-value = 0.9909
```

As the p-value 0.991 is greater than the .05 significance level, we do not reject the null hypothesis that the sample data in survey supports the smoking statistics.  

**Sources**

[Multinomial Goodness of Fit](https://www.r-tutor.com/elementary-statistics/goodness-fit/multinomial-goodness-fit)

<!--chapter:end:08_chi_squared_test.Rmd-->

# Non-parametric Methods
A statistical method is called non-parametric if it makes no assumption on the population distribution or sample size.

This is in contrast with most parametric methods in elementary statistics that assume the data is quantitative, the population has a normal distribution and the sample size is sufficiently large.

In general, conclusions drawn from non-parametric methods are not as powerful as the parametric ones. However, as non-parametric methods make fewer assumptions, they are more flexible, more robust, and applicable to non-quantitative data.

## Sign Test
## Wilcoxon Signed-Rank Test
## Mann-Whitney-Wilcoxon Test
## Kruskal-Wallis Test

<!--chapter:end:08_non-parametric_tests.Rmd-->

# Wilcoxon signed-rank test

The Wilcoxon signed-rank test is a non-parametric statistical hypothesis test used to compare two related samples, 
matched samples, or repeated measurements on a single sample to assess whether their population mean ranks differ 
(i.e. it is a paired difference test). It can be used as an alternative to the paired Student's t-test 
(also known as "t-test for matched pairs" or "t-test for dependent samples") when the distribution of the 
difference between two samples' means cannot be assumed to be normally distributed. A Wilcoxon signed-rank test can be 
used to determine whether two dependent samples were selected from populations having the same distribution. 
[wiki](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)  

Two data samples are matched if they come from repeated observations of the same subject. 
Using the Wilcoxon Signed-Rank Test, we can decide whether the corresponding data population distributions are 
identical without assuming them to follow the normal distribution.

Example
In the built-in data set named immer, the barley yield in years 1931 and 1932 of the same field are recorded. 
The yield data are presented in the data frame columns Y1 and Y2.  


```r
library(MASS)
head(immer)
```

```
##   Loc Var    Y1    Y2
## 1  UF   M  81.0  80.7
## 2  UF   S 105.4  82.3
## 3  UF   V 119.7  80.4
## 4  UF   T 109.7  87.2
## 5  UF   P  98.3  84.2
## 6   W   M 146.6 100.4
```

Problem
Without assuming the data to have normal distribution, test at .05 significance level if the barley yields of 1931 and 1932 in data set immer have identical data distributions.  

Solution
The null hypothesis is that the barley yields of the two sample years are identical populations. 
To test the hypothesis, we apply the wilcox.test function to compare the matched samples. 
For the paired test, we set the "paired" argument as TRUE. As the p-value turns out to be 0.005318, 
and is less than the .05 significance level, we reject the null hypothesis.  


```r
wilcox.test(immer$Y1, immer$Y2, paired=TRUE)
```

```
## Warning in wilcox.test.default(immer$Y1, immer$Y2, paired = TRUE): cannot compute exact p-value with ties
```

```
## 
## 	Wilcoxon signed rank test with continuity correction
## 
## data:  immer$Y1 and immer$Y2
## V = 368.5, p-value = 0.005318
## alternative hypothesis: true location shift is not equal to 0
```

Answer
At .05 significance level, we conclude that the barley yields of 1931 and 1932 from the data set immer are nonidentical populations.  

**Source**
[Wilcoxon Signed-Rank Test](http://www.r-tutor.com/elementary-statistics/non-parametric-methods/wilcoxon-signed-rank-test)  

<!--chapter:end:09_wilcoxon_signed-rank_test.Rmd-->

# Support Vector Machine

**Definition**: Support vector machine is a representation of the training data as points in space 
separated into categories by a clear gap that is as wide as possible. New examples are then mapped 
into that same space and predicted to belong to a category based on which side of the gap they fall.  
**Advantages**: Effective in high dimensional spaces and uses a subset of training points in the decision function so it is also memory efficient.  
**Disadvantages**: The algorithm does not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation.  

Parameters of SVM:  
* Type of kernel  
* Gamma value  
* C value  

<!--chapter:end:11_svm.Rmd-->

# Correlation


```r
df <- mtcars
# correlation miles per galon of petrol ~ horse power
cor.test(x = df$mpg, y = df$hp)

# get parameters of analysed correlation
fit <- cor.test(x = df$mpg, y = df$hp)
str(fit)
fit$p.value

# the same correlation in 'formula' form (see. ?cor.test)
?cor.test
cor.test(~mpg+hp, df)

# Plot using generic plot function of ggplot from ggplot2 package
plot(x = df$mpg, y = df$hp)

library(ggplot2)
# add color by number of engine's cylinders
ggplot(df, aes(x = mpg, y = hp, col = factor(cyl)))+geom_point(size=5)

# Subset of necessary data from mtcars
df.sub <- df[,c(1,3:7)]
# Scatterplots for all pairs of variables
pairs(df.sub)

# Correlations of all pairs of variables
cor(df.sub)

# Correlation using corr.test function from 'psych' package
# Correlation for all pairs of variables
library(psych)
fit.sub <- corr.test(df.sub)
fit.sub$r    # correlations
fit.sub$p    # dependecies of variables
```

<!--chapter:end:19_correlation.Rmd-->

# Methods and algorithms of machine learning  
**Regression Analysis**  
* Ordinary Least Squares Regression (OLSR)  
* Linear Regression  
* Logistic Regression  
* Stepwise Regression  
* Polynomial Regression  
* Locally Estimated Scatterplot Smoothing (LOESS)  
  
**Distance-based algorithms**  
* k-Nearest Neighbor (kNN)  
* Learning Vector Quantization (LVQ)  
* Self-organizing Map (SOM)  
  
**Regularization Algorithms**  
* Ridge Regression  
* Least Absolute Shrinkage and Selection Operator (LASSO)  
* Elastic Net  
* Least-Angle Regression (LARS)  
  
**Decision Tree Algorithms**  
* Classification and Regression Tree (CART)  
* Iterative Dichotomiser 3 (ID3)  
* C4.5 and C5.0  
* Chi-squared Automatic Interation Detection (CHAID)  
* Random Forest  
* Conditional Decision Trees  
  
**Bayesian Algorithms**  
* Naive Bayes  
* Gaussian Naive Bayes  
* Multinomial Naive Bayes  
* Bayesian Belief Network (BBN)  
* Bayesian Network (BN)  
  
**Clustering Algorithms**  
* k-Means  
* k-Medians  
* Partitioning Around MEdoids (PAM)  
* Hierarchical Clustering  
  
**Association Rule Mining Algorithms**  
* Apriori algorithm  
* Eclat algorithm  
* FP-growth algorithm  
* Context Based Rule Mining  
  
**Artifical Neural Network Algorithms**  
* Perceptron  
* Back-Propagation  
* Hopfield Network  
* Radial Basis Function Network (RBFN)  
  
**Deep Learining Algorithms**  
* Deep Boltzmann Machine (DBM)  
* Deep Belief Networks (DBN)  
* Convolutional NEural Network (CNN)  
* Stacked Auto-Encoders  
  
**Dimensionality Reduction Algorithms**  
* Principal Component Analysis (PCA)  
* Principal Compnent Regression (PCR)  
* Partial LEast Squares Regression (PLSR)  
* Multidimensional Scaling (MDS)  
* Linear Discriminant Analysis (LDA)  
* Mixtrue Discriminant Analysis (MDA)  
* Quadratic Discriminant Analysis (QDA)
1. PCA (linear)  
2. t-SNE (non-parametric/ nonlinear)  
3. Sammon mapping (nonlinear)  
4. Isomap (nonlinear)  
5. LLE (nonlinear)  
6. CCA (nonlinear)  
7. SNE (nonlinear)  
8. MVU (nonlinear)  
9. Laplacian Eigenmaps (nonlinear)  

  
**Ensemble Algorithms**  
* Boosting  
* Bagging  
* AdaBoost  
* Stacked Generalization (blending)  
* Gradient Boosting Machines (GBM)  
  
**Text Mining**  
* Automatic summarization  
* Named entity recognition (NER)  
* Optical character recognition (OCR)  
* Part-of-speech tagging
* Sentiment analysis
* Speech recognition
* Topic Modeling

<!--chapter:end:20_machine_learning.Rmd-->

# Machine Learning Functions Reference
## Linear Regression

```r
lm_model <- lm(y ∼ x1 + x2, data=as.data.frame(cbind(y,x1,x2)))
summary(lm_model)

lm(y ~ x1 + x2 + x3)     # multiple linear regression

lm(log(y) ~ x)           # log transformed
lm(sqrt(y) ~ x)          # sqrt transformed
lm( y ~ log(x))          # fields transformed
llm(log(y) ~ log(x))     # everything is transformed

lm(y ~ .)                # use all fields for regression model

lm(y ~ x + 0)            # forced zero intercept

lm(y ~ x*k)              # interaction of two variables

lm(y ~ x + k + x:k)      # product of xkl but without interaction

lm(y ~ (x + k + ... + l)^2) # all first order interactions

lm(y ~ I(x1 + x2))       # sum of variables
lm(y ~ I(x1^2))          # product of variables (not interation)

lm(y ~ x + I(x^2) + I(x^3)) # polynomial regression
lm(y ~ poly(x,3))           # same as previous

# Forward/backward stepwise regression
# improve model
fit <- lm(y ~ x1 + x2)
bwd.fit <- step(fit, direction = 'backward')
fwd.fit <- step(fit, direction = 'forward', scope( ~ x1 + x2))

# Test linear model
plot(m)            # plot residuals
car::outlier.test(m)
dwtest(m)          # Durbin-Watson Test of the model residuals

# Prediction
predicted_values <- predict(lm_model, newdata=as.data.frame(cbind(x1_test, x2_test)))

# Apriori
dataset <- read.csv("C:\\Datasets\\mushroom.csv", header = TRUE)
mushroom_rules <- apriori(as.matrix(dataset), parameter = list(supp = 0.8, conf = 0.9))
summary(mushroom_rules)
inspect(mushroom_rules)

# Logistic Regression
glm()
glm_mod <-glm(y ∼ x1+x2, family=binomial(link="logit"), data=as.data.frame(cbind(y,x1,x2))) 

# K-Means Clustering
kmeans_model <- kmeans(x=X, centers=m)

# k-Nearest Neighbor Classification
knn_model <- knn(train=X_train, test=X_test, cl=as.factor(labels), k=K)

# Naıve Bayes
library(e1071)
nB_model <- naiveBayes(y ∼ x1 + x2, data=as.data.frame(cbind(y,x1,x2)))

# Decision Trees (CART)
library(rpart)
# Tree-based models

rpart(formula, data, method,control)

formula	outcome~predictor1+predictor2+predictor3+etc.
data    data frame
method 'class' for a classification tree
       'anova' for a regression tree
control optional parameters for controlling tree growth.
        minsplit = 50 - minimal number of observation in a node
        cp=0.001 - cost coplexity factor

cart_model <- rpart(y ∼ x1 + x2, data=as.data.frame(cbind(y,x1,x2)), method="class")
        
printcp(fit)     # display cp table
plotcp(fit)		 # plot cross-validation results
rsq.rpart(fit)   # plot approximate R-squared and relative error for different splits (2 plots). labels are only appropriate for the "anova" method.
print(fit)       # print results
summary(fit)     # detailed results including surrogate splits
plot(fit)        # plot decision tree
text(fit)        # label the decision tree plot
post(fit, file=) # create postscript plot of decision tree

plot.rpart(cart_model)
text.rpart(cart_model)

# AdaBoost
# boosting functions - uses decision trees as base classifiers
library(rpart)
library(ada)
# Let X be the matrix of features, and labels be a vector of 0-1 class labels.
boost_model <- ada(x=X, y=labels)

# Support Vector Machines (SVM)
library(e1071)
# Let X be the matrix of features, and labels be a vector of 0-1 class labels. Let the regularization parameter be C. Use the following commands to generate the SVM model and view details:
svm_model <- svm(x=X, y=as.factor(labels), kernel ="radial", cost=C)
summary(svm_model)
```

<!--chapter:end:21_ml_fun_ref.Rmd-->

# Split data into train and test subsets
  
Here you can find several simple approaches to split data into train and test subset to fit and to test parameters of your model. 
We want to take 0.8 of our initial data to train our model.  
Data: `datasets::iris`.

1. First approach is to create a **vector containing randomly selected row ids** and to apply this vector to split data.  


```r
inTrain = sample(nrow(iris), nrow(iris)*0.8)

# split data
train = iris[inTrain, ]
test = iris[-inTrain, ]
```

2. The same idea to split data as before using `caret` package.  
The advantage is that `createDataPartition` function allows to split data many `times` and  use these subsets to estimate parameters of our model.  

```r
library(caret)
trainIndex <- createDataPartition(iris$Species, p=.8,
                                  list = FALSE,        # if FALSE - create a vector/matrix, if TRUE - create a list
                                  times = 1)           # how many subsets
# split data
train <- iris[trainIndex, ]
test <- iris[-trainIndex, ]
```

3. Another approch is to create a **logical vecotor** containing randomly distributed true/false and apply this vector to subset data.  

```r
inTrain = sample(c(TRUE, FALSE), nrow(iris), replace = T, prob = c(0.8,0.2))

# select data
train = iris[inTrain, ]
test = iris[!inTrain, ]
```

4. Using [`caTools`](https://cran.r-project.org/web/packages/caTools/index.html).

```r
library(caTools)
inTrain = sample.split(iris, SplitRatio = .8)
train = subset(iris, inTrain == TRUE)
test  = subset(iris, inTrain == FALSE)
```

5. Using [`dplyr`](https://cran.r-project.org/web/packages/dplyr/)

```r
library(dplyr)
iris$id <- 1:nrow(iris)
train <- iris %>% dplyr::sample_frac(.8)
test  <- dplyr::anti_join(iris, train, by = 'id')
```

<!--chapter:end:22_split_data_ways.Rmd-->

# Linear Regression

## Linear regression - theory
Linear regression model is a line y=ax+b, where sum of distances between all y=axi+b and given yi (sum of squares) is minimal.  

Assume that there is approximately a linear relationship between X and Y:  
$$ Y \approx \beta_0 + \beta_1X$$
where \(\beta\)~0~ is an __intercept__ and \(\beta\)~1~ is a __slope__

Parameters of the line could be calculated using __least squares methods__:

$$\beta_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}{(x_i - \bar{x})^2}} $$
$$\beta_0 = \bar{y} - \beta_1\bar{x} $$

## Generate random data set for the linear model
Suppose we want to simulate from the following linear model:
y = \(\beta\)~0~ + \(\beta\)~1~x + \(\epsilon\),  
where \(\epsilon\) ~ N(0,2^2^). Assume x ~ N(0,1^2^), \(\beta\)~0~ = 0.5, \(\beta\)~1~ = 2.  


```r
set.seed(20)
x <-rnorm(100)
e <- rnorm(100, 0, 2)
y <- 0.5 + 2*x + e
summary(y)
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
## -6.4084 -1.5402  0.6789  0.6893  2.9303  6.5052
```

```r
plot(x,y)
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-52-1.png" width="672" />

## Practical example
Practical example from [Wikipedia](https://en.wikipedia.org/wiki/Linear_least_squares_(mathematics))  
Set of data: (1,6), (2,5), (3, 7), (4,10)


```r
x <- c(1,2,3,4)
y <- c(6,5,7,10)
plot(y~x, xlim=c(0,5), ylim=c(4,10))
abline(3.5, 1.4)
r <- lm(y~x)
segments(x, y, x, r$fitted.values, col="green")
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-53-1.png" width="672" />

We have to find the line corresponding to the minimal sum of errors (distances from the each point to this line):
1. For all points:  
$$\beta_1 + 1\beta_2 = 6$$

$$\beta_1 + 2\beta_2 = 5$$
$$\beta_1 + 3\beta_2 = 7$$
$$\beta_1 + 4\beta_2 = 10$$
the least squares S:  
$$S(\beta_1, \beta_2) = [6 - (\beta_1 + 1\beta_2)]^2 + [5 - (\beta_1 + 2\beta_2)]^2 + [7 - (\beta_1 + 3\beta_2)]^2 + [10 - (\beta_1 + 4\beta_2)]^2 = 4\beta_1^2 + 30\beta_2^2 + 20\beta_1\beta_2 - 56\beta_1 - 154\beta_2 + 210$$
The minimum is:
$$\frac{\partial{S}}{\partial{\beta_1}} = 0 = 8 \beta_1 + 60\beta_2 - 154$$
$$\frac{\partial{S}}{\partial{\beta_2}} = 0 = 20 \beta_1 + 20\beta_2 - 56$$
Result in a system of two equations in two unkowns gives:
$$\beta_1 = 3.5$$
$$\beta_2 = 1.4$$
The line of best fit:  
$y = 3.5 + 1.4x$  
All possible regression lines goes through the intersection point $(\bar{x}, \bar{y})$

## Mean squared error (MSE)

**Standard error of train data**  
$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{f}(x_i))^2$$

**Standard error of learn data**  
$$MSE = \frac{1}{n_o}\sum_{i=1}^{n_o}(y_i^o - \hat{f}(x_i^o))^2$$

## Linear model in R

```r
# Height and weight vectors for 19 children
height <- c(69.1,56.4,65.3,62.8,63,57.3,59.8,62.5,62.5,59.0,51.3,64,56.4,66.5,72.2,65.0,67.0,57.6,66.6)
weight <- c(113,84,99,103,102,83,85,113,84,99,51,90,77,112,150,128,133,85,112)

plot(height,weight)
# Fit linear model
model <- lm(weight ~ height) # weight = slope*weight + intercept
abline(model)   # Regression line
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-54-1.png" width="672" />

```r
# correlation between variables
cor(height,weight)
```

```
## [1] 0.8848454
```

```r
# Get data from the model
#get the intercept(b0) and the slope(b1) values
model
```

```
## 
## Call:
## lm(formula = weight ~ height)
## 
## Coefficients:
## (Intercept)       height  
##    -143.227        3.905
```

```r
# detailed information about the model
summary(model)
```

```
## 
## Call:
## lm(formula = weight ~ height)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -16.816  -5.678   0.003   9.156  17.423 
## 
## Coefficients:
##              Estimate Std. Error t value Pr(>|t|)    
## (Intercept) -143.2266    31.1802  -4.594 0.000259 ***
## height         3.9047     0.4986   7.831 4.88e-07 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 10.89 on 17 degrees of freedom
## Multiple R-squared:  0.783,	Adjusted R-squared:  0.7702 
## F-statistic: 61.32 on 1 and 17 DF,  p-value: 4.876e-07
```

```r
# check all attributes calculated by lm
attributes(model)
```

```
## $names
##  [1] "coefficients"  "residuals"     "effects"       "rank"          "fitted.values" "assign"        "qr"            "df.residual"   "xlevels"       "call"         
## [11] "terms"         "model"        
## 
## $class
## [1] "lm"
```

```r
# getting only the intercept
model$coefficients[1] #or model$coefficients[[1]]
```

```
## (Intercept) 
##   -143.2266
```

```r
# getting only the slope
model$coefficients[2] #or model$coefficients[[2]]
```

```
##   height 
## 3.904675
```

```r
# checking the residuals
residuals(model)
```

```
##             1             2             3             4             5             6             7             8             9            10            11            12 
## -13.586376027   7.002990499 -12.748612814   1.013073510  -0.767861396   2.488783423  -5.272902901  12.184475869 -16.815524131  11.850836722  -6.083169400 -16.672535926 
##            13            14            15            16            17            18            19 
##   0.002990499  -4.434222250  11.309132932  17.422789545  14.613440486   3.317381064  -4.824689703
```

```r
# predict the weight for a given height, say 60 inches
model$coefficients[[2]]*60 + model$coefficients[[1]]
```

```
## [1] 91.05384
```

```r
# Mean squared error (MSE)
predicted.weights <- predict(model, newdata = as.data.frame(weight))
mse <- mean(( weight - predicted.weights)^2, na.rm = TRUE)
mse
```

```
## [1] 106.177
```

## Linear regression model for multiple parameters

## Choosing explanatory variables for the model
In general, less MSE then better prediction. We can arrange variables by their importance 
to predict and choose the number of significant variables.  

1. run model with different number of variables.  
2. Arrange MSE for each variable.  
3. Compair MSE for different number of variables using t-test.  
If difference is significant => use more variables.  

In the same way we can compair different models.  


```r
library(mosaicData)
head(CPS85)
```

```
##   wage educ race sex hispanic south married exper union age   sector
## 1  9.0   10    W   M       NH    NS Married    27   Not  43    const
## 2  5.5   12    W   M       NH    NS Married    20   Not  38    sales
## 3  3.8   12    W   F       NH    NS  Single     4   Not  22    sales
## 4 10.5   12    W   F       NH    NS Married    29   Not  47 clerical
## 5 15.0   12    W   M       NH    NS Married    40 Union  58    const
## 6  9.0   16    W   F       NH    NS Married    27   Not  49 clerical
```

```r
# relation wage ~ education
# wage - response variable
# educ, exper, age are explanatory variables

# linear model
model1 <- lm(wage ~ educ, data = CPS85)
model2 <- lm(wage ~ educ + age, data = CPS85)
model3 <- lm(wage ~ educ + age + exper, data = CPS85)

pred1 <- predict(model1, newdata = CPS85)
pred2 <- predict(model2, newdata = CPS85)
pred3 <- predict(model3, newdata = CPS85)

# Compair MSE
mse1 <- mean(( CPS85$wage - pred1)^2, na.rm = TRUE)
mse2 <- mean(( CPS85$wage - pred2)^2, na.rm = TRUE)
mse3 <- mean(( CPS85$wage - pred3)^2, na.rm = TRUE)

mse <- data.frame(model_1 = mse1,
                  model_2 = mse2,
                  model_3 = mse3)
mse
```

```
##    model_1  model_2 model_3
## 1 22.51575 21.03578 21.0353
```

```r
# Using both educ and age variables reduese MSE => improve model, where
# adding exper does not improve model
```

## Assessment of model performance for categorical data.
Errors for categorical data can be calculated as number of errors prediction model makes.  
Test whether predicted values match actual values.  
Likelihood: extract the probability that the model assigned to the observed outcome.  

## Confidence intervals for linear model

```r
# 0. Build linear model 
data("cars", package = "datasets")
model <- lm(dist ~ speed, data = cars)
# 1. Add predictions 
pred.int <- predict(model, interval = "prediction")
```

```
## Warning in predict.lm(model, interval = "prediction"): predictions on current data refer to _future_ responses
```

```r
mydata <- cbind(cars, pred.int)
# 2. Regression line + confidence intervals
library("ggplot2")
p <- ggplot(mydata, aes(speed, dist)) +
  geom_point() +
  stat_smooth(method = lm)
# 3. Add prediction intervals
p + geom_line(aes(y = lwr), color = "red", linetype = "dashed")+
    geom_line(aes(y = upr), color = "red", linetype = "dashed")
```

```
## `geom_smooth()` using formula 'y ~ x'
```

```
## Warning in grid.Call.graphics(C_polygon, x$x, x$y, index): semi-transparency is not supported on this device: reported only once per page
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-56-1.png" width="672" />

<!--chapter:end:23_linear_regression.Rmd-->

## Practical examples for linear model regression
In this simple example we have 6 persons (3 males and 3 femails) and their score from 0 to 10.  
We want to build a model to see the dependence of score on gender: score ~ gender + $\epsilon$, where $\epsilon$ is an error  


```r
# create data frame for the dataset
df = data.frame(gender=c(rep(0,3), rep(1,3)), score=c(10,8,7, 1,3,2))
df
```

```
##   gender score
## 1      0    10
## 2      0     8
## 3      0     7
## 4      1     1
## 5      1     3
## 6      1     2
```

```r
# build linear model
x = lm(score ~ gender, df)
summary(x)
```

```
## 
## Call:
## lm(formula = score ~ gender, data = df)
## 
## Residuals:
##          1          2          3          4          5          6 
##  1.667e+00 -3.333e-01 -1.333e+00 -1.000e+00  1.000e+00  1.110e-16 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)   8.3333     0.7454  11.180 0.000364 ***
## gender       -6.3333     1.0541  -6.008 0.003863 ** 
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 1.291 on 4 degrees of freedom
## Multiple R-squared:  0.9002,	Adjusted R-squared:  0.8753 
## F-statistic:  36.1 on 1 and 4 DF,  p-value: 0.003863
```

<!--chapter:end:24_linear_model_categorical.Rmd-->

# Linear regression complex cases

## Cars 

```r
df <- mtcars

# Subset of necessary data from mtcars
df.sub <- df[,c(1,3:7)]

# Linear regression model
fit <- lm(mpg ~ hp, df)
fit
summary(fit)

# Plot regression models using ggplot
# parameter geom_smooth builds regression model
library(ggplot2)
# auto model
ggplot(df, aes(hp, mpg))+geom_point(size=2)+geom_smooth()
# for linear model: method = 'lm'
ggplot(df, aes(hp, mpg))+geom_point(size=2)+geom_smooth(method = "lm")

# Split data into two groups by am - Transmission (0 = automatic, 1 = manual)
ggplot(df, aes(hp, mpg, col=factor(am)))+
    geom_point(size=2)+
    geom_smooth(method = "lm")

# Prediction of values using linear regression model
fitted_values_mpg <- data.frame(mpg=df$mpg, fitted=fit$fitted.values)
fitted_values_mpg
View(fitted_values_mpg)

# Lets predict galons of petrol for given horse powers
new_hp <- data.frame(hp=c(100, 150, 129, 300))

predict(fit, new_hp)

new_hp$mpg <- predict(fit, new_hp)
new_hp

# Lets make regression model for cylinders as numeric (not factor)
fit <- lm(mpg ~ cyl, df)
fit

# Regression line for cyclinders
ggplot(df, aes(cyl, mpg))+
    geom_point()+
    geom_smooth(method="lm")+
    theme(axis.text=element_text(size=25),
        axis.title=element_text(size=25, face="bold"))
```

## Linear regression modeling, compair with kNN

```r
library('GGally')
library('lmtest')
library('FNN')

# константы
my.seed <- 12345
train.percent <- 0.85

# загрузка данных
fileURL <- 'https://sites.google.com/a/kiber-guu.ru/msep/mag-econ/salary_data.csv?attredirects=0&d=1'

# преобразуем категориальные переменные в факторы
wages.ru <- read.csv(fileURL, row.names = 1, sep = ';', as.is = T)
wages.ru$male <- as.factor(wages.ru$male)
wages.ru$educ <- as.factor(wages.ru$educ)
wages.ru$forlang <- as.factor(wages.ru$forlang)

# обучающая выборка
set.seed(my.seed)
inTrain <- sample(seq_along(wages.ru$salary), 
                  nrow(wages.ru) * train.percent)
df.train <- wages.ru[inTrain, c(colnames(wages.ru)[-1], colnames(wages.ru)[1])]
df.test <- wages.ru[-inTrain, -1]

# Variable description
# salary – среднемесячная зарплата после вычета налогов за последние 12 месяцев (рублей);
# male – пол: 1 – мужчина, 0 – женщина;
# educ – уровень образования:
# 1 – 0-6 классов,
# 2 – незаконченное среднее (7-8 классов),
# 3 - незаконченное среднее плюс что-то ещё,
# 4 – законченное среднее,
# 5 – законченное среднее специальное, 6 – законченное высшее образование и выше;
# forlang - иност. язык: 1 – владеет, 0 – нет;
# exper – официальный стаж c 1.01.2002 (лет).

summary(df.train)
ggp <- ggpairs(df.train)
print(ggp, progress = F)
# цвета по фактору male
ggp <- ggpairs(df.train[, c('exper', 'male', 'salary')], 
               mapping = ggplot2::aes(color = male))
print(ggp, progress = F)
# цвета по фактору educ
ggp <- ggpairs(df.train[, c('exper', 'educ', 'salary')], 
               mapping = ggplot2::aes(color = educ))
print(ggp, progress = F)

# цвета по фактору forlang
ggp <- ggpairs(df.train[, c('exper', 'forlang', 'salary')], 
               mapping = ggplot2::aes(color = forlang))
print(ggp, progress = F)

# Linear regression model
model.1 <- lm(salary ~ . + exper:educ + exper:forlang + exper:male, data = df.train)
summary(model.1)

## Exclude uninfluencial parameters
# Exclude eper:educ as paramaeters are not important
model.2 <- lm(salary ~ . + exper:forlang + exper:male, data = df.train)
summary(model.2)

# Exclude male1:exper
model.3 <- lm(salary ~ . + exper:forlang, data = df.train)
summary(model.3)

# forlang1 is less important, has no sence
model.4 <- lm(salary ~ male + educ + exper, data = df.train)
summary(model.4)

df.train$educ <- as.numeric(df.train$educ)
df.test$educ <- as.numeric(df.test$educ)

model.6 <- lm(salary ~ ., data = df.train)
summary(model.6)

# Model 6 is week, let's add exper:male interactions
df.train$educ <- as.numeric(df.train$educ)

model.7 <- lm(salary ~ . + exper:male, data = df.train)
summary(model.7)

# Obviously the best decision is not to use interactions for modeling
# Test remainers
# тест Бройша-Пагана
bptest(model.6)
# статистика Дарбина-Уотсона
dwtest(model.6)
# графики остатков
par(mar = c(4.5, 4.5, 2, 1))
par(mfrow = c(1, 3))
plot(model.7, 1)
plot(model.7, 4)
plot(model.7, 5)

### Comparison with kNN-method
par(mfrow = c(1, 1))
# фактические значения y на тестовой выборке
y.fact <- wages.ru[-inTrain, 1]
y.model.lm <- predict(model.6, df.test)
MSE.lm <- sum((y.model.lm - y.fact)^2) / length(y.model.lm)

# kNN требует на вход только числовые переменные
df.train.num <- as.data.frame(apply(df.train, 2, as.numeric))
df.test.num <- as.data.frame(apply(df.test, 2, as.numeric))

for (i in 2:50){
    model.knn <- knn.reg(train = df.train.num[, !(colnames(df.train.num) %in% 'salary')], 
                         y = df.train.num[, 'salary'], 
                         test = df.test.num, k = i)
    y.model.knn <- model.knn$pred
    if (i == 2){
        MSE.knn <- sum((y.model.knn - y.fact)^2) / length(y.model.knn)
    } else {
        MSE.knn <- c(MSE.knn, 
                     sum((y.model.knn - y.fact)^2) / length(y.model.knn))
    }
}

# график
par(mar = c(4.5, 4.5, 1, 1))
plot(2:50, MSE.knn, type = 'b', col = 'darkgreen',
     xlab = 'значение k', ylab = 'MSE на тестовой выборке')
lines(2:50, rep(MSE.lm, 49), lwd = 2, col = grey(0.2), lty = 2)
legend('bottomright', lty = c(1, 2), pch = c(1, NA), 
       col = c('darkgreen', grey(0.2)), 
       legend = c('k ближайших соседа', 'регрессия (все факторы)'), 
       lwd = rep(2, 2))
```
**Source**

[Course 'Math modeling' practical work, State University of Management, Moscow](https://sites.google.com/a/kiber-guu.ru/r-practice/home)

## More complex example

```r
# Linear regression modeling, compair with kNN

# Source: Course 'Math modeling' practical work, State University of Management, Moscow
# link: https://sites.google.com/a/kiber-guu.ru/r-practice/home

library('GGally')
library('lmtest')
library('FNN')

# variable
my.seed <- 12345
train.percent <- 0.85

wages.ru <- read.csv("~/Projects/data_analysis/DATA/salary.csv", row.names = 1, sep = '\t', 
                     as.is = T)  # transform data into factors when possible
wages.ru$male <- as.factor(wages.ru$male)
wages.ru$educ <- as.factor(wages.ru$educ)
wages.ru$forlang <- as.factor(wages.ru$forlang)

# test data
set.seed(my.seed)
inTrain <- sample(seq_along(wages.ru$salary), 
                  nrow(wages.ru) * train.percent)
df.train <- wages.ru[inTrain, c(colnames(wages.ru)[-1], colnames(wages.ru)[1])]
df.test <- wages.ru[-inTrain, -1]

summary(df.train)
ggp <- ggpairs(df.train)
print(ggp, progress = F)
# цвета по фактору male
ggp <- ggpairs(df.train[, c('exper', 'male', 'salary')], 
               mapping = ggplot2::aes(color = male))
print(ggp, progress = F)
# цвета по фактору educ
ggp <- ggpairs(df.train[, c('exper', 'educ', 'salary')], 
               mapping = ggplot2::aes(color = educ))
print(ggp, progress = F)

# цвета по фактору forlang
ggp <- ggpairs(df.train[, c('exper', 'forlang', 'salary')], 
               mapping = ggplot2::aes(color = forlang))
print(ggp, progress = F)

# Linear regression model
model.1 <- lm(salary ~ . + exper:educ + exper:forlang + exper:male, data = df.train)
summary(model.1)

## Exclude uninfluencial parameters
# Exclude eper:educ as paramaeters are not important
model.2 <- lm(salary ~ . + exper:forlang + exper:male, data = df.train)
summary(model.2)

# Exclude male1:exper
model.3 <- lm(salary ~ . + exper:forlang, data = df.train)
summary(model.3)

# forlang1 is less important, has no sence
model.4 <- lm(salary ~ male + educ + exper, data = df.train)
summary(model.4)

df.train$educ <- as.numeric(df.train$educ)
df.test$educ <- as.numeric(df.test$educ)

model.6 <- lm(salary ~ ., data = df.train)
summary(model.6)

# Model 6 is week, let's add exper:male interactions
df.train$educ <- as.numeric(df.train$educ)

model.7 <- lm(salary ~ . + exper:male, data = df.train)
summary(model.7)

# Obviously the best decision is not to use interactions for modeling
# Test remainers
# тест Бройша-Пагана
bptest(model.6)
# статистика Дарбина-Уотсона
dwtest(model.6)
# графики остатков
par(mar = c(4.5, 4.5, 2, 1))
par(mfrow = c(1, 3))
plot(model.7, 1)
plot(model.7, 4)
plot(model.7, 5)

### Comparison with kNN-method
par(mfrow = c(1, 1))
# фактические значения y на тестовой выборке
y.fact <- wages.ru[-inTrain, 1]
y.model.lm <- predict(model.6, df.test)
MSE.lm <- sum((y.model.lm - y.fact)^2) / length(y.model.lm)

# kNN требует на вход только числовые переменные
df.train.num <- as.data.frame(apply(df.train, 2, as.numeric))
df.test.num <- as.data.frame(apply(df.test, 2, as.numeric))

for (i in 2:50){
    model.knn <- knn.reg(train = df.train.num[, !(colnames(df.train.num) %in% 'salary')], 
                         y = df.train.num[, 'salary'], 
                         test = df.test.num, k = i)
    y.model.knn <- model.knn$pred
    if (i == 2){
        MSE.knn <- sum((y.model.knn - y.fact)^2) / length(y.model.knn)
    } else {
        MSE.knn <- c(MSE.knn, 
                     sum((y.model.knn - y.fact)^2) / length(y.model.knn))
    }
}

# график
par(mar = c(4.5, 4.5, 1, 1))
plot(2:50, MSE.knn, type = 'b', col = 'darkgreen',
     xlab = 'значение k', ylab = 'MSE на тестовой выборке')
lines(2:50, rep(MSE.lm, 49), lwd = 2, col = grey(0.2), lty = 2)
legend('bottomright', lty = c(1, 2), pch = c(1, NA), 
       col = c('darkgreen', grey(0.2)), 
       legend = c('k ближайших соседа', 'регрессия (все факторы)'), 
       lwd = rep(2, 2))
```

## NEXT part

```r
# Linear regression (part 1)


# Linear Regression (part 2)
x <- read.table("~/DataAnalysis/R_data_analysis/DATA/Diamond.dat", header=T)

# Check 3 models
# 1. ax+b
# 2. ax^2+b
# 3. ax^2+bx+c
res.1 <- lm(x[,2]~x[,1])
summary(res.1)

ves2 <- x[,1]*x[,1]
res.2 <- lm(x[,2]~ves2)
summary(res.2)

res.3 <- lm(x[,2]~x[,1]+ves2)
summary(res.3)

# weight of diamands ~ weight^2
plot(x[,1]~ves2) 

# Conclusion: from 3 models, the 2d is optimal

### Временные ряды

ser.g.01 <- read.table("~/DataAnalysis/R_data_analysis/DATA/series_g.csv", header=T, sep=";")

# visual inspection
ser.g.01
dim(ser.g.01)
names(ser.g.01)

# plot vaiable
plot(ser.g.01$series_g, type="l")

log.ser.g <- log(ser.g.01$series_g)

plot(log.ser.g, type="l")
time. <- 1:(144+12)

month.01 <- rep(c(1,0,0,0,0,0,0,0,0,0,0,0), 12+1)
month.02 <- rep(c(0,1,0,0,0,0,0,0,0,0,0,0), 12+1)
month.03 <- rep(c(0,0,1,0,0,0,0,0,0,0,0,0), 12+1)
month.04 <- rep(c(0,0,0,1,0,0,0,0,0,0,0,0), 12+1)
month.05 <- rep(c(0,0,0,0,1,0,0,0,0,0,0,0), 12+1)
month.06 <- rep(c(0,0,0,0,0,1,0,0,0,0,0,0), 12+1)
month.07 <- rep(c(0,0,0,0,0,0,1,0,0,0,0,0), 12+1)
month.08 <- rep(c(0,0,0,0,0,0,0,1,0,0,0,0), 12+1)
month.09 <- rep(c(0,0,0,0,0,0,0,0,1,0,0,0), 12+1)
month.10 <- rep(c(0,0,0,0,0,0,0,0,0,1,0,0), 12+1)
month.11 <- rep(c(0,0,0,0,0,0,0,0,0,0,1,0), 12+1)
month.12 <- rep(c(0,0,0,0,0,0,0,0,0,0,0,1), 12+1)

log.ser.g[145:(144+12)] <-NA

ser.g.02 <- data.frame(log.ser.g, time., month.01, month.02, month.03, 
                       month.04, month.05, month.06, month.07, month.08, month.09, month.10, 
                       month.11, month.12)

ser.g.02

res.01 <- lm(log.ser.g ~ time. +              month.02 + month.03 + month.04 + 
                 month.05 + month.06 + month.07 + month.08 + month.09 + month.10 + 
                 month.11 + month.12, ser.g.02)

summary(res.01)
res.01$fitted.values

plot(ser.g.02$log.ser.g, type="l")
lines(res.01$fitted.values)

plot(ser.g.02$log.ser.g, type="l", col="green")
lines(res.01$fitted.values, col="red")

x.lg = predict.lm(res.01, ser.g.02)

x.lg

plot(x.lg, type="l", col="red")
lines(ser.g.02$log.ser.g, col="green")

y <- exp(x.lg)

plot(y, type="l", col="red")
lines(ser.g.01$series_g, col="green")
```

## NEXT Part

```r
# провести отбор оптимального подмножества переменных;
# отобрать предикторы методами пошагового включения и исключения;
# как построить ридж- и лассо-регрессию;
# как использовать снижение размерности: PCR и PLS;
# как применять эти методы в сочетании с кросс-валидацией.

library('ISLR')              # набор данных Hitters
library('leaps')             # функция regsubset() -- отбор оптимального 
#  подмножества переменных
library('glmnet')            # функция glmnet() -- лассо
library('pls')               # регрессия на главные компоненты -- pcr()
#  и частный МНК -- plsr()

my.seed <- 1
?Hitters
fix(Hitters)
names(Hitters)
dim(Hitters)
sum(is.na(Hitters$Salary))
Hitters <- na.omit(Hitters)
dim(Hitters)
sum(is.na(Hitters$Salary))

##  Отбор оптимального подмножества
# подгоняем модели с сочетаниями предикторов до 8 включительно
regfit.full <- regsubsets(Salary ~ ., Hitters)
summary(regfit.full)

# подгоняем модели с сочетаниями предикторов до 19 (максимум в данных)
regfit.full <- regsubsets(Salary ~ ., Hitters, nvmax = 19)
reg.summary <- summary(regfit.full)
reg.summary

# структура отчёта по модели (ищем характеристики качества)
names(reg.summary)

# R^2 и скорректированный R^2
round(reg.summary$rsq, 3)

# на графике
plot(1:19, reg.summary$rsq, type = 'b',
     xlab = 'Количество предикторов', ylab = 'R-квадрат')
# сода же добавим скорректированный R-квадрат
points(1:19, reg.summary$adjr2, col = 'red')
# модель с максимальным скорректированным R-квадратом
which.max(reg.summary$adjr2)

### 11
points(which.max(reg.summary$adjr2), 
       reg.summary$adjr2[which.max(reg.summary$adjr2)],
       col = 'red', cex = 2, pch = 20)
legend('bottomright', legend = c('R^2', 'R^2_adg'),
       col = c('black', 'red'), lty = c(1, NA),
       pch = c(1, 1))

# C_p
reg.summary$cp

# число предикторов у оптимального значения критерия
which.min(reg.summary$cp)

### 10
# график
plot(reg.summary$cp, xlab = 'Число предикторов',
     ylab = 'C_p', type = 'b')
points(which.min(reg.summary$cp),
       reg.summary$cp[which.min(reg.summary$cp)], 
       col = 'red', cex = 2, pch = 20)

# BIC
reg.summary$bic

# число предикторов у оптимального значения критерия
which.min(reg.summary$bic)

### 6
# график
plot(reg.summary$bic, xlab = 'Число предикторов',
     ylab = 'BIC', type = 'b')
points(which.min(reg.summary$bic),
       reg.summary$bic[which.min(reg.summary$bic)], 
       col = 'red', cex = 2, pch = 20)

# метод plot для визуализации результатов
?plot.regsubsets
plot(regfit.full, scale = 'r2')

plot(regfit.full, scale = 'adjr2')

plot(regfit.full, scale = 'Cp')

plot(regfit.full, scale = 'bic')

# коэффициенты модели с наименьшим BIC
round(coef(regfit.full, 6), 3)

##  Отбор путём пошагового включения и исключения переменных
# Пошаговое включение
regfit.fwd <- regsubsets(Salary ~ ., data = Hitters, nvmax = 19, method = 'forward')
summary(regfit.fwd)

regfit.bwd <- regsubsets(Salary ~ ., data = Hitters,
                         nvmax = 19, method = 'backward')
summary(regfit.bwd)

round(coef(regfit.full, 7), 3)

round(coef(regfit.fwd, 7), 3)

round(coef(regfit.bwd, 7), 3)

### Нахождение оптимальной модели при помощи методов проверочной выборки и перекрёстной проверки
set.seed(my.seed)
train <- sample(c(T, F), nrow(Hitters), rep = T)
test <- !train

# обучаем модели
regfit.best <- regsubsets(Salary ~ ., data = Hitters[train, ],
                          nvmax = 19)
# матрица объясняющих переменных модели для тестовой выборки
test.mat <- model.matrix(Salary ~ ., data = Hitters[test, ])

# вектор ошибок
val.errors <- rep(NA, 19)
# цикл по количеству предикторов
for (i in 1:19){
    coefi <- coef(regfit.best, id = i)
    pred <- test.mat[, names(coefi)] %*% coefi
    # записываем значение MSE на тестовой выборке в вектор
    val.errors[i] <- mean((Hitters$Salary[test] - pred)^2)
}
round(val.errors, 0)

# находим число предикторов у оптимальной модели
which.min(val.errors)

### 10
# коэффициенты оптимальной модели
round(coef(regfit.best, 10), 3)

# функция для прогноза для функции regsubset()
predict.regsubsets <- function(object, newdata, id, ...){
    form <- as.formula(object$call[[2]])
    mat <- model.matrix(form, newdata)
    coefi <- coef(object, id = id)
    xvars <- names(coefi)
    mat[, xvars] %*% coefi
}

# набор с оптимальным количеством переменных на полном наборе данных
regfit.best <- regsubsets(Salary ~ ., data = Hitters,
                          nvmax = 19)
round(coef(regfit.best, 10), 3)

# k-кратная кросс-валидация
# отбираем 10 блоков наблюдений
k <- 10
set.seed(my.seed)
folds <- sample(1:k, nrow(Hitters), replace = T)

# заготовка под матрицу с ошибками
cv.errors <- matrix(NA, k, 19, dimnames = list(NULL, paste(1:19)))

# заполняем матрицу в цикле по блокам данных
for (j in 1:k){
    best.fit <- regsubsets(Salary ~ ., data = Hitters[folds != j, ],
                           nvmax = 19)
    # теперь цикл по количеству объясняющих переменных
    for (i in 1:19){
        # модельные значения Salary
        pred <- predict(best.fit, Hitters[folds == j, ], id = i)
        # вписываем ошибку в матрицу
        cv.errors[j, i] <- mean((Hitters$Salary[folds == j] - pred)^2)
    }
}

# усредняем матрицу по каждому столбцу (т.е. по блокам наблюдений), 
#  чтобы получить оценку MSE для каждой модели с фиксированным 
#  количеством объясняющих переменных
mean.cv.errors <- apply(cv.errors, 2, mean)
round(mean.cv.errors, 0)

# на графике
plot(mean.cv.errors, type = 'b')
points(which.min(mean.cv.errors), mean.cv.errors[which.min(mean.cv.errors)],
       col = 'red', pch = 20, cex = 2)

# перестраиваем модель с 11 объясняющими переменными на всём наборе данных
reg.best <- regsubsets(Salary ~ ., data = Hitters, nvmax = 19)
round(coef(reg.best, 11), 3)

# из-за синтаксиса glmnet() формируем явно матрицу объясняющих...
x <- model.matrix(Salary ~ ., Hitters)[, -1]

# и вектор значений зависимой переменной
y <- Hitters$Salary

### Гребневая регрессия
# вектор значений гиперпараметра лямбда
grid <- 10^seq(10, -2, length = 100)

# подгоняем серию моделей ридж-регрессии
ridge.mod <- glmnet(x, y, alpha = 0, lambda = grid)

# размерность матрицы коэффициентов моделей
dim(coef(ridge.mod))
## [1]  20 100
# значение лямбда под номером 50
round(ridge.mod$lambda[50], 0)
## [1] 11498
# коэффициенты соответствующей модели
round(coef(ridge.mod)[, 50], 3)

# норма эль-два
round(sqrt(sum(coef(ridge.mod)[-1, 50]^2)), 2)

# всё то же для лямбды под номером 60
# значение лямбда под номером 50
round(ridge.mod$lambda[60], 0)

# коэффициенты соответствующей модели
round(coef(ridge.mod)[, 60], 3)

# норма эль-два
round(sqrt(sum(coef(ridge.mod)[-1, 60]^2)), 1)

# мы можем получить значения коэффициентов для новой лямбды
round(predict(ridge.mod, s = 50, type = 'coefficients')[1:20, ], 3)

## Метод проверочной выборки
set.seed(my.seed)
train <- sample(1:nrow(x), nrow(x)/2)
test <- -train
y.test <- y[test]

# подгоняем ридж-модели с большей точностью (thresh ниже значения по умолчанию)
ridge.mod <- glmnet(x[train, ], y[train], alpha = 0, lambda = grid,
                    thresh = 1e-12)
plot(ridge.mod)

# прогнозы для модели с лямбда = 4
ridge.pred <- predict(ridge.mod, s = 4, newx = x[test, ])
round(mean((ridge.pred - y.test)^2), 0)
# сравним с MSE для нулевой модели (прогноз = среднее)
round(mean((mean(y[train]) - y.test)^2), 0)
# насколько модель с лямбда = 4 отличается от обычной ПЛР
ridge.pred <- predict(ridge.mod, s = 0, newx = x[test, ], exact = T,
                      x = x[train, ], y = y[train])
round(mean((ridge.pred - y.test)^2), 0)
# predict с лямбдой (s) = 0 даёт модель ПЛР
lm(y ~ x, subset = train)

round(predict(ridge.mod, s = 0, exact = T, type = 'coefficients',
              x = x[train, ], y = y[train])[1:20, ], 3)

## Подбор оптимального значения лямбда с помощью перекрёстной проверки
# k-кратная кросс-валидация
set.seed(my.seed)
# оценка ошибки
cv.out <- cv.glmnet(x[train, ], y[train], alpha = 0)
plot(cv.out)

# значение лямбда, обеспечивающее минимальную ошибку перекрёстной проверки
bestlam <- cv.out$lambda.min
round(bestlam, 0)
## [1] 212
# MSE на тестовой для этого значения лямбды
ridge.pred <- predict(ridge.mod, s = bestlam, newx = x[test, ])
round(mean((ridge.pred - y.test)^2), 0)
## [1] 96016
# наконец, подгоняем модель для оптимальной лямбды, 
#  найденной по перекрёстной проверке
out <- glmnet(x, y, alpha = 0)
round(predict(out, type = 'coefficients', s = bestlam)[1:20, ], 3)

## Лассо
lasso.mod <- glmnet(x[train, ], y[train], alpha = 1, lambda = grid)
plot(lasso.mod)

# Подбор оптимального значения лямбда с помощью перекрёстной проверки
set.seed(my.seed)
cv.out <- cv.glmnet(x[train, ], y[train], alpha = 1)
plot(cv.out)

bestlam <- cv.out$lambda.min
lasso.pred <- predict(lasso.mod, s = bestlam, newx = x[test, ])
round(mean((lasso.pred - y.test)^2), 0)

# коэффициенты лучшей модели
out <- glmnet(x, y, alpha = 1, lambda = grid)
lasso.coef <- predict(out, type = 'coefficients', s = bestlam)[1:20, ]
round(lasso.coef, 3)
round(lasso.coef[lasso.coef != 0], 3)

### Лабораторная работа 3: регрессия при помощи методов PCR и PLS
### 6.7.1 Регрессия на главные компоненты
# кросс-валидация 
set.seed(2)   # непонятно почему они сменили зерно; похоже, опечатка
pcr.fit <- pcr(Salary ~ ., data = Hitters, scale = T, validation = 'CV')
summary(pcr.fit)

# график ошибок
validationplot(pcr.fit, val.type = 'MSEP')
# Подбор оптиального M: кросс-валидация на обучающей выборке
set.seed(my.seed)
pcr.fit <- pcr(Salary ~ ., data = Hitters, subset = train, scale = T,
               validation = 'CV')
validationplot(pcr.fit, val.type = 'MSEP')

# MSE на тестовой выборке
pcr.pred <- predict(pcr.fit, x[test, ], ncomp = 7)
round(mean((pcr.pred - y.test)^2), 0)

# подгоняем модель на всей выборке для M = 7 
#  (оптимально по методу перекрёстной проверки)
pcr.fit <- pcr(y ~ x, scale = T, ncomp = 7)
summary(pcr.fit)

# Регрессия по методу частных наименьших квадратов
set.seed(my.seed)
pls.fit <- plsr(Salary ~ ., data = Hitters, subset = train, scale = T,
                validation = 'CV')
summary(pls.fit)

# теперь подгоняем модель для найденного оптимального M = 2 
#  и оцениваем MSE на тестовой
pls.pred <- predict(pls.fit, x[test, ], ncomp = 2)
round(mean((pls.pred - y.test)^2), 0)
## [1] 101417
# подгоняем модель на всей выборке
pls.fit <- plsr(Salary ~ ., data = Hitters, scale = T, ncomp = 2)
summary(pls.fit)
```


<!--chapter:end:24_linear_regression_complex.Rmd-->

# Nonlinear regression

Nonlinear regression is a form of regression analysis in which observational data are modeled by a function which is a nonlinear combination of the model parameters and **depends on one or more independent variables**.  
Some nonlinear data sets can be transformed to a linear model.  
Sone can not be transformed. For such modeling methods of Numerical analysis should be applied such as Newton's method, Gauss-Newton method and Levenberg–Marquardt method.  


```r
Математическое моделирование
Практика 7
Нелинейные модели
В практических примерах ниже показано как:
    
    оценивать полиномиальную регрессию;
аппроксимировать нелинейные модели ступенчатыми функциями;
строить сплайны;
работать с локальной регрессией;
строить обобщённые линейные модели (GAM).
Модели: полиномиальная регрессия, полиномиальная логистическая регрессия, ступенчатая модель, обобщённая линейная модель.
Данные: Wage {ISLR}

Подробные комментарии к коду лабораторных см. в [1], глава 7.

library('ISLR')              # набор данных Auto
library('splines')           # сплайны
library('gam')               # обобщённые аддитивные модели
## Warning: package 'gam' was built under R version 3.3.3
## Loading required package: foreach
## Warning: package 'foreach' was built under R version 3.3.3
## Loaded gam 1.14
library('akima')             # график двумерной плоскости
## Warning: package 'akima' was built under R version 3.3.3
library('ggplot2')           # красивые графики
## Warning: package 'ggplot2' was built under R version 3.3.3
my.seed <- 1
Работаем с набором данных по зарплатам 3000 работников-мужчин среднеатлантического региона Wage. Присоединяем его к пространству имён функцией attach(), и дальше обращаемся напрямую к столбцам таблицы.

attach(Wage)
Работаем со столбцами:
    * wage – заработная плата работника до уплаты налогов;
* age – возраст работника в годах.

Полиномиальная регрессия
Зависимость зарплаты от возраста
Судя по графику ниже, ззаимосвязь заработной платы и возраста нелинейна. Наблюдается также группа наблюдений с высоким значением wage, граница проходит примерно на уровне 250.

gp <- ggplot(data = Wage, aes(x = age, y = wage))
gp <- gp + geom_point() + geom_abline(slope = 0, intercept = 250, col = 'red')
gp


Подгоняем полином четвёртой степени для зависимости заработной платы от возраста.

fit <- lm(wage ~ poly(age, 4), data = Wage)
round(coef(summary(fit)), 2)
##               Estimate Std. Error t value Pr(>|t|)
## (Intercept)     111.70       0.73  153.28     0.00
## poly(age, 4)1   447.07      39.91   11.20     0.00
## poly(age, 4)2  -478.32      39.91  -11.98     0.00
## poly(age, 4)3   125.52      39.91    3.14     0.00
## poly(age, 4)4   -77.91      39.91   -1.95     0.05
Функция poly(age, 4) создаёт таблицу с базисом ортогональных полиномов: линейные комбинации значений переменной age в степенях от 1 до 4.

round(head(poly(age, 4)), 3)
##           1      2      3      4
## [1,] -0.039  0.056 -0.072  0.087
## [2,] -0.029  0.026 -0.015 -0.003
## [3,]  0.004 -0.015  0.000  0.014
## [4,]  0.001 -0.015  0.005  0.013
## [5,]  0.012 -0.010 -0.011  0.010
## [6,]  0.018 -0.002 -0.017 -0.001
# можно получить сами значения age в заданных степенях
round(head(poly(age, 4, raw = T)), 3)
##       1    2      3       4
## [1,] 18  324   5832  104976
## [2,] 24  576  13824  331776
## [3,] 45 2025  91125 4100625
## [4,] 43 1849  79507 3418801
## [5,] 50 2500 125000 6250000
## [6,] 54 2916 157464 8503056
# на прогноз не повлияет, но оценки параметров изменяются
fit.2 <- lm(wage ~ poly(age, 4, raw = T), data = Wage)
round(coef(summary(fit.2)), 2)
##                        Estimate Std. Error t value Pr(>|t|)
## (Intercept)             -184.15      60.04   -3.07     0.00
## poly(age, 4, raw = T)1    21.25       5.89    3.61     0.00
## poly(age, 4, raw = T)2    -0.56       0.21   -2.74     0.01
## poly(age, 4, raw = T)3     0.01       0.00    2.22     0.03
## poly(age, 4, raw = T)4     0.00       0.00   -1.95     0.05
# границы изменения переменной age
agelims <- range(age)

# значения age, для которых делаем прогноз (от min до max с шагом 1)
age.grid <- seq(from = agelims[1], to = agelims[2])

# рассчитать прогнозы и их стандартные ошибки
preds <- predict(fit, newdata = list(age = age.grid), se = T)

# границы доверительного интервала для заработной платы
se.bands <- cbind(lower.bound = preds$fit - 2*preds$se.fit,
                  upper.bound = preds$fit + 2*preds$se.fit)

# смотрим результат
round(head(se.bands), 2)
##   lower.bound upper.bound
## 1       41.33       62.53
## 2       49.76       67.24
## 3       57.39       71.76
## 4       64.27       76.09
## 5       70.44       80.27
## 6       75.94       84.28
Рисуем левую панель графика со слайда 4 презентации (рис. 7.1 книги). Функция matlines() рисует грфик столбцов одной матрицы против столбцов другой.

# наблюдения
plot(age, wage, xlim = agelims, cex = 0.5, col = 'darkgrey')

# заголовок
title('Полином четвёртой степени')

# модель
lines(age.grid, preds$fit, lwd = 2, col = 'blue')

# доверительные интервалы прогноза
matlines(x = age.grid, y = se.bands, lwd = 1, col = 'blue', lty = 3)


Убедимся, что прогнозы по моделям с различными вызовами poly() совпадают.

# прогнозы по второму вызову модели
preds2 <- predict(fit.2, newdata = list(age = age.grid), se = T)

# максимальное расхождение между прогнозами по двум вариантам вызова модели
max(abs(preds$fit - preds2$fit))
## [1] 7.389644e-13
Теперь подбираем степень полинома, сравнивая модели со степенями от 1 до 5 с помощью дисперсионного анализа (ANOVA).

fit.1 <- lm(wage ~ age, data = Wage)
fit.2 <- lm(wage ~ poly(age, 2), data = Wage)
fit.3 <- lm(wage ~ poly(age, 3), data = Wage)
fit.4 <- lm(wage ~ poly(age, 4), data = Wage)
fit.5 <- lm(wage ~ poly(age, 5), data = Wage)

round(anova(fit.1, fit.2, fit.3, fit.4, fit.5), 2)
Res.Df
<dbl>
    RSS
<dbl>
    Df
<dbl>
    Sum of Sq
<dbl>
    F
<dbl>
    Pr(>F)
<dbl>
    2998	5022216	NA	NA	NA	NA
2997	4793430	1	228786.01	143.59	0.00
2996	4777674	1	15755.69	9.89	0.00
2995	4771604	1	6070.15	3.81	0.05
2994	4770322	1	1282.56	0.80	0.37
5 rows
Рассматриваются пять моделей, в которых степени полинома от age идут по возрастанию. В крайнем правом столбце таблице приводятся p-значения для проверки нулевой гипотезы: текущая модель не даёт статистически значимого сокращения RSS по сравнению с предыдущей моделью. Можно сделать вывод, что степени 3 достаточно, дальнейшее увеличение степени не даёт значимого улучшения качества модели.

Зависимость вероятности получать зарплату > 250 от возраста
Теперь вернёмся к группе наблюдений с высоким wage. Рассмотрим зависимость вероятности того, что величина зарплаты больше 250, от возраста.
Подгоняем логистическую регрессию и делаем прогнозы, для этого используем функцию для оценки обобщённой линейной модели  glm() и указываем тип модели binomial:
    
    fit <- glm(I(wage > 250) ~ poly(age, 4), data = Wage, family = 'binomial')

# прогнозы
preds <- predict(fit, newdata = list(age = age.grid), se = T)

# пересчитываем доверительные интервалы и прогнозы в исходные ЕИ
pfit <- exp(preds$fit) / (1 + exp(preds$fit))
se.bands.logit <- cbind(lower.bound = preds$fit - 2*preds$se.fit,
                        upper.bound = preds$fit + 2*preds$se.fit)
se.bands <- exp(se.bands.logit)/(1 + exp(se.bands.logit))

# результат - доверительный интервал для вероятности события 
#   "Заработная плата выше 250".   
round(head(se.bands), 3)
##   lower.bound upper.bound
## 1           0       0.002
## 2           0       0.003
## 3           0       0.004
## 4           0       0.005
## 5           0       0.006
## 6           0       0.007
Достраиваем график с 4 слайда презентации (рис. 7.1 книги). Рисуем правую панель.

# сетка для графика (изображаем вероятности, поэтому интервал изменения y мал)
plot(age, I(wage > 250), xlim = agelims, type = 'n', ylim = c(0, 0.2),
     ylab = 'P(Wage > 250 | Age)')

# фактические наблюдения показываем засечками
points(jitter(age), I((wage > 250) / 5), cex = 0.5, pch = '|', col = 'darkgrey')

# модель
lines(age.grid, pfit, lwd = 2, col = 'blue')

# доверительные интервалы
matlines(age.grid, se.bands, lwd = 1, col = 'blue', lty = 3)

# заголовок
title('Полином четвёртой степени')


Ступенчатые функции
Для начала определим несколько интервалов, на каждом из которых будем моделировать зависимость wage от age своим средним уровнем.

# нарезаем предиктор age на 4 равных интервала
table(cut(age, 4))
## 
## (17.9,33.5]   (33.5,49]   (49,64.5] (64.5,80.1] 
##         750        1399         779          72
# подгоняем линейную модель на интервалах
fit <- lm(wage ~ cut(age, 4), data = Wage)
round(coef(summary(fit)), 2)
##                        Estimate Std. Error t value Pr(>|t|)
## (Intercept)               94.16       1.48   63.79     0.00
## cut(age, 4)(33.5,49]      24.05       1.83   13.15     0.00
## cut(age, 4)(49,64.5]      23.66       2.07   11.44     0.00
## cut(age, 4)(64.5,80.1]     7.64       4.99    1.53     0.13
# прогноз -- это средние по `wage` на каждом интервале
preds.cut <- predict(fit, newdata = list(age = age.grid), se = T)

# интервальный прогноз
se.bands.cut <- cbind(lower.bound = preds.cut$fit - 2*preds.cut$se.fit,
                      upper.bound = preds.cut$fit + 2*preds.cut$se.fit)
Воспроизведём график со слайда 7 презентации (рис. 7.2 книги).

# наблюдения
plot(age, wage, xlim = agelims, cex = 0.5, col = 'darkgrey')

# модель
lines(age.grid, preds.cut$fit, lwd = 2, col = 'darkgreen')

# доверительные интервалы прогноза
matlines(x = age.grid, y = se.bands.cut, lwd = 1, col = 'darkgreen', lty = 3)

# заголовок
title('Ступенчатая функция')


Правая часть графика, для вероятности того, что зарплата выше 250.

fit <- glm(I(wage > 250) ~ cut(age, 4), data = Wage, family = 'binomial')

# прогнозы
preds <- predict(fit, newdata = list(age = age.grid), se = T)

# пересчитываем доверительные интервалы и прогнозы в исходные ЕИ
pfit <- exp(preds$fit) / (1 + exp(preds$fit))
se.bands.logit <- cbind(lower.bound = preds$fit - 2*preds$se.fit,
                        upper.bound = preds$fit + 2*preds$se.fit)
se.bands <- exp(se.bands.logit)/(1 + exp(se.bands.logit))

# результат - доверительный интервал для вероятности события 
#   "Заработная плата выше 250".   
round(head(se.bands), 3)
##   lower.bound upper.bound
## 1       0.003       0.016
## 2       0.003       0.016
## 3       0.003       0.016
## 4       0.003       0.016
## 5       0.003       0.016
## 6       0.003       0.016
# сетка для графика (изображаем вероятности, поэтому интервал изменения y мал)
plot(age, I(wage > 250), xlim = agelims, type = 'n', ylim = c(0, 0.2),
     ylab = 'P(Wage > 250 | Age)')

# фактические наблюдения показываем засечками
points(jitter(age), I((wage > 250) / 5), cex = 0.5, pch = '|', col = 'darkgrey')

# модель
lines(age.grid, pfit, lwd = 2, col = 'darkgreen')

# доверительные интервалы
matlines(age.grid, se.bands, lwd = 1, col = 'darkgreen', lty = 3)

# заголовок
title('Ступенчатая функция')


Сплайны
Построим кубический сплайн с тремя узлами.

# кубический сплайн с тремя узлами
fit <- lm(wage ~ bs(age, knots = c(25, 40, 60)), data = Wage)
# прогноз
preds.spl <- predict(fit, newdata = list(age = age.grid), se = T)
Теперь построим натуральный по трём узлам. Три узла это 6 степеней свободы. Если функции bs(), которая создаёт матрицу с базисом для полиномиального сплайна, передать только степени свободы, она распределит узлы равномерно. В данном случае это квартили распределения age.

# 3 узла -- 6 степеней свободы (столбцы матрицы)
dim(bs(age, knots = c(25, 40, 60)))
## [1] 3000    6
# если не указываем узлы явно...
dim(bs(age, df = 6))
## [1] 3000    6
#  они привязываются к квартилям
attr(bs(age, df = 6), 'knots')
##   25%   50%   75% 
## 33.75 42.00 51.00
# натуральный сплайн
fit2 <- lm(wage ~ ns(age, df = 4), data = Wage)
preds.spl2 <- predict(fit2, newdata = list(age = age.grid), se = T)
График сравнения кубического и натурального сплайнов.

par(mfrow = c(1, 1), mar = c(4.5, 4.5, 1, 8.5), oma = c(0, 0, 0, 0), xpd = T)

# наблюдения
plot(age, wage, col = 'grey')

# модель кубического сплайна
lines(age.grid, preds.spl$fit, lwd = 2)

# доверительный интервал
lines(age.grid, preds.spl$fit + 2*preds.spl$se, lty = 'dashed')
lines(age.grid, preds.spl$fit - 2*preds.spl$se, lty = 'dashed')

# натуральный сплайн
lines(age.grid, preds.spl2$fit, col = 'red', lwd = 2)

# легенда
legend("topright", inset = c(-0.7, 0),
       c('Кубический \n с 3 узлами', 'Натуральный'),
       lwd = rep(2, 2), col = c('black', 'red'))

# заголовок
title("Сплайны")


Построим график со слайда 20 (рисунок 7.8 книги).

par(mfrow = c(1, 1), mar = c(4.5, 4.5, 1, 1), oma = c(0, 0, 4, 0))

# наблюдения
plot(age, wage, xlim = agelims, cex = 0.5, col = 'darkgrey')

# заголовок
title('Сглаживающий сплайн')

# подгоняем модель с 16 степенями свободы
fit <- smooth.spline(age, wage, df = 16)

# подгоняем модель с подбором лямбды с помощью перекрёстной проверки
fit2 <- smooth.spline(age, wage, cv = T)
## Warning in smooth.spline(age, wage, cv = T): cross-validation with non-
## unique 'x' values seems doubtful
fit2$df
## [1] 6.794596
# рисуем модель
lines(fit, col = 'red', lwd = 2)
lines(fit2, col = 'blue', lwd = 2)
legend('topright', 
       c('16 df', '6.8 df'),
       col = c('red', 'blue'), lty = 1, lwd = 2, cex = 0.8)


Локальная регрессия
Строим график со слайда 24 (рис. 7.10).

plot(age, wage, xlim = agelims, cex = 0.5, col = 'darkgrey')

title('Локальная регрессия')

# подгоняем модель c окном 0.2
fit <- loess(wage ~ age, span = 0.2, data = Wage)

# подгоняем модель c окном 0.5
fit2 <- loess(wage ~ age, span = 0.5, data = Wage)

# рисум модели
lines(age.grid, predict(fit, data.frame(age = age.grid)),
      col = 'red', lwd = 2)
lines(age.grid, predict(fit2, data.frame(age = age.grid)),
      col = 'blue', lwd = 2)

# легенда
legend('topright', 
       c('s = 0.2', 's = 0.5'),
       col = c('red', 'blue'), lty = 1, lwd = 2, cex = 0.8)


Обобщённые аддитивные модели (GAM) с непрерывным откликом
Построим GAM на натуральных сплайнах степеней 4 (year), 5 (age) с категориальным предиктором edication.

# GAM на натуральных сплайнах
gam.ns <- gam(wage ~ ns(year, 4) + ns(age, 5) + education, data = Wage)
Также построим модель на сглаживающих сплайнах.

# GAM на сглаживающих сплайнах
gam.m3 <- gam(wage ~ s(year, 4) + s(age, 5) + education, data = Wage)
График со слайда 28 (рис. 7.12).

par(mfrow = c(1, 3))
plot(gam.m3, se = T, col = 'blue')


График со слайда 27 (рис. 7.11).

par(mfrow = c(1, 3))
plot(gam.ns, se = T, col = 'red')


График функции от year похож на прямую. Сделаем ANOVA, чтобы понять, какая степень для year лучше.

gam.m1 <- gam(wage ~ s(age, 5) + education, data = Wage)          # без year
gam.m2 <- gam(wage ~ year + s(age, 5) + education, data = Wage)   # year^1

anova(gam.m1, gam.m2, gam.m3, test = 'F')
Resid. Df
<dbl>
    Resid. Dev
<dbl>
    Df
<dbl>
    Deviance
<dbl>
    F
<dbl>
    Pr(>F)
<dbl>
    2990	3711731	NA	NA	NA	NA
2989	3693842	1.000000	17889.243	14.477130	0.0001447167
2986	3689770	2.999989	4071.134	1.098212	0.3485661430
3 rows
Третья модель статистически не лучше второй. Кроме того, один из параметров этой модели незначим.

# сводка по модели gam.m3
summary(gam.m3)
## 
## Call: gam(formula = wage ~ s(year, 4) + s(age, 5) + education, data = Wage)
## Deviance Residuals:
##     Min      1Q  Median      3Q     Max 
## -119.43  -19.70   -3.33   14.17  213.48 
## 
## (Dispersion Parameter for gaussian family taken to be 1235.69)
## 
##     Null Deviance: 5222086 on 2999 degrees of freedom
## Residual Deviance: 3689770 on 2986 degrees of freedom
## AIC: 29887.75 
## 
## Number of Local Scoring Iterations: 2 
## 
## Anova for Parametric Effects
##              Df  Sum Sq Mean Sq F value    Pr(>F)    
## s(year, 4)    1   27162   27162  21.981 2.877e-06 ***
## s(age, 5)     1  195338  195338 158.081 < 2.2e-16 ***
## education     4 1069726  267432 216.423 < 2.2e-16 ***
## Residuals  2986 3689770    1236                      
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Anova for Nonparametric Effects
##             Npar Df Npar F  Pr(F)    
## (Intercept)                          
## s(year, 4)        3  1.086 0.3537    
## s(age, 5)         4 32.380 <2e-16 ***
## education                            
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
Работаем с моделью gam.m2.

# прогноз по обучающей выборке
preds <- predict(gam.m2, newdata = Wage)
Также можно использовать в GAM локальные регрессии.

# GAM на локальных регрессиях
gam.lo <- gam(wage ~ s(year, df = 4) + lo(age, span = 0.7) + education, 
              data = Wage)

par(mfrow = c(1, 3))
plot.gam(gam.lo, se = T, col = 'green')


# модель со взаимодействием регрессоров year и age
gam.lo.i <- gam(wage ~ lo(year, age, span = 0.5) + education, data = Wage)
## Warning in lo.wam(x, z, wz, fit$smooth, which, fit$smooth.frame,
## bf.maxit, : liv too small. (Discovered by lowesd)
## Warning in lo.wam(x, z, wz, fit$smooth, which, fit$smooth.frame,
## bf.maxit, : lv too small. (Discovered by lowesd)
## Warning in lo.wam(x, z, wz, fit$smooth, which, fit$smooth.frame,
## bf.maxit, : liv too small. (Discovered by lowesd)
## Warning in lo.wam(x, z, wz, fit$smooth, which, fit$smooth.frame,
## bf.maxit, : lv too small. (Discovered by lowesd)
plot(gam.lo.i)


Логистическая GAM
Построим логистическую GAM для всероятности того, что wage превышает 250.

gam.lr <- gam(I(wage > 250) ~ year + s(age, df = 5) + education, 
              family = 'binomial', data = Wage)
par(mfrow = c(1, 3))
plot(gam.lr, se = T, col = 'green')


# уровни образования по группам разного достатка
table(education, I(wage > 250))
##                     
## education            FALSE TRUE
##   1. < HS Grad         268    0
##   2. HS Grad           966    5
##   3. Some College      643    7
##   4. College Grad      663   22
##   5. Advanced Degree   381   45
В категории с самым низким уровнем образования нет wage > 250, поэтому убираем её.

gam.lr.s <- gam(I(wage > 250) ~ year + s(age, df = 5) + education,
                family = 'binomial', data = Wage, 
                subset = (education != "1. < HS Grad"))
# график
par(mfrow = c(1, 3))
plot(gam.lr.s, se = T, col = 'green')


detach(Wage)
```



```r
# Nonlinear modeling
Математическое моделирование
Практика 8
Нелинейные модели
В практических примерах ниже показано как:
    
    строить регрессионные деревья;
строить деревья классификации;
делать обрезку дерева;
использовать бэггинг, бустинг, случайный лес для улучшения качества прогнозирования.
Модели: деревья решений.
Данные: Sales {ISLR}, Boston {ISLR}

Подробные комментарии к коду лабораторных см. в [1], глава 8.

library('tree')              # деревья
## Warning: package 'tree' was built under R version 3.4.4
library('ISLR')              # наборы данных
library('MASS')
library('randomForest')      # случайный лес
## Warning: package 'randomForest' was built under R version 3.4.4
## randomForest 4.6-14
## Type rfNews() to see new features/changes/bug fixes.
library('gbm')
## Warning: package 'gbm' was built under R version 3.4.4
## Loading required package: survival
## Loading required package: lattice
## Loading required package: splines
## Loading required package: parallel
## Loaded gbm 2.1.3
Деревья решений
Загрузим таблицу с данными по продажам детских кресел и добавим к ней переменную High – “высокие продажи” со значениями:
    
    Yes если продажи больше 8 (тыс. шт.);
No в противном случае.
?Carseats
## starting httpd help server ... done
attach(Carseats)

# новая переменная
High <- ifelse(Sales <= 8, "No", "Yes")

# присоединяем к таблице данных
Carseats <- data.frame(Carseats, High)
Строим дерево для категориального отклика High, отбросив непрерывный отклик Sales.

# модель бинарного  дерева
tree.carseats <- tree(High ~ . -Sales, Carseats)
summary(tree.carseats)
## 
## Classification tree:
## tree(formula = High ~ . - Sales, data = Carseats)
## Variables actually used in tree construction:
## [1] "ShelveLoc"   "Price"       "Income"      "CompPrice"   "Population" 
## [6] "Advertising" "Age"         "US"         
## Number of terminal nodes:  27 
## Residual mean deviance:  0.4575 = 170.7 / 373 
## Misclassification error rate: 0.09 = 36 / 400
# график результата
plot(tree.carseats)            # ветви
text(tree.carseats, pretty=0)  # подписи


tree.carseats                  # посмотреть всё дерево в консоли
## node), split, n, deviance, yval, (yprob)
##       * denotes terminal node
## 
##   1) root 400 541.500 No ( 0.59000 0.41000 )  
##     2) ShelveLoc: Bad,Medium 315 390.600 No ( 0.68889 0.31111 )  
##       4) Price < 92.5 46  56.530 Yes ( 0.30435 0.69565 )  
##         8) Income < 57 10  12.220 No ( 0.70000 0.30000 )  
##          16) CompPrice < 110.5 5   0.000 No ( 1.00000 0.00000 ) *
##          17) CompPrice > 110.5 5   6.730 Yes ( 0.40000 0.60000 ) *
##         9) Income > 57 36  35.470 Yes ( 0.19444 0.80556 )  
##          18) Population < 207.5 16  21.170 Yes ( 0.37500 0.62500 ) *
##          19) Population > 207.5 20   7.941 Yes ( 0.05000 0.95000 ) *
##       5) Price > 92.5 269 299.800 No ( 0.75465 0.24535 )  
##        10) Advertising < 13.5 224 213.200 No ( 0.81696 0.18304 )  
##          20) CompPrice < 124.5 96  44.890 No ( 0.93750 0.06250 )  
##            40) Price < 106.5 38  33.150 No ( 0.84211 0.15789 )  
##              80) Population < 177 12  16.300 No ( 0.58333 0.41667 )  
##               160) Income < 60.5 6   0.000 No ( 1.00000 0.00000 ) *
##               161) Income > 60.5 6   5.407 Yes ( 0.16667 0.83333 ) *
##              81) Population > 177 26   8.477 No ( 0.96154 0.03846 ) *
##            41) Price > 106.5 58   0.000 No ( 1.00000 0.00000 ) *
##          21) CompPrice > 124.5 128 150.200 No ( 0.72656 0.27344 )  
##            42) Price < 122.5 51  70.680 Yes ( 0.49020 0.50980 )  
##              84) ShelveLoc: Bad 11   6.702 No ( 0.90909 0.09091 ) *
##              85) ShelveLoc: Medium 40  52.930 Yes ( 0.37500 0.62500 )  
##               170) Price < 109.5 16   7.481 Yes ( 0.06250 0.93750 ) *
##               171) Price > 109.5 24  32.600 No ( 0.58333 0.41667 )  
##                 342) Age < 49.5 13  16.050 Yes ( 0.30769 0.69231 ) *
##                 343) Age > 49.5 11   6.702 No ( 0.90909 0.09091 ) *
##            43) Price > 122.5 77  55.540 No ( 0.88312 0.11688 )  
##              86) CompPrice < 147.5 58  17.400 No ( 0.96552 0.03448 ) *
##              87) CompPrice > 147.5 19  25.010 No ( 0.63158 0.36842 )  
##               174) Price < 147 12  16.300 Yes ( 0.41667 0.58333 )  
##                 348) CompPrice < 152.5 7   5.742 Yes ( 0.14286 0.85714 ) *
##                 349) CompPrice > 152.5 5   5.004 No ( 0.80000 0.20000 ) *
##               175) Price > 147 7   0.000 No ( 1.00000 0.00000 ) *
##        11) Advertising > 13.5 45  61.830 Yes ( 0.44444 0.55556 )  
##          22) Age < 54.5 25  25.020 Yes ( 0.20000 0.80000 )  
##            44) CompPrice < 130.5 14  18.250 Yes ( 0.35714 0.64286 )  
##              88) Income < 100 9  12.370 No ( 0.55556 0.44444 ) *
##              89) Income > 100 5   0.000 Yes ( 0.00000 1.00000 ) *
##            45) CompPrice > 130.5 11   0.000 Yes ( 0.00000 1.00000 ) *
##          23) Age > 54.5 20  22.490 No ( 0.75000 0.25000 )  
##            46) CompPrice < 122.5 10   0.000 No ( 1.00000 0.00000 ) *
##            47) CompPrice > 122.5 10  13.860 No ( 0.50000 0.50000 )  
##              94) Price < 125 5   0.000 Yes ( 0.00000 1.00000 ) *
##              95) Price > 125 5   0.000 No ( 1.00000 0.00000 ) *
##     3) ShelveLoc: Good 85  90.330 Yes ( 0.22353 0.77647 )  
##       6) Price < 135 68  49.260 Yes ( 0.11765 0.88235 )  
##        12) US: No 17  22.070 Yes ( 0.35294 0.64706 )  
##          24) Price < 109 8   0.000 Yes ( 0.00000 1.00000 ) *
##          25) Price > 109 9  11.460 No ( 0.66667 0.33333 ) *
##        13) US: Yes 51  16.880 Yes ( 0.03922 0.96078 ) *
##       7) Price > 135 17  22.070 No ( 0.64706 0.35294 )  
##        14) Income < 46 6   0.000 No ( 1.00000 0.00000 ) *
##        15) Income > 46 11  15.160 Yes ( 0.45455 0.54545 ) *
Теперь построим дерево на обучающей выборке и оценим ошибку на тестовой.

# ядро генератора случайных чисел
set.seed(2)

# обучающая выборка
train <- sample(1:nrow(Carseats), 200)

# тестовая выборка
Carseats.test <- Carseats[-train,]
High.test <- High[-train]

# строим дерево на обучающей выборке
tree.carseats <- tree(High ~ . -Sales, Carseats, subset = train)

# делаем прогноз
tree.pred <- predict(tree.carseats, Carseats.test, type = "class")

# матрица неточностей
tbl <- table(tree.pred, High.test)
tbl
##          High.test
## tree.pred No Yes
##       No  86  27
##       Yes 30  57
# оценка точности
acc.test <- sum(diag(tbl))/sum(tbl)
acc.test
## [1] 0.715
Обобщённая характеристика точности: доля верных прогнозов: 0.72.

Теперь обрезаем дерево, используя в качестве критерия частоту ошибок классификации. Функция cv.tree() проводит кросс-валидацию для выбора лучшего дерева, аргумент prune.misclass означает, что мы минимизируем ошибку классификации.

set.seed(3)
cv.carseats <- cv.tree(tree.carseats, FUN = prune.misclass)
# имена элементов полученного объекта
names(cv.carseats)
## [1] "size"   "dev"    "k"      "method"
# сам объект
cv.carseats
## $size
## [1] 19 17 14 13  9  7  3  2  1
## 
## $dev
## [1] 55 55 53 52 50 56 69 65 80
## 
## $k
## [1]       -Inf  0.0000000  0.6666667  1.0000000  1.7500000  2.0000000
## [7]  4.2500000  5.0000000 23.0000000
## 
## $method
## [1] "misclass"
## 
## attr(,"class")
## [1] "prune"         "tree.sequence"
# графики изменения параметров метода по ходу обрезки дерева ###################

# 1. ошибка с кросс-валидацией в зависимости от числа узлов
par(mfrow = c(1, 2))
plot(cv.carseats$size, cv.carseats$dev, type = "b",
     ylab = 'Частота ошибок с кросс-вал. (dev)',
     xlab = 'Число узлов (size)')
# размер дерева с минимальной ошибкой
opt.size <- cv.carseats$size[cv.carseats$dev == min(cv.carseats$dev)]
abline(v = opt.size, col = 'red', 'lwd' = 2)     # соотв. вертикальная прямая
mtext(opt.size, at = opt.size, side = 1, col = 'red', line = 1)

# 2. ошибка с кросс-валидацией в зависимости от штрафа на сложность
plot(cv.carseats$k, cv.carseats$dev, type = "b",
     ylab = 'Частота ошибок с кросс-вал. (dev)',
     xlab = 'Штраф за сложность (k)')


Как видно на графике слева, минимум частоты ошибок достигается при числе узлов 9. Оценим точность дерева с 9 узлами.

# дерево с 9 узлами
prune.carseats <- prune.misclass(tree.carseats, best = 9)

# визуализация
plot(prune.carseats)
text(prune.carseats, pretty = 0)


# прогноз на тестовую выборку
tree.pred <- predict(prune.carseats, Carseats.test, type = "class")

# матрица неточностей
tbl <- table(tree.pred, High.test)
tbl
##          High.test
## tree.pred No Yes
##       No  94  24
##       Yes 22  60
# оценка точности
acc.test <- sum(diag(tbl))/sum(tbl)
acc.test
## [1] 0.77
Точность этой модели чуть выше точности исходного дерева и составляет 0.77. Увеличив количество узлов, получим более глубокое дерево, но менее точное.

# дерево с 13 узлами
prune.carseats <- prune.misclass(tree.carseats, best = 15)

# визуализация
plot(prune.carseats)
text(prune.carseats, pretty = 0)


# прогноз на тестовую выборку
tree.pred <- predict(prune.carseats, Carseats.test, type = "class")

# матрица неточностей
tbl <- table(tree.pred, High.test)
tbl
##          High.test
## tree.pred No Yes
##       No  86  22
##       Yes 30  62
# оценка точности
acc.test <- sum(diag(tbl))/sum(tbl)
acc.test
## [1] 0.74
# сбрасываем графические параметры
par(mfrow = c(1, 1))
Регрессионные деревья
Воспользуемся набором данных Boston.

?Boston

# обучающая выборка
set.seed(1)
train <- sample(1:nrow(Boston), nrow(Boston)/2) # обучающая выборка -- 50%
Построим дерево регрессии для зависимой переменной medv: медианная стоимости домов, в которых живут собственники (тыс. долл.).

# обучаем модель
tree.boston <- tree(medv ~ ., Boston, subset = train)
summary(tree.boston)
## 
## Regression tree:
## tree(formula = medv ~ ., data = Boston, subset = train)
## Variables actually used in tree construction:
## [1] "lstat" "rm"    "dis"  
## Number of terminal nodes:  8 
## Residual mean deviance:  12.65 = 3099 / 245 
## Distribution of residuals:
##      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
## -14.10000  -2.04200  -0.05357   0.00000   1.96000  12.60000
# визуализация
plot(tree.boston)
text(tree.boston, pretty = 0)


Снова сделаем обрезку дерева в целях улучшения качества прогноза.

cv.boston <- cv.tree(tree.boston)

# размер дерева с минимальной ошибкой
plot(cv.boston$size, cv.boston$dev, type = 'b')
opt.size <- cv.boston$size[cv.boston$dev == min(cv.boston$dev)]
abline(v = opt.size, col = 'red', 'lwd' = 2)     # соотв. вертикальная прямая
mtext(opt.size, at = opt.size, side = 1, col = 'red', line = 1)


В данном случаем минимум ошибки соответствует самому сложному дереву, с 8 узлами. Покажем, как при желании можно обрезать дерево до 7 узлов (ошибка ненамного выше, чем минимальная).

# дерево с 7 узлами
prune.boston = prune.tree(tree.boston, best = 7)

# визуализация
plot(prune.boston)
text(prune.boston, pretty = 0)


Прогноз сделаем по необрезанному дереву, т.к. там ошибка, оцененная по методу перекрёстной проверки, минимальна.

# прогноз по лучшей модели (8 узлов)
yhat <- predict(tree.boston, newdata = Boston[-train, ])
boston.test <- Boston[-train, "medv"]

# график "прогноз -- реализация"
plot(yhat, boston.test)
# линия идеального прогноза
abline(0, 1)


# MSE на тестовой выборке
mse.test <- mean((yhat - boston.test)^2)
MSE на тестовой выборке равна 25.05 (тыс.долл.).

Бэггинг и метод случайного леса
Рассмотрим более сложные методы улучшения качества дерева. Бэггинг – частный случай случайного леса с m=p, поэтому и то, и другое можно построить функцией randomForest().

Для начала используем бэггинг, причём возьмём все 13 предикторов на каждом шаге (аргумент mtry).

# бэггинг с 13 предикторами
set.seed(1)
bag.boston <- randomForest(medv ~ ., data = Boston, subset = train, 
                           mtry = 13, importance = TRUE)
bag.boston
## 
## Call:
##  randomForest(formula = medv ~ ., data = Boston, mtry = 13, importance = TRUE,      subset = train) 
##                Type of random forest: regression
##                      Number of trees: 500
## No. of variables tried at each split: 13
## 
##           Mean of squared residuals: 11.15723
##                     % Var explained: 86.49
# прогноз
yhat.bag = predict(bag.boston, newdata = Boston[-train, ])

# график "прогноз -- реализация"
plot(yhat.bag, boston.test)
# линия идеального прогноза
abline(0, 1)


# MSE на тестовой
mse.test <- mean((yhat.bag - boston.test)^2)
mse.test
## [1] 13.50808
Ошибка на тестовой выборке равна 13.51.
Можно изменить число деревьев с помощью аргумента ntree.

bag.boston <- randomForest(medv ~ ., data = Boston, subset = train,
                           mtry = 13, ntree = 25)

# прогноз
yhat.bag <- predict(bag.boston, newdata = Boston[-train, ])

# MSE на тестовой
mse.test <- mean((yhat.bag - boston.test)^2)
mse.test
## [1] 13.94835
Но, как видно, это только ухудшает прогноз.
Теперь попробуем вырастить случайный лес. Берём 6 предикторов на каждом шаге.

# обучаем модель
set.seed(1)
rf.boston <- randomForest(medv ~ ., data = Boston, subset = train,
                          mtry = 6, importance = TRUE)

# прогноз
yhat.rf <- predict(rf.boston, newdata = Boston[-train, ])

# MSE на тестовой выборке
mse.test <- mean((yhat.rf - boston.test)^2)

# важность предикторов
importance(rf.boston)  # оценки 
##           %IncMSE IncNodePurity
## crim    12.132320     986.50338
## zn       1.955579      57.96945
## indus    9.069302     882.78261
## chas     2.210835      45.22941
## nox     11.104823    1044.33776
## rm      31.784033    6359.31971
## age     10.962684     516.82969
## dis     15.015236    1224.11605
## rad      4.118011      95.94586
## tax      8.587932     502.96719
## ptratio 12.503896     830.77523
## black    6.702609     341.30361
## lstat   30.695224    7505.73936
varImpPlot(rf.boston)  # графики


Ошибка по модели случайного леса равна 11.66, что ниже, чем для бэггинга.

Бустинг
Построим 5000 регрессионных деревьев с глубиной 4.

set.seed(1)
boost.boston <- gbm(medv ~ ., data = Boston[train, ], distribution = "gaussian",
                    n.trees = 5000, interaction.depth = 4)
# график и таблица относительной важности переменных
summary(boost.boston)

# графики частной зависимости для двух наиболее важных предикторов
par(mfrow = c(1, 2))
plot(boost.boston, i = "rm")
plot(boost.boston, i = "lstat")


# прогноз
yhat.boost <- predict(boost.boston, newdata = Boston[-train, ], n.trees = 5000)

# MSE на тестовой
mse.test <- mean((yhat.boost - boston.test)^2)
mse.test
## [1] 11.84434
Настройку бустинга можно делать с помощью гиперпараметра λ (аргумент shrinkage). Установим его равным 0.2.

# меняем значение гиперпараметра (lambda) на 0.2 -- аргумент shrinkage
boost.boston <- gbm(medv ~ ., data = Boston[train, ], distribution = "gaussian",
                    n.trees = 5000, interaction.depth = 4, 
                    shrinkage = 0.2, verbose = F)
# прогноз
yhat.boost <- predict(boost.boston, newdata = Boston[-train, ], n.trees = 5000)

# MSE а тестовой
mse.test <- mean((yhat.boost - boston.test)^2)
mse.test
## [1] 11.51109
Таким образом, изменив гиперпараметр, мы ещё немного снизили ошибку прогноза.
```

<!--chapter:end:24_nonlinear_regression.Rmd-->

# Multiple linear regression

Source: Анализ данных в R. Множественная линейная регрессия  


```r
# Dataset swiss
?swiss
swiss <- data.frame(swiss)
str(swiss)

# Histogram of fertility
hist(swiss$Fertility, col='red')

# Numeric predictors for Fertility prediction
fit <- lm(Fertility ~ Examination + Catholic, data = swiss)
summary(fit)
# the principal predictor is an 'examination' with negative correlation.

# Interaction of variables 'examination' and 'catholics' '*'
fit2 <- lm(Fertility ~ Examination*Catholic, data = swiss)
summary(fit2)

confint(fit2)

# Categorical predictors
# Histogram obviously have two parts -> we can split data for two factors
hist(swiss$Catholic, col = 'red')

# Lets split 'Catholics' for two groups: with many 'lots' and few 'few'
swiss$religious <- ifelse(swiss$Catholic > 60, 'Lots', 'Few')
swiss$religious <- as.factor(swiss$religious)

fit3 <- lm(Fertility ~ Examination + religious, data = swiss)
summary(fit3)

# Interaction of variables
fit4 <- lm(Fertility ~ religious*Examination, data = swiss)
summary(fit4)

# plots

ggplot(swiss, aes(x = Examination, y = Fertility)) + 
  geom_point() 

ggplot(swiss, aes(x = Examination, y = Fertility)) + 
  geom_point() + 
  geom_smooth()

ggplot(swiss, aes(x = Examination, y = Fertility)) + 
  geom_point() + 
  geom_smooth(method = 'lm')

ggplot(swiss, aes(x = Examination, y = Fertility, col = religious)) + 
  geom_point() 

ggplot(swiss, aes(x = Examination, y = Fertility, col = religious)) + 
  geom_point()  + 
  geom_smooth()

ggplot(swiss, aes(x = Examination, y = Fertility, col = religious)) + 
  geom_point()  + 
  geom_smooth(method = 'lm')


#

fit5 <- lm(Fertility ~ religious*Infant.Mortality*Examination, data = swiss)
summary(fit5)


# model comparison

rm(swiss)
swiss <- data.frame(swiss)

fit_full <- lm(Fertility ~ ., data = swiss)
summary(fit_full)

fit_reduced1 <- lm(Fertility ~ Infant.Mortality + Examination + Catholic + Education, data = swiss)
summary(fit_reduced1)

anova(fit_full, fit_reduced1)

fit_reduced2 <- lm(Fertility ~ Infant.Mortality + Education + Catholic + Agriculture, data = swiss)
summary(fit_reduced2)

anova(fit_full, fit_reduced2)

# model selection

optimal_fit <-  step(fit_full, direction = 'backward')
summary(optimal_fit)
```

<!--chapter:end:26_multiple_regression.Rmd-->

# Spline model

In this example we will generate data from a given function and then build a model using splines and estimate quality of the model.

## Generate dataset from a given function


```r
# parameters to generate a dataset
n.all <- 100             # number of observations
train.percent <- 0.85    # portion of the data for training
res.sd <- 1              # standard deviation of noise
x.min <- 5               # min limit of the data
x.max <- 105             # max limit of the data

# generate x
set.seed(1)       # to get reproducible results by randomizer
x <- runif(x.min, x.max, n = n.all)

# noise from normal destibution
set.seed(1)
res <- rnorm(mean = 0, sd = res.sd, n = n.all)

# generate y using a given function
y.func <- function(x) {4 - 2e-02*x + 5.5e-03*x^2 - 4.9e-05*x^3}

# add noise
y <- y.func(x) + res
```

## Split data for train and test


```r
# split dataset for training and test
set.seed(1)
# generate vector of chosen x for train data
inTrain <- sample(seq_along(x), size = train.percent*n.all)

# train data set
x.train <- x[inTrain]
y.train <- y[inTrain]

# test data set
x.test <- x[-inTrain]
y.test <- y[-inTrain]
```

## Diagram of the given function and generated datasets

```r
# lines of generated data for plot
x.line <- seq(x.min, x.max, length = n.all)
y.line <- y.func(x.line)

# PLOT
# generate plot by train data
par(mar = c(4, 4, 1, 1)) # reduce margins (optional)
plot(x.train, y.train,
     main = 'Generated data and original function',
     col = grey(0.2), bg = grey(0.2), pch = 21,
     xlab = 'X', ylab = 'Y', 
     xlim = c(x.min, x.max),
     ylim = c(min(y), max(y)), 
     cex = 1.2, cex.lab = 1.2, cex.axis = 1.2)

# add points of test data
points(x.test, y.test, col = 'red', bg = 'red', pch = 21)

# add the given function
lines(x.line, y.line, lwd = 2, lty = 2)

# add legend
legend('topleft', legend = c('train', 'test', 'f(X)'),
       pch = c(16, 16, NA), 
       col = c(grey(0.2), 'red', 'black'),  
       lty = c(0, 0, 2), lwd = c(1, 1, 2), cex = 1.2)
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-68-1.png" width="672" />

## Build a model using splines
We will compair sevaral models with degree of freedoms (df) from 2 to 40, where 2 correspond to a linear model.


```r
max.df <- 40                       # max degree of freedom (df)
# 
tbl <- data.frame(df = 2:max.df)   # data frame for writing errors
tbl$MSE.train <- 0                 # column 1: errors of train data
tbl$MSE.test <- 0                  # сcolumn 2: errors of test data

# generate models using for cycle
for (i in 2:max.df) {
    mod <- smooth.spline(x = x.train, y = y.train, df = i)
    
    # predicted values for train and test data using built model
    y.model.train <- predict(mod, data.frame(x = x.train))$y[, 1]
    y.model.test <- predict(mod, data.frame(x = x.test))$y[, 1]
    
    # MSE errors for train and test data
    MSE <- c(sum((y.train - y.model.train)^2) / length(x.train),
             sum((y.test - y.model.test)^2) / length(x.test))
    
    # write errors to the previously created data frame
    tbl[tbl$df == i, c('MSE.train', 'MSE.test')] <- MSE
}

# view first rows of the table
head(tbl, 4)
```

```
##   df MSE.train  MSE.test
## 1  2 3.6484333 3.3336892
## 2  3 1.5185881 1.1532857
## 3  4 0.8999800 0.8874002
## 4  5 0.7477105 0.9483290
```

## Diagram of MSE for train and test data

```r
# plot MSE from our table
plot(x = tbl$df, y = tbl$MSE.test,
     main = "Changes of MSE from degrees of freedom",
     type = 'l', col = 'red', lwd = 2,
     xlab = 'spline degree of freedom', ylab = 'MSE',
     ylim = c(min(tbl$MSE.train, tbl$MSE.test), 
              max(tbl$MSE.train, tbl$MSE.test)),
     cex = 1.2, cex.lab = 1.2, cex.axis = 1.2)

# add 
points(x = tbl$df, y = tbl$MSE.test,
       pch = 21, col = 'red', bg = 'red')
lines(x = tbl$df, y = tbl$MSE.train, col = grey(0.3), lwd = 2)
# minimal MSE
abline(h = res.sd, lty = 2, col = grey(0.4), lwd = 2)

# add legend
legend('topright', legend = c('train', 'test'),
       pch = c(NA, 16), 
       col = c(grey(0.2), 'red'),  
       lty = c(1, 1), lwd = c(2, 2), cex = 1.2)

# df of minimal MSE for test data
min.MSE.test <- min(tbl$MSE.test)
df.min.MSE.test <- tbl[tbl$MSE.test == min.MSE.test, 'df']

# optimal df for precise model and maximal simplicity
df.my.MSE.test <- 6
my.MSE.test <- tbl[tbl$df == df.my.MSE.test, 'MSE.test']

# show the optimal solution
abline(v = df.my.MSE.test, 
       lty = 2, lwd = 2)
points(x = df.my.MSE.test, y = my.MSE.test, 
       pch = 15, col = 'blue')
mtext(df.my.MSE.test, 
      side = 1, line = -1, at = df.my.MSE.test, col = 'blue', cex = 1.2)
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-70-1.png" width="672" />

## Build optimal model and plot for the model

```r
mod.MSE.test <- smooth.spline(x = x.train, y = y.train, df = df.my.MSE.test)

# predict data for 250 x's to get smoothed curve
x.model.plot <- seq(x.min, x.max, length = 250)
y.model.plot <- predict(mod.MSE.test, data.frame(x = x.model.plot))$y[, 1]

# plot train data
par(mar = c(4, 4, 1, 1))
plot(x.train, y.train,
     main = "Initial data and the best fit model",
     col = grey(0.2), bg = grey(0.2), pch = 21,
     xlab = 'X', ylab = 'Y', 
     xlim = c(x.min, x.max),
     ylim = c(min(y), max(y)), 
     cex = 1.2, cex.lab = 1.2, cex.axis = 1.2)

# add test data
points(x.test, y.test, col = 'red', bg = 'red', pch = 21)

# function
lines(x.line, y.line,lwd = 2, lty = 2)

# add model
lines(x.model.plot, y.model.plot, lwd = 2, col = 'blue')

# legend
legend('topleft', legend = c('train', 'test', 'f(X)', 'model'),
       pch = c(16, 16, NA, NA), 
       col = c(grey(0.2), 'red', 'black', 'blue'),  
       lty = c(0, 0, 2, 1), lwd = c(1, 1, 2, 2), cex = 1.2)
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-71-1.png" width="672" />

## Bibliograpy
[An Introduction to Statistical Learning by Gareth James](http://faculty.marshall.usc.edu/gareth-james/)

<!--chapter:end:27_spline_model.Rmd-->

# Logistic Regression

The **logistic model** (or **logit model**) is a statistical model with input (independent variable) 
a continuous variable and output (dependent variable) a binary variable (discret choice, e.g. yes/no or 1/0).

## Confusion matrix  
A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known.  

| n=165       | Predicted: NO | Predicted: Yes-|
|-------------|---------------|----------------|
| Actual: No  | TN = 50       | FP = 10        |
| Actual: Yes | FN = 5        | TP = 100       |

TN - true negatives  
TP - true positives  
FN - false negatives  
FP - false posistives  

**Accuracy** - Overall, how often is the classifier correct?  
(TP+TN)/total = (100+50)/165 = 0.91

**Misclassification Rate** - Overall, how often is it wrong?  
(FP+FN)/total = (10+5)/165 = 0.09
equivalent to 1 - Accuracy
also known as "Error Rate"

**True Positive Rate**: When it's actually yes, how often does it predict yes?
TP/actual yes = 100/105 = 0.95
also known as "Sensitivity" or "Recall"

**False Positive Rate**: When it's actually no, how often does it predict yes?
FP/actual no = 10/60 = 0.17

**True Negative Rate**: When it's actually no, how often does it predict no?
TN/actual no = 50/60 = 0.83
equivalent to 1 minus False Positive Rate
also known as "Specificity"

**Precision**: When it predicts yes, how often is it correct?
TP/predicted yes = 100/110 = 0.91

**Prevalence**: How often does the yes condition actually occur in our sample?
actual yes/total = 105/165 = 0.64

**Null Error Rate**: This is how often you would be wrong if you always predicted the majority class. (In our example, the null error rate would be 60/165=0.36 because if you always predicted yes, you would only be wrong for the 60 "no" cases.) This can be a useful baseline metric to compare your classifier against. However, the best classifier for a particular application will sometimes have a higher error rate than the null error rate, as demonstrated by the Accuracy Paradox.  
**Cohen's Kappa**: This is essentially a measure of how well the classifier performed as compared to how well it would have performed simply by chance. In other words, a model will have a high Kappa score if there is a big difference between the accuracy and the null error rate. (More details about Cohen's Kappa.)  
**F Score**: This is a weighted average of the true positive rate (recall) and precision. (More details about the F Score.)  
**ROC Curve**: This is a commonly used graph that summarizes the performance of a classifier over all possible thresholds. It is generated by plotting the True Positive Rate (y-axis) against the False Positive Rate (x-axis) as you vary the threshold for assigning observations to a given class. (More details about ROC Curves.) 


**Example**:  
A group of 20 students spend between 0 and 6 hours studying for an exam. How does the number of hours spent studying affect the probability that the student will pass the exam?  


```r
hours <- c(0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5)
pass <- c(0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1)
model = glm(pass ~ hours, family = binomial)
newdat <- data.frame(hours=seq(min(hours), max(hours),len=100))
newdat$pass = predict(model, newdata=newdat, type="response")

# plot
plot(pass ~ hours)
lines(pass ~ hours, newdat, col="red")
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-72-1.png" width="672" />

```r
# data
data <- data.frame(hours=c(0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5,
                         2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5),
                 pass=c(0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1),
                 pass.predic = rep(NA, 20), # slot for predicted pass
                 pass.logit = rep(NA, 20))  # slot logit prediction
                 

# model
model <- glm(data$pass ~ data$hours, family = binomial)

# predict values of logit function
data$pass.logit <- predict(model, newdata=data, type='response')
# predict pass for threshold = 0.5
data$pass.predic <- ifelse(data$pass.logit > 0.5, 1, 0)

# Confusion matrix
library(caret)
caret::confusionMatrix(data =      factor(data$pass.predic),
                       reference = factor(data$pass))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction 0 1
##          0 8 2
##          1 2 8
##                                           
##                Accuracy : 0.8             
##                  95% CI : (0.5634, 0.9427)
##     No Information Rate : 0.5             
##     P-Value [Acc > NIR] : 0.005909        
##                                           
##                   Kappa : 0.6             
##                                           
##  Mcnemar's Test P-Value : 1.000000        
##                                           
##             Sensitivity : 0.8             
##             Specificity : 0.8             
##          Pos Pred Value : 0.8             
##          Neg Pred Value : 0.8             
##              Prevalence : 0.5             
##          Detection Rate : 0.4             
##    Detection Prevalence : 0.5             
##       Balanced Accuracy : 0.8             
##                                           
##        'Positive' Class : 0               
## 
```



```r
p <- seq(0,1, by=0.05)
data <- data.frame(probability=p, odds=p/(1-p))
head(data)
```

```
##   probability       odds
## 1        0.00 0.00000000
## 2        0.05 0.05263158
## 3        0.10 0.11111111
## 4        0.15 0.17647059
## 5        0.20 0.25000000
## 6        0.25 0.33333333
```

```r
plot(data$odds~data$probability, type='o', pch=19, xlab='Probability', ylab='Odds')
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-73-1.png" width="672" />

```r
plot(log(data$odds)~data$odds, type='o', pch=19, xlab='Odds', ylab='log(odds)')
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-73-2.png" width="672" />

```r
plot(data$probability~log(data$odds), type='o', pch=19, xlab='log(odds)', ylab='Probability')
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-73-3.png" width="672" />

## Next part


```r
library(data.table)
df <- fread('https://raw.githubusercontent.com/suvarzz/data/master/data_classification.csv', header=T, sep=",")
head(df)

plot(df[pass==1][,!3], col='red')
points(df[pass==0][,!3], col='blue')

model.logit <- glm(pass ~ studied + slept, data = df, family = 'binomial')
summary(model.logit)
p.lda <- predict(model.logit, df, type = 'response')
df$predicted <- ifelse(p.lda > 0.5, 1, 0)
head(df)

a=-coef(model.logit)[1]/coef(model.logit)[2],
b=-coef(model.logit)[1]/coef(model.logit)[3])

b0 = coef(model.logit)[1]
b1 = mymodel$coefficients[[2]]
b2 = mymodel$coefficients[[3]]
z = b0 + (b1 * 1) + (b2 * 4)
p = 1 / (1 + exp(-z))

if p=0.5 => z = 0 => b0 + b1*x + b2*y => 
  
segments(0,10.87,9.26,0)
slept=(3.77-0.474*studied)/0.338
(0, 3.77/0.474) = (9.2, 0)
(3.77/0.474,0) = (0, 10.87)

segments(9.2,0, 0,10.87, lwd=2)
```

## NExt part

```r
# Example of logistic regression
# Source: 17 - Анализ данных в R. Логистическая регрессия by Anatoliy Karpov

# Read data set train.csv:
# Statistics of students in a school
# gender - male/femail
# read, write, math - points for subjects
# hon - if honorary degree Y/N

# FIX combine train and test into one csv file. Split train and test inside this script
setwd("~/RData")
df <- read.csv("train.csv", sep=";")

# Visual inspection of the dataset
head(df)
str(df)
View(df)

# N-not the best mark, Y-the best mark
library(ggplot2)
ggplot(df, aes(read,math,col=gender))+geom_point()+facet_grid(.~hon)+
    theme(axis.text=element_text(size=25), axis.title=element_text(size=25, face='bold'))

# Apply logistic regression
# How hon depends on different variables: read, math, gender
fit <- glm(hon ~ read + math + gender, df, family = "binomial")
summary(fit)

# Meanings of coefficients:
# read-estimate: 0.06677 - if female, math is fixed, if read change to 1, then ln(odds) will be changed to 0.06677

# Get data from fit
exp(fit$coefficients)

# Predict model - ln(odds)
head(predict(object=fit))

# Predict model - return probability to get the best mark for every person
head(predict(object = fit, type = "response"))

# Add probabilities to get the best mark for every person in df
df$prob <- predict(object = fit, type = "response")

df

# Part 2
# ROC-curve of predicted model
library(ROCR)
# Predicted values and real values
pred_fit <- prediction(df$prob, df$hon)

# Calculate tpr - true positive rate and fpr - false positive rate
perf_fit <- performance(pred_fit, "tpr", "fpr")

# plot ROC-curve
plot(perf_fit, colorize=T, print.cutoffs.at = seq(0,1, by=0.1))

# Area under the curve: 0.87
auc <- performance(pred_fit, measure = "auc")
str(auc)

# How to detect the border and make a decision if student will get honorary degree
# Specificity - how good we can predict negative results
perf3 <- performance(pred_fit, x.measure = "cutoff", measure = "spec")
# Sencitivity - how good we can predict positive results
perf4 <- performance(pred_fit, x.measure = "cutoff", measure = "sens")
# Общая интенсивность классификатора
perf5 <- performance(pred_fit, x.measure = "cutoff", measure = "acc")

plot(perf3, col = 'red', lwd = 2)
plot(add=T, perf4, col = "green", lwd = 2)
plot(add=T, perf5, lwd = 2)

legend(x = 0.6, y = 0.3, c("spec", "sens", "accur"), lty=1, col=c("red", "green", "black"),
       bty='n', cex=1, lwd=2)
abline(v=0.255, lwd=2)
# The border is the intersection of all three curves

# Add column with prediced values Y/N
df$pred_resp <- factor(ifelse(df$prob > 0.255, 1, 0), labels=c("N", "Y"))

# 1 if prediction is correct, 0 if not correct
df$correct <- ifelse(df$pred_resp == df$hon, 1, 0)
df

# blue - correct classified, red - incorrect classified
# it is more difficult to predict positive result
ggplot(df, aes(prob, fill = factor(correct)))+
    geom_dotplot()+
    theme(axis.text=element_text(size=25),
          axis.title=element_text(size=25, face="bold"))

# Percent of positive predictions
mean(df$correct)

# Part 3 - Prediction using test data
test_df <- read.csv("test.csv", sep=";")
test_df

# Predict honorary members
test_df$prob_predict <- predict(fit, newdata=test_df, type="response")
test_df$pred_resp <- factor(ifelse(test_df$prob_predict > 0.255, 1, 0), labels=c("N", "Y"))
test_df
```

<!--chapter:end:28_logistic_regression.Rmd-->

# Clustering

5 classes of clustering methods:  
1. **Partitioning methods** - split into k-groups (k-means, k-dedoids (PAM), CLARA)  
2. **Hierarchical clustering**  
3. **Fuzzy clustering**  
4. **Density-based clustering**  
5. **Model-based clustering**  


```r
bv <- read.table("./DATA/beverage.csv", header=T, sep=";")  
head(bv)  
# no needs to normalize because all data is binary (0,1)

# Hierarchical clustering
# dist - calculate distances
# hclust - hierarchical clustering

clust.bv <- hclust(dist(bv[,2:9]), "ward.D")
clust.bv

# Plot clusters
plot(clust.bv)
plot(clust.bv, hang = -1)
rect.hclust(clust.bv, k=3, border="red") 

# Group data by clusters
groups <- cutree(clust.bv, k=3)
groups

# Percentage in broups by drinking different beverages
colMeans(bv[groups==1, 2:9])*100
colMeans(bv[groups==2, 2:9])*100
colMeans(bv[groups==3, 2:9])*100

# Interpretation
# 1. People who does not have specific preference
# 2. People who prefers cola and pepsi
# 3. Not clear (others)

# atributes of cluster analysis
names(clust.bv)

# chronic of combining
clust.bv$merge
clust.bv[1]

clust.bv$height
clust.bv$order
clust.bv$labels
clust.bv$method
clust.bv$call
clust.bv$dist.method

# Detect the best choice for number of cluster by elbow-plot
plot(1:33, clust.bv$height, type="l")

### Task. Analyse data and find groups of people
# Scores (0,10) of 10 tests for candidates to get a job.
# 1. Memorizing numbers
# 2. Math task
# 3. Solving tasks in dialoge
# 4. Algorithms
# 5. Self confidence
# 6. Work in group
# 7. Find solution
# 8. Collaboration
# 9. Acceptance by others

setwd("~/DataAnalysis")
job <- read.table("DATA/assess.dat", header=T, sep="\t")
job

# Clustering
clust.job <- hclust(dist(job[,3:ncol(job)]), "ward.D")
# no needs to normalize, because all numbers have the same min, max
plot(clust.job)  # visual number of clusters is 4

# Group data by clusters
groups <- cutree(clust.job, k=4)
groups
colMeans(job[groups==1, 3:12])*100

### Find clusters using k-means method

setwd("~/DataAnalysis")
bv <- read.table("DATA/beverage.csv", header=T, sep=";")
bv
dim(bv)
names(bv)

# k-means clustering, with initial 3 clusters
# nstart = x - run x times with different initial clusters
summ.1 = kmeans(bv[,2:9], 3, iter.max = 100)
names(summ.1)

# Objects by clusters
summ.1$cluster

# Centers of clusters
summ.1$centers
# 2 digits after point
options(digits=2)

t(summ.1$centers)
options(digits=7)

# Square summs
summ.1$withinss

# Summ of elements of vector
summ.1$tot.withinss

# sum(33*(apply(bv[,2:9], 2, sd))^2)
summ.1$totss
summ.1$tot.betweenss

# Size of clusters
summ.1$size

# Elbow plot to detect optimal number of clusters
wss <- (nrow(bv[,2:9])-1)*sum(apply(bv[,2:9],2,var))
for (i in 2:15) { wss[i] <- kmeans(bv[,2:9],
                centers=i)$tot.withinss }
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")

# We can see that diagram is rough. This is because clusters are not allways optimal
# To improve situation, we have to run many initiall start coordinates and choose the best
# option (add nstart=500):
wss <- (nrow(bv[,2:9])-1)*sum(apply(bv[,2:9],2,var))
for (i in 2:15) { wss[i] <- kmeans(bv[,2:9],
                                   centers=i, nstart=500)$tot.withinss }
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")
# Warnings means that iterations were not finished for some cases.

# Let's compair results for 3 and 4 clusters
summ.1 = kmeans(bv[,2:9], 3, iter.max=100)
summ.2 = kmeans(bv[,2:9], 4, iter.max=100)

# Compair clusters. How many elements in each cluster
# We can see how elements move if we take more clusters
table(summ.1$cluster, summ.2$cluster)

# Multidimentional scaling
# Project multidimentional data to 2d
bv.dist <- dist(bv[,2:9])
bv.mds <- cmdscale(bv.dist)
plot(bv.mds, col = summ.1$cluster, xlab="Index", ylab="")

# Detect optimal number of clusters
install.packages("NbClust")
library("NbClust")
Best <- NbClust(bv[,2:9],              # data 
                distance="euclidean",  # distance method
                min.nc=2,              # min number of clusters
                max.nc=8,             # max number of clusters
                method="ward.D",       # ward methodes 
                index = "alllong" )    # choose indices
```
## Next part


```r
library(cluster)
library(factoextra)


Distances:
stats::dist()
factoextra::get_dist()   # compute a distance matrix between the rows of a data matrix
factoextra::fviz_dist()  # visualize distance matrix
cluster::daisy()         # handle both numeric and not numeric (nominal, ordinal,...) data types

d <- factoextra::get_dist(USArrests, stand = TRUE, method = 'pearson')
factoextra::fviz_dist(d, gradient = list(low='blue', mid='white', high='red'))

#####
library(tidyverse)
library(cluster)
library(factoextra)

data <- USArrests %>% na.omit() %>% scale()
data
factoextra::fviz_nbclust(data, kmeans, method = 'gap_stat')


km.res <- kmeans(data, 3, nstart = 25)
factoextra::fviz_cluster(km.res, data = data,
                         ellipse.type = 'convex',
                         palette = 'jco',
                         repel = TRUE,
                         ggtheme = theme_minimal())

# PAM clustering
pam.res <- cluster::pam(data, 4)
factoextra::fviz_cluster(pam.res)

# CLARA clustering
clara.res <- clara(df, 2, samples = 50, pamLike = TRUE)
clara.res

dd <- cbind(df, cluster = clara.res$cluster)

# Medoids
clara.res$medoids

# Clustering
head(clara.res$clustering,10)
```

## Example

```r
library(datasets)
head(iris)
```

```
##   Sepal.Length Sepal.Width Petal.Length Petal.Width Species id
## 1          5.1         3.5          1.4         0.2  setosa  1
## 2          4.9         3.0          1.4         0.2  setosa  2
## 3          4.7         3.2          1.3         0.2  setosa  3
## 4          4.6         3.1          1.5         0.2  setosa  4
## 5          5.0         3.6          1.4         0.2  setosa  5
## 6          5.4         3.9          1.7         0.4  setosa  6
```

```r
# Plot Petal.Length ~ Petal.Width data
plot(iris$Petal.Length ~ iris$Petal.Width)
```

<img src="bookdown-demo_files/figure-html/setup-1.png" width="672" />

```r
set.seed(20)

# Find number of clusters using wss
wss <- (nrow(iris[, 3:4])-1)*sum(apply(iris[, 3:4],2,var))
for (i in 2:15) wss[i] <- sum(kmeans(iris[, 3:4], i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")
```

<img src="bookdown-demo_files/figure-html/setup-2.png" width="672" />

```r
#More than 3 clusters give no obvious advantages

# Make k-means with 3 clasters
ncl <- 3
irisCluster <- kmeans(iris[, 3:4], ncl, nstart = 20)
irisCluster
```

```
## K-means clustering with 3 clusters of sizes 48, 50, 52
## 
## Cluster means:
##   Petal.Length Petal.Width
## 1     5.595833    2.037500
## 2     1.462000    0.246000
## 3     4.269231    1.342308
## 
## Clustering vector:
##   [1] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1 3 3 3 3 3 1
##  [85] 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1 1 1 1 1 1 3 1 1 1 1 1 1 1 1 1 1 1 1 3 1 1 1 1 1 1 3 1 1 1 1 1 1 1 1 1 1 1 3 1 1 1 1 1 1 1 1 1 1 1
## 
## Within cluster sum of squares by cluster:
## [1] 16.29167  2.02200 13.05769
##  (between_SS / total_SS =  94.3 %)
## 
## Available components:
## 
## [1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss" "betweenss"    "size"         "iter"         "ifault"
```

```r
# Compair result of clustering with real data (3 species of iris are in analysis)
table(irisCluster$cluster, iris$Species)
```

```
##    
##     setosa versicolor virginica
##   1      0          2        46
##   2     50          0         0
##   3      0         48         4
```

```r
# Plot data
clusters <- split.data.frame(iris, irisCluster$cluster)
xlim <- c(min(iris$Petal.Width), max(iris$Petal.Width))
ylim <- c(min(iris$Petal.Length), max(iris$Petal.Length))
col <- c('red', 'green', 'blue')
plot(0, xlab='Petal width', ylab='Petal length', xlim=xlim, ylim=ylim)
for ( i in 1:ncl ) {
points(clusters[[i]]$Petal.Length ~ clusters[[i]]$Petal.Width, col=col[i], xlim=xlim, ylim=ylim)
}
```

<img src="bookdown-demo_files/figure-html/setup-3.png" width="672" />

## NEXT PART


```r
# K-Nearest Neighbors or KNN is a clustering algorithm
# k is known number of clusters (usually sqrt(N), between 3-10, but may be different)
# samples must be normalized x = (x - min(x))/(max(x)-min(x))

head(iris)
summary(iris)   # detailed view of the data set
str(iris)   # view data types, sample values, categorical values, etc
plot(iris)

#normalization function

min_max_normalizer <- function(x)
{
    num <- x - min(x) 
    denom <- max(x) - min(x)
    return (num/denom)
}

#normalizing iris data set
normalized_iris <- as.data.frame(lapply(iris[1:4], min_max_normalizer))

#viewing normalized data
summary(normalized_iris)

#checking the data constituency
table(iris$Species)

#set seed for randomization
set.seed(1234)

# setting the training-test split to 67% and 33% respectively
random_samples <- sample(2, nrow(iris), replace=TRUE, prob=c(0.67, 0.33))

# training data set
iris.training <- iris[
    random_samples ==1, 1:4] 

#training labels
iris.trainLabels <- iris[
    random_samples ==1, 5]


# test data set
iris.test <- iris[
    random_samples ==2, 1:4]

#testing labels
iris.testLabels <- iris[
    random_samples ==2, 5]

#setting library
library(class)

#executing knn for k=3
iris_model <- knn(train = iris.training, test = iris.test, cl = iris.trainLabels, k=3)

#summary of the model learnt
iris_model
```

<!--chapter:end:30_clustering.Rmd-->

# Learning Vector Quantization

Learning Vector Quantiztion (LVQ) is a supervised classification algorithm for binary and multiclass problems. LVQ is a special case of a neural network.  
LVQ model creates codebook vectors by learning training dataset. Codebook vectors represent class regions. They contain elements that placed around the respective class according to their matching level. If the element matches, it comes closer to the target class, if it does not match, it moves farther from it. With this codebooks, the model classifies new data.
[Here](http://jsalatas.ictpro.gr/implementation-of-competitive-learning-networks-for-weka/) is a nice explanation how it works.

There are several versions of **LVQ** function:  
`lvq1()`, `olvq1()`, `lvq2()`, `lvq3()`, `dlvq()`.    


```r
library(class) # olvq1()
library(caret) # to split data

# generate dataset
df <- iris

id = caret::createDataPartition(df$Species, p = .8, list = F)

train = df[id, ]
test = df[-id, ]

# initialize an LVQ codebook
cb = class::lvqinit(train[1:4], train$Species)

# training set in a codebook.
build.cb = class::olvq1(train[1:4], train$Species, cb)

# classify test set from LVQ Codebook for test data
predict = class::lvqtest(build.cb, test[1:4])

# confusion matrix.
caret::confusionMatrix(test$Species, predict)
```

```
## Confusion Matrix and Statistics
## 
##             Reference
## Prediction   setosa versicolor virginica
##   setosa         10          0         0
##   versicolor      0         10         0
##   virginica       0          1         9
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9667          
##                  95% CI : (0.8278, 0.9992)
##     No Information Rate : 0.3667          
##     P-Value [Acc > NIR] : 4.476e-12       
##                                           
##                   Kappa : 0.95            
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: setosa Class: versicolor Class: virginica
## Sensitivity                 1.0000            0.9091           1.0000
## Specificity                 1.0000            1.0000           0.9524
## Pos Pred Value              1.0000            1.0000           0.9000
## Neg Pred Value              1.0000            0.9500           1.0000
## Prevalence                  0.3333            0.3667           0.3000
## Detection Rate              0.3333            0.3333           0.3000
## Detection Prevalence        0.3333            0.3333           0.3333
## Balanced Accuracy           1.0000            0.9545           0.9762
```

<!--chapter:end:41_learning_vector_quantization.Rmd-->

# Bayesian inference

**Bayesian data analysis** the use of Bayesian inference to learn from data.  

**Bayesian inference** is a method for figuring out unobservable quantities 
given known facts that uses probability to describe the uncertainty over what 
the values of the unknown quantities could be.  

**Bayesian inference** is  conditioning on data, in order to learn about parameter values.  

**Bayesian inference** is a method of statistical inference in which Bayes' 
theorem is used to update the probability for a hypothesis as more evidence or 
information becomes available.  

**Bayes' theorem**  
$$P( \theta | D ) = frac{P( D | \theta) \pdot P( \theta)}{\sum{P( D | \theta) \pdot P( \theta ) }}$$

$\theta$ - Parameter values  
$D$ - New data  
$P(D | \theta)$ - Likelihood (the relative) probability of the data given different parameter value.  
$P( \theta )$ - Prior  
$\sum{P( D | \theta) \pdot P( \theta ) }$ - The total sum of the likelihod weighted by the prior.  

- A **prior** is a probability distribution that represents what the model knows before seeing the data.  
- A **posterior** is a probability distribution that represents what the model knows after having seen the data.  

**Bayesian inference components**:  

- Data  
- Bayesian Model:  
+ Generative model (`dbinom`, `dnorm`, `poisson` etc.)  
+ Priors  
- Computational methods:  
+ Rejection sampling  
+ Grid approximation  
+ Markov Chain Monte Carlo (MCMC)  


```r
# Visualization function of Bayesian inference for binary data 
library(ggjoy)
library(ggExtra)
library(ggjoy)
library(tidyverse)

# prop_model function to visualize bayesian posterior distributions
prop_model <- function(data = c(), prior_prop = c(1, 1), n_draws = 10000, show_plot = TRUE) 
{
    data <- as.logical(data)
    proportion_success <- c(0, seq(0, 1, length.out = 100), 1)
    data_indices <- round(seq(0, length(data), length.out = min(length(data) + 
                                                                    1, 20)))
    post_curves <- map_dfr(data_indices, function(i) {
        value <- ifelse(i == 0, "Prior", ifelse(data[i], "Success", "Failure"))
        label <- paste0("n=", i)
        probability <- dbeta(proportion_success, prior_prop[1] + 
                                 sum(data[seq_len(i)]), prior_prop[2] + sum(!data[seq_len(i)]))
        probability <- probability/max(probability)
        tibble(value, label, proportion_success, probability)
    })
    post_curves$label <- fct_rev(factor(post_curves$label, levels = paste0("n=", 
                                                                           data_indices)))
    post_curves$value <- factor(post_curves$value, levels = c("Prior", 
                                                              "Success", "Failure"))
    p <- ggplot(post_curves, aes(x = proportion_success, y = label, 
                                 height = probability, fill = value)) + geom_joy(stat = "identity", 
                                                                                 color = "white", alpha = 1, panel_scaling = TRUE, size = 1) + 
        scale_y_discrete("", expand = c(0.01, 0)) + scale_x_continuous("Underlying proportion of success") + 
        scale_fill_manual(values = hcl(120 * 2:0 + 15, 100, 65), 
                          name = "", drop = FALSE, labels = c("Prior   ", "Success   ", 
                                                              "Failure   ")) + theme_light(base_size = 18) + 
        theme(legend.position = "top")
    if (show_plot) {
        print(p)
    }
    invisible(rbeta(n_draws, prior_prop[1] + sum(data), prior_prop[2] + 
                        sum(!data)))
}
```

## Simple model with one binary parameter


```r
# Generate 50 random binary data with P(1)=0.75
data <- sample(c(1, 0), prob = c(0.75, 0.25), size = 10, replace = TRUE)

# Visualize posteriors
posterior <- prop_model(data)
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-81-1.png" width="672" />

```r
# View posterior
head(posterior)
```

```
## [1] 0.6907880 0.6526425 0.4462395 0.6992505 0.5693596 0.5330336
```

```r
# Center of the posterior
median(posterior)
```

```
## [1] 0.5882531
```

```r
# The credible interval (CI)
quantile(posterior, c(0.05, 0.95))
```

```
##        5%       95% 
## 0.3505923 0.7977435
```

```r
# Probability of successes > 0.3
sum(posterior > 0.3) / length(posterior)
```

```
## [1] 0.977
```

**Generate binomial data**  

```r
p <- 0.42     # probability of success
n <- 100      # number of observations
# Create logical vector, TRUE if numeric value from uniform distribution < p
data <- c()
for (i in 1:n) {
    data[i] <- runif(1, min = 0, max = 1) < p
}
# convert logical to numeric 0/1
data <- as.numeric(data)
data
```

```
##   [1] 0 0 1 0 1 1 0 0 0 0 1 1 0 0 0 0 1 1 0 1 1 1 0 1 1 1 1 0 1 0 0 1 0 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 0 1 0 1 0 1 0 1 0 1 0 0 1 1 0 1 1 0 1 1 0 1 0 0 0 1 1 1 0 0 0 0 0 0 1 1
##  [85] 0 1 0 1 1 0 0 0 1 1 0 0 1 0 0 0
```

```r
# the same vector using rbinom distribution
n = 100      # number of observations
size = 1     # 0 - fail, 1 - success, for 1 trial
p = 0.42     # probability of success
rbinom(n, size, p)
```

```
##   [1] 1 0 1 0 1 0 1 0 0 1 1 0 1 0 0 0 1 1 0 1 0 0 1 0 0 1 0 0 0 1 1 1 0 0 0 1 0 1 0 1 1 0 1 1 1 0 1 1 1 0 1 0 1 0 1 1 0 0 0 0 0 1 0 0 0 1 0 0 1 0 1 1 0 0 0 1 1 1 1 1 0 0 1 1
##  [85] 0 1 0 0 0 0 0 0 1 1 0 0 0 0 0 1
```

## Grid approximation

To get more visitors to your website you are considering paying for an ad to be 
**shown 100 times** on a popular social media site.  
According to the social media site, their ads get **clicked on 10% of the time**. 
How many visitors could your site get?  

We can fix % of clicks as 0.1 but we do not know exactly how many clicks people 
do (uncertain). We suggest the % of clicks is uniformly distributed between 0 and 20.  

Define model:  
$n_{ads}  = 100$
$p_{clicks} ~ Uniform(0.0, 0.2)$
$p_{visitors} ~ Binomial(n_{ads}, p_{clicks})$


```r
n <- 100000          # number of simulations
n_ads_shown <- 100   # number of shown ads

# probability that add will be clicked (assume uniform [0,0.2])
p_clicks <- runif(n, min = 0.0, max = 0.2)  

# MODEL: We assume that the binomial distribution is a reasonable generative model
n_visitors <- rbinom(n = n, 
                     size = n_ads_shown, 
                     prob = p_clicks)

# Visualize the prior model simulation
library(ggplot2)
df <- data.frame(x = n_visitors, y = p_clicks)
p <- ggplot(df, aes(n_visitors, p_clicks)) + geom_point() + theme_classic()
ggExtra::ggMarginal(p, type = "histogram",  bins = 18)
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-83-1.png" width="672" />

**Generate posterior distribution**  

After the first day of showing ads we registered that **13 people** clicked and 
**visited** site when the ad was shown 100 times.  


```r
# Prior distribution binomial model with uniform clicks
prior <- data.frame(p_clicks, n_visitors)

# To get posterior we subset from prior 13 visitors
posterior <- prior[prior$n_visitors == 13, ]
head(posterior)
```

```
##     p_clicks n_visitors
## 6  0.1650342         13
## 11 0.1480911         13
## 20 0.1011115         13
## 23 0.1511866         13
## 40 0.1314564         13
## 51 0.1263088         13
```

```r
# visualize posterior
p <- ggplot(posterior, aes(n_visitors, p_clicks)) + geom_point() + theme_classic()
ggExtra::ggMarginal(p, type = "histogram",  bins = 18)
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-84-1.png" width="672" />

For the next iteration we make **prior from posterior** and include new data for 
subsetting from prior to get posterior.  

```r
# Assign posterior to prior for the next iteration
prior <- posterior

# next iteration posterior ~ f(prior)
n <-  nrow(prior)
n_ads_shown <- 100
prior$n_visitors <- rbinom(n, 
                           size = n_ads_shown, 
                           prob = prior$p_clicks)

# plot of prior distributions
p <- ggplot(prior, aes(n_visitors, p_clicks)) + geom_point() + theme_classic()
ggExtra::ggMarginal(p, type = "histogram",  bins = 18)
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-85-1.png" width="672" />

Use result of the model for prediction:  


```r
# Calculate the probability that you will get 5 or more visitors.
sum(prior$n_visitors >= 5) / length(prior$n_visitors)
```

```
## [1] 0.9863665
```

```r
# Median number of visitors
median(prior$n_visitors)
```

```
## [1] 13
```

**Updated model**
We will change prior distribution with knowledge we learn from people that the 
number of clicks are 5% and sometimes 2 or 8%.
We will use **beta distribution** to setup number of clicks to update prior in 
our model.  


```r
n <- 100000
n_ads_shown <- 100

# Change the prior on proportion_clicks
p_clicks <- rbeta(n, shape1 = 5, shape2 = 95)

# Updated model
n_visitors <- rbinom(n = n,
                     size = n_ads_shown, 
                     prob = p_clicks)

prior <- data.frame(p_clicks, n_visitors)

posterior <- prior[prior$n_visitors == 13, ]

# Plots the prior and the posterior
par(mfcol = c(1, 2))
hist(prior$p_clicks, xlim = c(0, 0.25))
hist(posterior$p_clicks, xlim = c(0, 0.25))
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-87-1.png" width="672" />

We tested for video banner and now we would like to compair distributions for:
- video ads (13 out of 100 clicked)  
- text ads (6 out of a 100 clicked).  


```r
n <- 100000
n_ads_shown <- 100
p_clicks <- runif(n, min = 0.0, max = 0.2)

n_visitors <- rbinom(n, 
                     size = n_ads_shown, 
                     prob = p_clicks)
prior <- data.frame(p_clicks, n_visitors)

# Create the posteriors for video and text ads
posterior_video <- prior[prior$n_visitors == 13, ]
posterior_text <- prior[prior$n_visitors == 6, ]

par(mfrow=c(1,2))
# Visualize the posteriors
hist(posterior_video$p_clicks, xlim = c(0, 0.25))
hist(posterior_text$p_clicks, xlim = c(0, 0.25))
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-88-1.png" width="672" />

```r
# combine data into dataframe
# Make sizes of distrib. the same (4000) to fit to dataframe
posterior <- data.frame(video_prop = posterior_video$p_clicks[1:4000],
                        text_prop = posterior_text$p_clicks[1:4000])

# Calculate the posterior difference: video_prop - text_prop
posterior$prop_diff <- posterior$video_prop - posterior$text_prop

# Calculate the median of prop_diff
median(posterior$prop_diff)
```

```
## [1] 0.06563094
```

```r
# Calculate the proportion
mean(posterior$prop_diff > 0.0)
```

```
## [1] 0.94825
```

```r
# Visualize prop_diff
hist(posterior$prop_diff)
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-88-2.png" width="672" />

**Decision analysis**  

Our analysis indicated that *the video ads are clicked more often*.  
We would like to estimate probable profit *video ads* vs *text ads* if we know:  

- Each visitor spends $2.53 on average on our website.  
- Video ad costs us $0.25 per click.   
- Text ad costs us $0.05 per click.  


```r
visitor_spend <- 2.53
video_cost <- 0.25
text_cost <- 0.05

# Add the column posterior$video_profit
posterior$video_profit <- posterior$video_prop * visitor_spend - video_cost

# Add the column posterior$text_profit
posterior$text_profit <- posterior$text_prop * visitor_spend - text_cost

head(posterior)
```

```
##   video_prop  text_prop  prop_diff video_profit text_profit
## 1  0.1991635 0.03699197 0.16217154   0.25388368  0.04358969
## 2  0.1931784 0.09495372 0.09822467   0.23874133  0.19023292
## 3  0.1316657 0.05753025 0.07413544   0.08311420  0.09555153
## 4  0.1654641 0.06146723 0.10399688   0.16862420  0.10551210
## 5  0.1260397 0.07971197 0.04632772   0.06888041  0.15167129
## 6  0.1152924 0.05278979 0.06250263   0.04168981  0.08355816
```

```r
# Visualize the video_profit and text_profit columns
par(mfrow=c(1,2))
hist(posterior$video_profit)
hist(posterior$text_profit)
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-89-1.png" width="672" />

```r
# Difference between video and text ad profits
posterior$profit_diff <- posterior$video_profit - posterior$text_profit

# Visualize posterior$profit_diff
hist(posterior$profit_diff)

# Calculate a "best guess" for the difference in profits
median(posterior$profit_diff)
```

```
## [1] -0.03395372
```

```r
# Calculate the probability that text ads are better than video ads
mean(posterior$profit_diff < 0)
```

```
## [1] 0.63125
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-89-2.png" width="672" />
Our analysis showed that text ads will bring probably more profit but the result 
is too uncertain and we need more data for making the correct decision.  

**Another model for another case**  

We would like to put up an ad on the site and pay for it per day.  
The site admin promise that we will get 19 clicks per day.  
How many daily clicks should we expect on average?  
We are going to use **Poisson distribution** for our new model which correspond 
to *number of successes in a period of time*.  


```r
n <- 100000
mean_clicks <- runif(n, min = 0, max = 80) # uniform dist between 0 and 80 clicks
# model
n_visitors <- rpois(n = n, mean_clicks)

prior <- data.frame(mean_clicks, n_visitors)

# first day trial showed 13 clicks
posterior <- prior[prior$n_visitors == 13, ]

# visualize prior and posterior
par(mfrow=c(1,2))
hist(prior$mean_clicks)
hist(posterior$mean_clicks)
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-90-1.png" width="672" />

## Grid approximation  

The following example shows how we can build the previous model more effective.  
We calculating the **distribution** using `dbinom` instead of **simulating** 
using `rbinom`.  
We can directly include condition in our model using `dbinom` instead of `rbinom`.  

Instead of using resul of our trial *13 visitors after showing ads 100 times* we 
can condition for all posible number of visitors `seq(0,100)`.  


```r
n_ads_shown <- 100
p_clicks <- 0.1
n_visitors <- seq(0, 100) # instead of 13 we get for all possible n visitors
# model
prob <- dbinom(n_visitors,
               size = n_ads_shown, 
               prob = p_clicks)
head(prob)
```

```
## [1] 0.0000265614 0.0002951267 0.0016231966 0.0058916025 0.0158745955 0.0338658038
```

```r
# Plot the distribution
plot(n_visitors, prob, type = "h")
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-91-1.png" width="672" />

**Calculating a joint distribution**  


```r
n_ads_shown <- 100
p_clicks <- seq(0, 1, by = 0.01)
n_visitors <- seq(0, 100, by = 1)

# Define a grid over all the parameter combinations you need to evaluate
pars <- expand.grid(proportion_clicks = p_clicks,
                    n_visitors = n_visitors)

pars$prior <- dunif(pars$proportion_clicks, min = 0, max = 0.2)

pars$likelihood <- dbinom(pars$n_visitors, 
                          size = n_ads_shown, 
                          prob = pars$proportion_clicks)

### According to Bayes' theorem:
# Combined probability by the rule of probabilities multiplication
pars$probability <- pars$likelihood * pars$prior
# normalize to the total number to get sum of all probabilities eq 1
pars$probability <- pars$probability / sum(pars$probability)
###

# Conditioning on the data for n_visitors == 6  
pars <- pars[pars$n_visitors == 6, ]

# Normalize again to get sum of all probabilities eq 1
pars$probability <- pars$probability / sum(pars$probability)

# Plot the posterior pars$probability
plot(pars$proportion_clicks, pars$probability, type = "h")
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-92-1.png" width="672" />

We can directly condition and change `n_visitors <- 6`. In this case we do not 
need to subset by `pars[pars$n_visitors == 6, ]`. Result will be the same.  

## Model of birth weights using normal distribution

Let's assume that the Normal distribution is a decent model of birth weight data.  


```r
# Assign mu and sigma
m <- 3500   # central value (mean weight)
s <- 600    # deviation

weight_dist <- rnorm(n = 100000, mean = m, sd = s)
hist(weight_dist, 60, xlim = c(0, 6000))
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-93-1.png" width="672" />

We calculating the **distribution** using `dnorm` instead of **simulating** using `rnorm`.  


```r
# Create weight
weight <- seq(0, 6000, by = 100)  # 100 g increment

# Calculate likelihood
likelihood <- dnorm(weight, m, s)

# Plot the distribution of weight
plot(weight, likelihood, type = "h")
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-94-1.png" width="672" />

Here is a small data set with the birth weights of six newborn babies in grams.  
`c(3164, 3362, 4435, 3542, 3578, 4529)`  

Mark: What to do with the data? Should we condition it to get posterior?  

## A Bayesian model of Zombie IQ
Check video 

```r
temp <- c(19, 23, 20, 17, 23)
mu <- seq(8, 30, by = 0.5)
sigma <- seq(0.1, 10, by = 0.3)
pars <- expand.grid(mu = mu, sigma = sigma)

pars$mu_prior    <- dnorm(pars$mu,    mean = 18, sd = 5)
pars$sigma_prior <- dunif(pars$sigma, min = 0, max = 10)

pars$prior <- pars$mu_prior * pars$sigma_prior

for (i in 1:nrow(pars)) {
    likelihoods <- dnorm(temp, pars$mu[i], pars$sigma[i])
    pars$likelihood[i] <- prod(likelihoods)
}

pars$probability <- pars$likelihood * pars$prior
pars$probability <- pars$probability / sum(pars$probability)

sample_indices <- sample(1:nrow(pars), size = 10000, 
                         replace = TRUE, prob = pars$probability)
head(sample_indices)
```

```
## [1] 655 434 839 386 745 471
```

```r
pars_sample <- pars[sample_indices, c("mu", "sigma")]
head(pars_sample)
```

```
##       mu sigma
## 655 20.0   4.3
## 434 22.0   2.8
## 839 22.0   5.5
## 386 20.5   2.5
## 745 20.0   4.9
## 471 18.0   3.1
```

```r
hist(pars_sample$mu, 30)
```

<img src="bookdown-demo_files/figure-html/bi-432-1.png" width="672" />

```r
quantile(pars_sample$mu, c(0.05, 0.95))
```

```
##   5%  95% 
## 17.5 22.5
```

```r
pred_temp <- rnorm(10000, mean = pars_sample$mu, sd = pars_sample$sigma)
hist(pred_temp, 30)
```

<img src="bookdown-demo_files/figure-html/bi-432-2.png" width="672" />

```r
# Probability of 18C
sum(pred_temp >= 18) / length(pred_temp)
```

```
## [1] 0.7385
```
for the model of temperature but use zombies IQs instead. 
Define model:  
$\mu = Normal(mean:18, sd:5)$
$\sigma = Uniform(min:0, max:10)$
$temp = 19,23,...$
How much we can learn about the mean zombie IQ from this data. 
We need to calculate the probability of each parameter combination in pars.  
Use Bayes Theorem to calculate these probabilities and assign them to 
`pars$probability` to complete the model.  


```r
# The IQ of a bunch of zombies
iq <- c(55, 44, 34, 18, 51, 40, 40, 49, 48, 46)

# Defining the parameter grid
pars <- expand.grid(mu = seq(0, 150, length.out = 100), 
                    sigma = seq(0.1, 50, length.out = 100))

# Defining and calculating the prior density for each parameter combination
pars$mu_prior <- dnorm(pars$mu, mean = 100, sd = 100)
pars$sigma_prior <- dunif(pars$sigma, min = 0.1, max = 50)
pars$prior <- pars$mu_prior * pars$sigma_prior

# Calculating the likelihood for each parameter combination
for(i in 1:nrow(pars)) {
  likelihoods <- dnorm(iq, pars$mu[i], pars$sigma[i])
  pars$likelihood[i] <- prod(likelihoods)
}

# Calculate the probability of each parameter combination
pars$probability <- pars$likelihood * pars$prior / sum(pars$likelihood * pars$prior)
```

Calculate new parameters.  


```r
head(pars)
```

```
##         mu sigma    mu_prior sigma_prior        prior likelihood probability
## 1 0.000000   0.1 0.002419707  0.02004008 4.849113e-05          0           0
## 2 1.515152   0.1 0.002456367  0.02004008 4.922578e-05          0           0
## 3 3.030303   0.1 0.002493009  0.02004008 4.996010e-05          0           0
## 4 4.545455   0.1 0.002529617  0.02004008 5.069373e-05          0           0
## 5 6.060606   0.1 0.002566174  0.02004008 5.142633e-05          0           0
## 6 7.575758   0.1 0.002602661  0.02004008 5.215754e-05          0           0
```

```r
sample_indices <- sample(nrow(pars), size = 10000,
                         replace = TRUE, prob = pars$probability)
head(sample_indices)
```

```
## [1] 3434 2627 5124 2934 2427 1528
```

```r
# Sample from pars to calculate some new measures
pars_sample <- pars[sample_indices, c("mu", "sigma")]

# Calculate quantiles
quantile(pars_sample$mu, c(0.025, 0.5, 0.975))
```

```
##     2.5%      50%    97.5% 
## 34.84848 42.42424 51.51515
```

```r
head(pars_sample)
```

```
##            mu     sigma
## 3434 50.00000 17.237374
## 2627 39.39394 13.205051
## 5124 34.84848 25.806061
## 2934 50.00000 14.717172
## 2427 39.39394 12.196970
## 1528 40.90909  7.660606
```

```r
pred_iq <- rnorm(10000, 
                 mean = pars_sample$mu, 
                 sd = pars_sample$sigma)

# Calculate the probability that zombi has IQ > 60
sum(pred_iq >= 60) / length(pred_iq)
```

```
## [1] 0.0867
```

```r
par(mfrow=c(1,2))
# Visualize the mean IQ
hist(pars_sample$mu, 100)
# Visualize pred_iq
hist(pred_iq)
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-95-1.png" width="672" />

## The BEST models 
The t-test is a classical statistical procedure used to compare the 
means of two data sets.  

In 2013 John Kruschke developed a souped-up Bayesian version of the 
t-test he named **BEST** (standing for **B**ayesian **E**stimation **S**upersedes 
the **t**-test).  

We would like to compair IQ of two groups of 10 patiens on different diets (a and b).  


```r
# The IQ of patients.
iq_a <- c(55, 44, 34, 18, 51, 40, 40, 49, 48, 46)
iq_b <- c(44, 52, 42, 66, 53, 42, 55, 57, 56, 51)

# Calculate the mean difference in IQ between the two groups
mean(iq_b) - mean(iq_a)
```

```
## [1] 9.3
```

```r
require(BEST)

# Fit the BEST model to the data from both groups
require(BEST)
best_posterior <- BEST::BESTmcmc(iq_b, iq_a)
```

```r
# Plot the model result
plot(best_posterior)
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-98-1.png" width="672" />

The Bayesian model behind BEST assumes that the generative model for 
the data is a **t-distribution**; a more flexible distribution than the normal 
distribution as it assumes that data points might be outliers to some degree. 
This makes BEST's estimate of the mean difference robust to outliers in the data.  

**Sources**  

[Fundamentals of Bayesian Data Analysis in R on Datacamp](https://learn.datacamp.com/courses/fundamentals-of-bayesian-data-analysis-in-r)

<!--chapter:end:42_bayesian_inference.Rmd-->

# Naive Bayes classifiers

**Naive Bayes classifiers** are a family of simple "probabilistic classifiers" 
based on applying Bayes' theorem with strong (naïve) independence assumptions 
between the features.  

They are among the simplest Bayesian network models, but coupled with kernel 
density estimation, they can achieve higher accuracy levels.  

Naïve Bayes classifiers are highly scalable, requiring a number of parameters 
linear in the number of variables (features/predictors) in a learning problem. 
Maximum-likelihood training can be done by evaluating a closed-form expression, 
which takes linear time, rather than by expensive iterative approximation as 
used for many other types of classifiers.  

$P(c|x) = \frac{P(x|c)(P(c))}{P(x)}$, where  
$P(c|x)$ - posteriour probability  
$P(x|c)$ - Likelihood  
$P(c)$ - Class Prior Probbility  
$P(x)$ - Predictor Prior Probability  

<!--chapter:end:43_naive_bayes.Rmd-->

# Markov Chain Monte Carlo (MCMC)

We have a seq of conditional probabilities:  
We know conditional probabilities for the weather tomorrow (t+1) depending of the weather today (t). R for raily day and S for sunny day.  
$P(S_{t+1} | R_t) = 0.5$  
$P(R_{t+1} | R_t) = 0.5$  
$P(R_{t+1} | S_t) = 0.1$  
$P(S_{t+1} | S_t) = 0.9$  


What is the probability of sunny (S) or rainy (R) day?  
Simulate data using conditional probabilities:  
Markov Chain 1:  
S-S-R-R-S-S-S-S-R-R-R-S-S-S

Markov Chain 2:  
R-S-S-S-S-S-R-S-S-S-S-S-R-R  

After simulation we calculate probabilities of sunny (S) and rainy (R) days:  
P(S) = 0.833
P(R) = 0.167

<!--chapter:end:44_monte_carlo.Rmd-->

# Simple Markov process


Here, we will consider a simple example of Markov process with implementation in R.  
The following example is taken from [Bodo Winter website](http://www.bodowinter.com).  

A **Markov process** is characterized by (1) **a finite set of states** and (2) **fixed transition probabilities between the states**.  

Let's consider an example. Assume you have a classroom, with students who could be either in the state **alert** or in the state **bored**. And	 then, at any given time point, there's a certain probability of an alert student becoming bored (say 0.2), and there's a probability of a  bored student becoming alert (say 0.25).  

Let's say there are 20 alert and 80 bored students in a particular class. This is your initial condition at time point	$t$. Given the	transition probabilities above,	what's the number of alert and bored students at the next point in time, $t+1$?  	
Multiply 20 by 0.2 (=4) and these will be the alert students that	turn bored.	 
And	then multiply 80 by 0.25 (=20) and these will be the bored students that turn alert.  
So, at $t+1$, there's going to be 20-4+20 alert students. And there's going to be 80+4-20 bored students. Before, 80% of the students were bored and now, only 64% of the students are bored. Conversely, 36% are alert.  

A handy way of representing this Markov process is by defining a transition probability matrix:  

|           | A   | B    |
|-----------|-----|------|
| A$_{t+1}$ | 0.8 | 0.25 |
| B$_{t+1}$ | 0.2 | 0.75 |

What this matrix says is: A proportion of 0.8 of the people who are in state A (alert) will also be at state A at time point $t+1$. And, a proportion of 0.25 of the people who are in state B (bored) will switch to alert at t+1. This is what the first row says. The next row is simply one minus the probabilities of the first row, because probabilities (or proportions) have to add up to 1. Now think about multiplying this matrix with the initial proportions of alert and bored students that we had above. 0.8 are bored and 0.2 are alert. In linear algebra this would look the following way:  

<div>
$$
\begin{bmatrix}
 0.8 & 0.25 \\
 0.2 & 0.75
\end{bmatrix}\times\begin{bmatrix}
 0.2 \\
 0.8
\end{bmatrix} = \begin{bmatrix}
 0.8\times0.2 + 0.25\times0.8 \\
 0.2\times0.2 + 0.75\times0.8
\end{bmatrix} = \begin{bmatrix}
0.36 \\
0.64
\end{bmatrix}
$$
</div>

The results of these calculations are exactly the proportions that we saw above: 36% alert student and 64% bored students.  

Now, you might ask yourself: What happens if this process continues? What happens at $t+2$, $t+3$ etc.? Will it be the case that at one point there are no bored students any more? Let's simulate this in R and find out! Let's call this **tpm** for **transition probability matrix**:  


```r
tpm = matrix(c(0.8,0.25, 0.2,0.75), nrow=2, byrow=TRUE)
colnames(tpm) = c('A','B')
rownames(tpm) = c('At+1', 'Bt+1')
tpm
```

```
##        A    B
## At+1 0.8 0.25
## Bt+1 0.2 0.75
```

Again this matrix shows that 0.8 students who were in state A at time point t will still be in state A at $t+1$. And 0.25 students who were in state B at time point t will be in state A at $t+1$. The second row has a similar interpretation for alert and bored students becoming bored at $t+1$. Remember that Markov processes assume fixed transition probabilities. This means that in the simulation that we'll be doing, we leave the transition probability matrix unchanged. However, we will define a vector of the actual proportions – and these are allowed to change. In time, we expect more and more students to become alert, because the transition probability from B to A (which, to remind you, was 0.25) is higher than from A to B (which was 0.2).  

Let's start our simulation by setting the initial condition as 0.1 students are alert and 0.9 students are bored and define a matrix called **sm** (short for **student matrix**):  

```r
sm = as.matrix(c(0.1, 0.9))
rownames(sm)= c('A', 'B')
sm
```

```
##   [,1]
## A  0.1
## B  0.9
```
Now let's repeat by looping:


```r
for(i in 1:10){
    sm = tpm %*% sm
    }
```
Here, we're looping 10 times and on each iteration, we multiply the matrix **tpm** with the student matrix **sm**. We take this result and store it in **sm**. This means that at the next iteration, our fixed **transition probability matrix** will be multiplied by a different student matrix, allowing for the proportions to slowly change over time.  
R operator '%*%' is used for matrix multiplication

Outcome of our ten loop iterations:

```r
sm
```

```
##           [,1]
## At+1 0.5544017
## Bt+1 0.4455983
```

So, after 10 iterations of the Markov process, we now have about 55% alert students and 45% bored ones. What is interesting to me is that even though 80% of the people who are alert at one time point remain alert at the next time point, the process only converged on 55% alert and 45% bored after 10 iterations.

Let's reset our initial condition to (0.1 alert and 0.9 bored students) and run a thousand iterations. 

```r
for(i in 1:1000){
    sm = tpm %*% sm
    }
sm
```

```
##           [,1]
## At+1 0.5555556
## Bt+1 0.4444444
```

A 1000 iterations, and we seem to be zoning in onto ~55% and ~44%. This phenomenon is called **Markov convergence**. You could run even more iterations, and your outcome would get closer and closer to 0.5555 (to infinity). So, the model converges on an equilibrium. However, this is not a fixed equilibrium. It's not the case that the Markov process comes to a hold or that nobody changes states between alertness and boredness any more. The equilibrium that we're dealing with here is a statistical equilibrium, where the proportions of alert and bored students remain the same. but there still is constant change (at each time step, 0.2 alert students become bored and 0.25 bored students become alert). Markov models always converge to a statistical equilibrium if the conditions (1) and (2) above are met, and if you can get from any state within your Markov model to any other state (in the case of just two states, that clearly is the case). What's so cool about this is that it is, at first sight, fairly counterintuitive.

At least when I thought about the transition probabilities for the first time, I somehow expected all students to become alert but as we saw, that's not the case. Moreover, this process is not sensitive to initial conditions. That means that when you start with any proportion of alert or bored students (even extreme ones such as 0.0001 alert students), the process will reach the statistical equilibrium – albeit sometimes a little faster or slower. You can play around with different values for the **sm** object to explore this property of Markov convergence. Another interesting thing is that the process is impervious to intervention: Say, you introduced something that made more students alert – the Markov model would quickly get back to equilibrium. So Markov processes are essentially ahistorical processes: history doesn't matter. Even with extreme initial conditions or extreme interventions, the process quickly converges to the equilibrium defined by the transition probabilities. The only way to persistently change the system is to change the transition probabilities. Finally, what I find so cool about Markov processes is their computational simplicity.  

### Sources
[Bodo Winter website](http://www.bodowinter.com)

<!--chapter:end:45_simple_markov_process.Rmd-->

# Tree-based models

Classification and regression trees (CART) are a non-parametric decision tree learning technique that produces either classification or regression trees, depending on whether the dependent variable is categorical or numeric, respectively.  

CART is both a generic term to describe tree algorithms and also a specific name for Breiman’s original algorithm for constructing classification and regression trees.  

* **Decision Tree**: A tree-shaped graph or model of decisions used to determine a course of action or show a statistical probability.  
* **Classification Tree**: A decision tree that performs classification (predicts a categorical response).  
* **Regression Tree**: A decision tree that performs regression (predicts a numeric response).  
* **Split Point**: A split point occurs at each node of the tree where a decision is made (e.g. x > 7 vs. x ≤ 7).  
* **Terminal Node**: A terminal node is a node which has no descendants (child nodes). Also called a “leaf node.”  

**Properties of Trees**  
* Can handle huge datasets.  
* Can handle mixed predictors implicitly – numeric and categorical.  
* Easily ignore redundant variables.  
* Handle missing data elegantly through surrogate splits.  
* Small trees are easy to interpret.  
* Large trees are hard to interpret.  
* Prediction performance is often poor (high variance).  

**Tree Algorithms**  
There are a handful of different tree algorithms in addition to Breiman’s original CART algorithm. Namely, ID3, C4.5 and C5.0, all created by Ross Quinlan. C5.0 is an improvement over C4.5, however, the C4.5 algorithm is still quite popular since the multi-threaded version of C5.0 is proprietary (although the single threaded is released as GPL).  

**CART vs C4.5**  
Here are some of the differences between CART and C4.5:

* Tests in CART are always binary, but C4.5 allows two or more outcomes.  
* CART uses the Gini diversity index to rank tests, whereas C4.5 uses information-based criteria.  
* CART prunes trees using a cost-complexity model whose parameters are estimated by cross-validation; C4.5 uses a single-pass algorithm derived from binomial confidence limits.  
* With respect to missing data, CART looks for surrogate tests that approximate the outcomes when the tested attribute has an unknown value, but C4.5 apportions the case probabilistically among the outcomes.  

Decision trees are formed by a collection of rules based on variables in the modeling data set:  

1. Rules based on variables’ values are selected to get the best split to differentiate observations based on the dependent variable.  
2. Once a rule is selected and splits a node into two, the same process is applied to each “child” node (i.e. it is a recursive procedure).  
3. Splitting stops when CART detects no further gain can be made, or some pre-set stopping rules are met. (Alternatively, the data are split as much as possible and then the tree is later pruned.)  

Each branch of the tree ends in a terminal node. Each observation falls into one and exactly one terminal node, and each terminal node is uniquely defined by a set of rules.  

## Classification Tree example
Let's use the data frame kyphosis to predict a type of deformation (kyphosis) after surgery, from age in months (Age), number of vertebrae involved (Number), and the highest vertebrae operated on (Start).


```r
# Classification Tree with rpart
library(rpart)

# grow tree
fit <- rpart(Kyphosis ~ Age + Number + Start,
   method="class", data=kyphosis)

printcp(fit) # display the results
```

```
## 
## Classification tree:
## rpart(formula = Kyphosis ~ Age + Number + Start, data = kyphosis, 
##     method = "class")
## 
## Variables actually used in tree construction:
## [1] Age   Start
## 
## Root node error: 17/81 = 0.20988
## 
## n= 81 
## 
##         CP nsplit rel error xerror    xstd
## 1 0.176471      0   1.00000 1.0000 0.21559
## 2 0.019608      1   0.82353 1.3529 0.23872
## 3 0.010000      4   0.76471 1.3529 0.23872
```

```r
plotcp(fit) # visualize cross-validation results
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-104-1.png" width="672" />

```r
summary(fit) # detailed summary of splits
```

```
## Call:
## rpart(formula = Kyphosis ~ Age + Number + Start, data = kyphosis, 
##     method = "class")
##   n= 81 
## 
##           CP nsplit rel error   xerror      xstd
## 1 0.17647059      0 1.0000000 1.000000 0.2155872
## 2 0.01960784      1 0.8235294 1.352941 0.2387187
## 3 0.01000000      4 0.7647059 1.352941 0.2387187
## 
## Variable importance
##  Start    Age Number 
##     64     24     12 
## 
## Node number 1: 81 observations,    complexity param=0.1764706
##   predicted class=absent   expected loss=0.2098765  P(node) =1
##     class counts:    64    17
##    probabilities: 0.790 0.210 
##   left son=2 (62 obs) right son=3 (19 obs)
##   Primary splits:
##       Start  < 8.5  to the right, improve=6.762330, (0 missing)
##       Number < 5.5  to the left,  improve=2.866795, (0 missing)
##       Age    < 39.5 to the left,  improve=2.250212, (0 missing)
##   Surrogate splits:
##       Number < 6.5  to the left,  agree=0.802, adj=0.158, (0 split)
## 
## Node number 2: 62 observations,    complexity param=0.01960784
##   predicted class=absent   expected loss=0.09677419  P(node) =0.7654321
##     class counts:    56     6
##    probabilities: 0.903 0.097 
##   left son=4 (29 obs) right son=5 (33 obs)
##   Primary splits:
##       Start  < 14.5 to the right, improve=1.0205280, (0 missing)
##       Age    < 55   to the left,  improve=0.6848635, (0 missing)
##       Number < 4.5  to the left,  improve=0.2975332, (0 missing)
##   Surrogate splits:
##       Number < 3.5  to the left,  agree=0.645, adj=0.241, (0 split)
##       Age    < 16   to the left,  agree=0.597, adj=0.138, (0 split)
## 
## Node number 3: 19 observations
##   predicted class=present  expected loss=0.4210526  P(node) =0.2345679
##     class counts:     8    11
##    probabilities: 0.421 0.579 
## 
## Node number 4: 29 observations
##   predicted class=absent   expected loss=0  P(node) =0.3580247
##     class counts:    29     0
##    probabilities: 1.000 0.000 
## 
## Node number 5: 33 observations,    complexity param=0.01960784
##   predicted class=absent   expected loss=0.1818182  P(node) =0.4074074
##     class counts:    27     6
##    probabilities: 0.818 0.182 
##   left son=10 (12 obs) right son=11 (21 obs)
##   Primary splits:
##       Age    < 55   to the left,  improve=1.2467530, (0 missing)
##       Start  < 12.5 to the right, improve=0.2887701, (0 missing)
##       Number < 3.5  to the right, improve=0.1753247, (0 missing)
##   Surrogate splits:
##       Start  < 9.5  to the left,  agree=0.758, adj=0.333, (0 split)
##       Number < 5.5  to the right, agree=0.697, adj=0.167, (0 split)
## 
## Node number 10: 12 observations
##   predicted class=absent   expected loss=0  P(node) =0.1481481
##     class counts:    12     0
##    probabilities: 1.000 0.000 
## 
## Node number 11: 21 observations,    complexity param=0.01960784
##   predicted class=absent   expected loss=0.2857143  P(node) =0.2592593
##     class counts:    15     6
##    probabilities: 0.714 0.286 
##   left son=22 (14 obs) right son=23 (7 obs)
##   Primary splits:
##       Age    < 111  to the right, improve=1.71428600, (0 missing)
##       Start  < 12.5 to the right, improve=0.79365080, (0 missing)
##       Number < 3.5  to the right, improve=0.07142857, (0 missing)
## 
## Node number 22: 14 observations
##   predicted class=absent   expected loss=0.1428571  P(node) =0.1728395
##     class counts:    12     2
##    probabilities: 0.857 0.143 
## 
## Node number 23: 7 observations
##   predicted class=present  expected loss=0.4285714  P(node) =0.08641975
##     class counts:     3     4
##    probabilities: 0.429 0.571
```

```r
# plot tree
plot(fit, uniform=TRUE,
   main="Classification Tree for Kyphosis")
text(fit, use.n=TRUE, all=TRUE, cex=.8)
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-104-2.png" width="672" />

```r
# create attractive postscript plot of tree
post(fit, title = "Classification Tree for Kyphosis")

# prune the tree
pfit<- prune(fit, cp=fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])

# plot the pruned tree
#FIXME: pfit is not a tree just a root error
#plot(pfit, uniform=TRUE,
#   main="Pruned Classification Tree for Kyphosis")
#text(pfit, use.n=TRUE, all=TRUE, cex=.8)
#post(pfit, file = "c:/ptree.ps",
#   title = "Pruned Classification Tree for Kyphosis")
```

## Regression Tree example

```r
# Regression Tree Example
library(rpart)

# grow tree
fit <- rpart(Mileage~Price + Country + Reliability + Type,
   method="anova", data=cu.summary)

printcp(fit) # display the results
```

```
## 
## Regression tree:
## rpart(formula = Mileage ~ Price + Country + Reliability + Type, 
##     data = cu.summary, method = "anova")
## 
## Variables actually used in tree construction:
## [1] Price Type 
## 
## Root node error: 1354.6/60 = 22.576
## 
## n=60 (57 observations deleted due to missingness)
## 
##         CP nsplit rel error  xerror     xstd
## 1 0.622885      0   1.00000 1.06766 0.183445
## 2 0.132061      1   0.37711 0.55189 0.106967
## 3 0.025441      2   0.24505 0.38624 0.085277
## 4 0.011604      3   0.21961 0.38576 0.079206
## 5 0.010000      4   0.20801 0.42680 0.084753
```

```r
plotcp(fit) # visualize cross-validation results
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-105-1.png" width="672" />

```r
summary(fit) # detailed summary of splits
```

```
## Call:
## rpart(formula = Mileage ~ Price + Country + Reliability + Type, 
##     data = cu.summary, method = "anova")
##   n=60 (57 observations deleted due to missingness)
## 
##           CP nsplit rel error    xerror       xstd
## 1 0.62288527      0 1.0000000 1.0676647 0.18344526
## 2 0.13206061      1 0.3771147 0.5518870 0.10696736
## 3 0.02544094      2 0.2450541 0.3862402 0.08527743
## 4 0.01160389      3 0.2196132 0.3857621 0.07920566
## 5 0.01000000      4 0.2080093 0.4267977 0.08475255
## 
## Variable importance
##   Price    Type Country 
##      48      42      10 
## 
## Node number 1: 60 observations,    complexity param=0.6228853
##   mean=24.58333, MSE=22.57639 
##   left son=2 (48 obs) right son=3 (12 obs)
##   Primary splits:
##       Price       < 9446.5  to the right, improve=0.6228853, (0 missing)
##       Type        splits as  LLLRLL,      improve=0.5044405, (0 missing)
##       Reliability splits as  LLLRR,       improve=0.1263005, (11 missing)
##       Country     splits as  --LRLRRRLL,  improve=0.1243525, (0 missing)
##   Surrogate splits:
##       Type    splits as  LLLRLL,     agree=0.950, adj=0.750, (0 split)
##       Country splits as  --LLLLRRLL, agree=0.833, adj=0.167, (0 split)
## 
## Node number 2: 48 observations,    complexity param=0.1320606
##   mean=22.70833, MSE=8.498264 
##   left son=4 (23 obs) right son=5 (25 obs)
##   Primary splits:
##       Type        splits as  RLLRRL,      improve=0.43853830, (0 missing)
##       Price       < 12154.5 to the right, improve=0.25748500, (0 missing)
##       Country     splits as  --RRLRL-LL,  improve=0.13345700, (0 missing)
##       Reliability splits as  LLLRR,       improve=0.01637086, (10 missing)
##   Surrogate splits:
##       Price   < 12215.5 to the right, agree=0.812, adj=0.609, (0 split)
##       Country splits as  --RRLRL-RL,  agree=0.646, adj=0.261, (0 split)
## 
## Node number 3: 12 observations
##   mean=32.08333, MSE=8.576389 
## 
## Node number 4: 23 observations,    complexity param=0.02544094
##   mean=20.69565, MSE=2.907372 
##   left son=8 (10 obs) right son=9 (13 obs)
##   Primary splits:
##       Type    splits as  -LR--L,      improve=0.515359600, (0 missing)
##       Price   < 14962   to the left,  improve=0.131259400, (0 missing)
##       Country splits as  ----L-R--R,  improve=0.007022107, (0 missing)
##   Surrogate splits:
##       Price < 13572   to the right, agree=0.609, adj=0.1, (0 split)
## 
## Node number 5: 25 observations,    complexity param=0.01160389
##   mean=24.56, MSE=6.4864 
##   left son=10 (14 obs) right son=11 (11 obs)
##   Primary splits:
##       Price       < 11484.5 to the right, improve=0.09693168, (0 missing)
##       Reliability splits as  LLRRR,       improve=0.07767167, (4 missing)
##       Type        splits as  L--RR-,      improve=0.04209834, (0 missing)
##       Country     splits as  --LRRR--LL,  improve=0.02201687, (0 missing)
##   Surrogate splits:
##       Country splits as  --LLLL--LR, agree=0.80, adj=0.545, (0 split)
##       Type    splits as  L--RL-,     agree=0.64, adj=0.182, (0 split)
## 
## Node number 8: 10 observations
##   mean=19.3, MSE=2.21 
## 
## Node number 9: 13 observations
##   mean=21.76923, MSE=0.7928994 
## 
## Node number 10: 14 observations
##   mean=23.85714, MSE=7.693878 
## 
## Node number 11: 11 observations
##   mean=25.45455, MSE=3.520661
```

```r
# create additional plots
par(mfrow=c(1,2)) # two plots on one page
rsq.rpart(fit) # visualize cross-validation results  
```

```
## 
## Regression tree:
## rpart(formula = Mileage ~ Price + Country + Reliability + Type, 
##     data = cu.summary, method = "anova")
## 
## Variables actually used in tree construction:
## [1] Price Type 
## 
## Root node error: 1354.6/60 = 22.576
## 
## n=60 (57 observations deleted due to missingness)
## 
##         CP nsplit rel error  xerror     xstd
## 1 0.622885      0   1.00000 1.06766 0.183445
## 2 0.132061      1   0.37711 0.55189 0.106967
## 3 0.025441      2   0.24505 0.38624 0.085277
## 4 0.011604      3   0.21961 0.38576 0.079206
## 5 0.010000      4   0.20801 0.42680 0.084753
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-105-2.png" width="672" />

```r
# plot tree
plot(fit, uniform=TRUE,
   main="Regression Tree for Mileage ")
text(fit, use.n=TRUE, all=TRUE, cex=.8)

# create attractive postcript plot of tree
post(fit, file = "tree2.ps",
   title = "Regression Tree for Mileage ")

# prune the tree
pfit<- prune(fit, cp=0.01160389) # from cptable   

# plot the pruned tree
plot(pfit, uniform=TRUE,
   main="Pruned Regression Tree for Mileage")
text(pfit, use.n=TRUE, all=TRUE, cex=.8)
```

<img src="bookdown-demo_files/figure-html/unnamed-chunk-105-3.png" width="672" />

```r
post(pfit, file = "ptree2.ps",
   title = "Pruned Regression Tree for Mileage")
```

**Sources**  
[Tree-Based Models at Quick-R by datacamp](https://www.statmethods.net/advstats/cart.html)  
[UseR! Machine Learnign Turorial](https://koalaverse.github.io/machine-learning-in-R/decision-trees.html)

<!--chapter:end:46_tree-based_models.Rmd-->

