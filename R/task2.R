# Description:
#
# Classification experiment for task 1 of CDMC 2015 using the
# Domain-Specific Classifier.

source("dsc.R")

library(caTools)
library(caret)
library(randomForest)
library(e1071)
library(testthat)

# Set working directory
setwd("~/h/proj/cdmc-2015/R")

# Load datasets
file.b.train <- "~/h/proj/cdmc-2015/data/task2/BuyerTrain.csv"
file.s.train <- "~/h/proj/cdmc-2015/data/task2/SellerTrain.csv"
df.buyers <- read.csv(file.b.train,
                      header = FALSE,
                      colClasses = c(rep("integer", 25), "factor"),
                      col.names = c(paste0("X", 1:25), "class"))
df.sellers <- read.csv(file.s.train,
                       header = FALSE,
                       colClasses = c(rep("integer", 25), "factor"),
                       col.names = c(paste0("X", 1:25), "class"))


### UNIT TESTS: dsc.build.DTM, dsc.build.docs.df
test_that("Testing statistics of BuyerTrain.csv and SellerTrain.csv", {
  expect_equal(table(df.buyers$class),
               c("1"=27739, "2"=2671, "3"=4291))
  expect_equal(table(df.sellers$class),
               c("4"=22191, "5"=3745, "6"=3442))
})

# Convert datasets to lower case
train <- df.train
train$text <- lapply(train$text, tolower)

test <- df.test
test$text <- lapply(test$text, tolower)

# Count frequency of the words in the document
corpus <- Corpus(VectorSource(train$text))
freq <- DocumentTermMatrix(corpus)
freq

# ##<<DocumentTermMatrix (documents: 1339, terms: 51320)>>
# Non-/sparse entries: 300356/68417124
# Sparsity           : 100%
# Maximal term length: 98
# Weighting          : term frequency (tf)

#inspection find lots of sparseness
inspect(freq[1000:1005,505:515])
#find most frequent words 
findFreqTerms(freq, lowfreq=500)
#remove sparse words (0.99 represent to only keep terms that appear in 1% or more)
sparse <- removeSparseTerms(freq, 0.995)
sparse
# <<DocumentTermMatrix (documents: 1338, terms: 3432)>>
#   Non-/sparse entries: 200133/4391883
# Sparsity           : 96%
# Maximal term length: 30
# Weighting          : term frequency (tf)

#convert sparse matrix into a data frame for prediction
trainDTM <- sparse
trainSparse <- as.data.frame(as.matrix(sparse))
#make sure colume names are words
colnames(trainSparse) <- make.names(colnames(trainSparse))

#split into train and test (70% for training)
# split <- sample.split(train$class, SplitRatio=0.95)
# train1.df <- subset(trainSparse, split==TRUE)
# test1.df <- subset(trainSparse, split==FALSE)
# train1.labels <- subset(train$class, split==TRUE)
# test1.labels <- subset(train$class, split==FALSE)

set.seed(1)
split <- sample.split(train$class, SplitRatio=0.6)
train1 <- trainDTM[split==TRUE, ]
test1 <- trainDTM[split==FALSE, ]
train1.labels <- subset(train$class, split==TRUE)
test1.labels <- subset(train$class, split==FALSE)

###
### Classification with Domain-Specific Classifier
###

model.dsc <- dsc(train1, train1.labels, alpha = 0.75, p = 1.0)
pred.dsc <- predict(model.dsc, test1)
cm <- confusionMatrix(test1.labels, pred.dsc)
cm$overall[1]

###
### Classification with Domain-Specific Classifier
### Do not drop any words from the corpus this time!
###

# Setup training/testing datasets
alpha <- 1.0
p <- Inf
SplitRatio <- 0.7
trainDMT <- freq

set.seed(0)
split <- sample.split(train$class, SplitRatio=SplitRatio)
train1 <- trainDTM[split==TRUE, ]
test1 <- trainDTM[split==FALSE, ]
train1.labels <- subset(train$class, split==TRUE)
test1.labels <- subset(train$class, split==FALSE)



###
### Classification with Domain-Specific Classifier
### Calculate the best accuracy over a grid of parameters for alpha and p
###

results <- c()
for (alpha in c(0, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5)) {
  for (p in c(0.5, 1.0, 1.5, 2.0, 3.0, 5.0, Inf)) {
    model.dsc <- dsc(train1, train1.labels, alpha = alpha, p = p)
    pred.dsc <- predict(model.dsc, test1)
    cm <- confusionMatrix(test1.labels, pred.dsc)
    cm$overall[1]
    
    newres <- c(alpha=alpha, p=p, accuracy=cm$overall[1])
    results <- rbind(results, newres)
    print(newres)
  }
}
results <- data.frame(results)
results[order(results$accuracy, decreasing = TRUE), ]

###
### Classification with Random Forests
###

model.rf <- randomForest(as.data.frame(as.matrix(train1)), factor(train1.labels), ntree=50)
pred.rf <- predict(model.rf, as.data.frame(as.matrix(test1)))
cm.rf <- confusionMatrix(test1.labels, pred.rf)
cm.rf$overall[1]

###
### Classification with SVM linear
###

model.svm.linear <- svm(as.data.frame(as.matrix(train1)), factor(train1.labels),
                        kernel = "linear", scale = FALSE)
pred.svm.linear <- predict(model.svm.linear, as.data.frame(as.matrix(test1)))
cm.svm.linear <- confusionMatrix(test1.labels, pred.svm.linear)
cm.svm.linear$overall[1]

model.svm.rb <- svm(as.data.frame(as.matrix(train1)), factor(train1.labels),
                    kernel = "radial", scale = FALSE)
pred.svm.rb <- predict(model.svm.rb, as.data.frame(as.matrix(test1)))
cm.svm.rb <- confusionMatrix(test1.labels, pred.svm.rb)
cm.svm.rb$overall[1]
