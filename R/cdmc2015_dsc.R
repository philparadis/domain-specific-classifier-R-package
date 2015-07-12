# Description:
#
# Classification experiment for task 1 of CDMC 2015 using the
# Domain-Specific Classifier.

source("dsc.R")

library(caTools)
library(caret)

# Set working directory
setwd("~/h/proj/cdmc-2015/R")

# Load datasets
file.train <- "~/h/proj/cdmc-2015/data/task1/EnewsTrain.csv"
file.test <- "~/h/proj/cdmc-2015/data/task1/EnewsTest.csv"
df.train <- read.csv(file.train,
                     header = FALSE,
                     colClasses = c("character", "integer"),
                     col.names = c("text", "class"))
df.test <- read.csv(file.test, header = FALSE,
                    colClasses = c("character"), col.names=c("text"))

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
sparse <- removeSparseTerms(freq, 0.99)
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

split <- sample.split(train$class, SplitRatio=0.7)
train1 <- trainDTM[split==TRUE, ]
test1 <- trainDTM[split==FALSE, ]
train1.labels <- subset(train$class, split==TRUE)
test1.labels <- subset(train$class, split==FALSE)

###
### Classification with Domain-Specific Classifier
###
model.dsc <- dsc(train1, train1.labels, alpha = 1.0, p = 3)
pred.dsc <- predict(model.dsc, test1)
cm <- confusionMatrix(test1.labels, pred.dsc)
cm$overall[1]

results <- c()
for (alpha in c(0, 0.25, 0.5, 0.75, 1, 2, 5, 10)) {
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
results

###
### Classification with Random Forests
###


###
### Classification with SVM linear
###