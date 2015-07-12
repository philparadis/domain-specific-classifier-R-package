# Description:
#
# Classification experiment for task 1 of CDMC 2015 using the
# Domain-Specific Classifier.

source("dsc.R")


# Set working directory
setwd("/proj/cdmc-2015")

# Load datasets
file.train <- "data/task1/EnewsTrain.csv"
file.test <- "data/task1/EnewsTest.csv"
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
library(caTools)
split <- sample.split(train$class, SplitRatio=0.95)
train1.df <- subset(trainSparse, split==TRUE)
test1.df <- subset(trainSparse, split==FALSE)
train1.labels <- subset(train$class, split==TRUE)
test1.labels <- subset(train$class, split==FALSE)

split <- sample.split(train$class, SplitRatio=0.95)
train1 <- trainDTM[split==TRUE, ]
test1 <- trainDTM[split==FALSE, ]
train1.labels <- subset(train$class, split==TRUE)
test1.labels <- subset(train$class, split==FALSE)

# Classification with Domain-Specific Classifier

model <- dsc(train1, train1.labels, alpha = 1.0, p = 1.0)
pred <- predict(model, test1)

confusionMatrix(test1.labels, pred)


# #baseline: predict the most frquent class
# table(test1$class)
# 84/nrow(test1)
# #baseline Acuracy=0.2089552
# 
# 
# #CART to predict
# #build model on training dataset
# library(rpart)
# library(rpart.plot)
# cart<-rpart(class~.,data=train1,method="class")
# prp(cart)
# 
# #predict on testing dataset
# predCart<-predict(cart, newdata=test1, type="class")
# table(test1$class,predCart)
# #predCart confusion matrix
# #   1  2  3  4  5
# #1 41  8 10  6 12
# #2  4 41 30  0  5
# #3  5  2 66  0  6
# #4  7 17  8 43  9
# #5  5 19 10  4 44
# (41+41+66+43+44)/nrow(test1)
# 
# #randomForest to predict
# library(randomForest)
# RF<-randomForest(class ~ .,data=train1)
# predRF<-predict(RF,newdata=test1)
# table(test1$class,predRF)
# (62+77+73+69+75)/nrow(test1)