# DESCRIPTION:
# Scratch file for experimenting for the CDMC 2015 competition.

library(tm)

# Set working directory
setwd("/proj/cdmc-2015")

# Load datasets
file.train <- "data/task1/EnewsTrain.csv"
file.test <- "data/task1/EnewsTest.csv"
df.train <- read.csv(file.train,
                     header = FALSE,
                     colClasses = c("character", "integer"),
                     col.names = c("text", "class"))
df.test <- read.csv(file.test, header = FALSE)

# Convert datasets to lower case
train <- df.train
train$text <- lapply(train$text, tolower)

test <- df.test
test$text <- lapply(test$text, tolower)

### Play around with Corpus
corpus <- Corpus(VectorSource(train$text))

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
trainSparse <- as.data.frame(as.matrix(sparse))
#make sure colume names are words
colnames(trainSparse) <- make.names(colnames(trainSparse))
#add class labels to the corpus
trainSparse$class <- train$class

#split into train and test (70% for training)
library(caTools)
split <- sample.split(trainSparse$class, SplitRatio=0.7)
train1 <- subset(trainSparse, split==TRUE)
test1 <- subset(trainSparse, split==FALSE)

#baseline: predict the most frquent class
table(test1$class)
84/nrow(test1)
#baseline Acuracy=0.2089552


#CART to predict
#build model on training dataset
library(rpart)
library(rpart.plot)
cart<-rpart(class~.,data=train1,method="class")
prp(cart)

#predict on testing dataset
predCart<-predict(cart, newdata=test1, type="class")
table(test1$class,predCart)
#predCart confusion matrix
#   1  2  3  4  5
#1 41  8 10  6 12
#2  4 41 30  0  5
#3  5  2 66  0  6
#4  7 17  8 43  9
#5  5 19 10  4 44
(41+41+66+43+44)/nrow(test1)

#randomForest to predict
library(randomForest)
RF<-randomForest(class ~ .,data=train1)
predRF<-predict(RF,newdata=test1)
table(test1$class,predRF)
(62+77+73+69+75)/nrow(test1)