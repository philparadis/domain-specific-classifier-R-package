# CDMC task 1

if (!file.exists("dsc.R")) {
  stop(paste0("Could not file required source file 'dsc.R'. Make sure that ",
              "you set the current working directory to the R source file ",
              "directory of the cdmc-2015 project."))
}


# prepare environment
source("dsc.r")

library(tm)
library(caTools)
library(caret)

# Load dataset
file.train <- "data/EnewsTrain.csv"
df.train <- read.csv(file.train,
                     header = FALSE,
                     colClasses = c("character", "factor"),
                     col.names = c("text", "class"))

# Convert dataset to lower case
df.train$text <- lapply(df.train$text, tolower)

# Count frequency of the words in the document
corpus <- Corpus(VectorSource(df.train$text))
(freq <- DocumentTermMatrix(corpus))

# Inspection find lots of sparseness
inspect(freq[1000:1005,505:515])

# Find most frequent words 
findFreqTerms(freq, lowfreq=500)

# Remove sparse words (0.99 represent to only keep terms that appear in 1% or more)
(sparseDTM <- removeSparseTerms(freq, 0.99))

# Split into train and test (70% for training)
set.seed(1)
split <- sample.split(1:nrow(sparseDTM), SplitRatio=0.7)
train1 <- sparseDTM[split==TRUE, ]
test1 <- sparseDTM[split==FALSE, ]
train1.labels <- subset(df.train$class, split==TRUE)
test1.labels <- subset(df.train$class, split==FALSE)

###
### Classification with Domain-Specific Classifier
###

# Train a model with 'dsc'
model.dsc <- dsc(train1, train1.labels, alpha = 0.75, p = 1.0)
model.dsc

# Classify the training set
train.pred.dsc <- predict(model.dsc, train1)
# Compute the training error
(cm <- confusionMatrix(train1.labels, train.pred.dsc))
cm$overall[1]

# Classify the testing set
test.pred.dsc <- predict(model.dsc, test1)
# Compute the testing error
(cm <- confusionMatrix(test1.labels, test.pred.dsc))
cm$overall[1]
