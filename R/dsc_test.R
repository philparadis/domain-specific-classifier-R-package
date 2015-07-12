# Description:
# 
# Unit tests for the Domain-Specific Classifier implemented in "dsc.R".
#
# Reference:
#
# [1] H.H. Duan, V.G. Pestov, and V. Singla, "Text Categorization via Similarity
#     Search: An Efficient and Effective Novel Algorithm"

library(testthat)
library(tm)

source("dsc.R")

df.test1 <- data.frame(text = c("aaa aaa aaa aaa",
                                "aaa aaa aaa bbb",
                                "aaa aaa bbb bbb",
                                "aaa bbb bbb ccc ccc"),
                       class = factor(c(0, 0, 1, 1)),
                       stringsAsFactors = FALSE)

# c(a, d1) = 4
# c(a, d2) = 3
# c(a, d3) = 2
# c(a, d4) = 1
# c(b, d1) = 0
# c(b, d2) = 1
# c(b, d3) = c(b, d4) = 2
# c(c, d1) = c(c, d2) = c(c, d3) = 0
# c(c, d4) = 2
# |d1| = |d2| = |d3| = 4
# |d4| = 5
#
# |D^0| = 2
# |D^1| = 2
#
# f_0(a) = 1/|D^0| * ( c(a, d1) / |d1| + c(a, d2) / |d2| )
#        = 1/2*[4/4 + 3/4] = 0.5 * 1.75 = 0.875
# f_0(b) = 1/|D^0| * ( c(b, d1) / |d1| + c(b, d2) / |d2| )
#        = 1/2*[0/4 + 1/4] = 0.5 * 0.25 = 0.125
# f_0(c) = 1/|D^0| * ( c(c, d1) / |d1| + c(c, d2) / |d2| )
#        = 1/2*[0/4 + 0/4] = 0.5 * 0 = 0.25
# f_1(a) = 1/|D^0| * ( c(a, d3) / |d3| + c(a, d4) / |d4| )
#        = 1/2*[2/4 + 1/5] = 0.5 * 0.7 = 0.35
# f_1(b) = 1/|D^0| * ( c(b, d3) / |d3| + c(b, d4) / |d4| )
#        = 1/2*[2/4 + 2/5] = 0.5 * 0.9 = 0.45
# f_1(c) = 1/|D^0| * ( c(c, d3) / |d3| + c(c, d4) / |d4| )
#        = 1/2*[0/4 + 2/5] = 0.5 * 0.4 = 0.2

df.test2 <- data.frame(text=c("one black dog two white cat pig black bat bat",
                              "cat cat cat three cat dog bat",
                              "pig dog pig green dog",
                              "red blue green bat",
                              "white cat black dog",
                              "blue green red red red",
                              "one dog two cat three pig",
                              "one one one one one two two"),
                       class=factor(c("animal", "animal", "animal",
                                      "color", "color", "color",
                                      "number", "number")),
                       stringsAsFactors = FALSE)
categories.test2 <- factor(levels(df.test2$class))

DTM.test1 <- dsc.build.DTM(df.test1$text)
docs.test1 <- dsc.build.docs.df(df.test1$text)
apm.test1 <- dsc.build.avg.prop.matrix(DTM.test1, df.test1$class)
alpha.matrix.test1 <- dsc.build.alpha.sums.matrix(apm.test1, df.test1$class, alpha = 1.0)
CS.test1 <- dsc.build.CS.matrix(apm.test1, alpha.matrix.test1)

### UNIT TESTS: dsc.build.DTM, dsc.build.docs.df
test_that("Testing dsc.build.DTM and dsc.build.docs.df functions", {
  expect_equivalent(as.data.frame(as.matrix(DTM.test1)),
                    data.frame(aaa=c(4,3,2,1), bbb=c(0,1,2,2), ccc=c(0,0,0,2)))
  expect_equivalent(docs.test1,
                    data.frame(aaa=c(4,3,2,1), bbb=c(0,1,2,2), ccc=c(0,0,0,2)))
  # TODO: Add tests for apm.test1, alpha.matrix.test1 and CS.test1 ?
})

### UNIT TESTS: dsc.build.avg.prop.matrix
test_that("Testing dsc.build.avg.prop.matrix function", {
  expect_equal(apm.test1,
               matrix(c(0.875, 0.35, 0.125, 0.450, 0, 0.2), 2,
                      dimnames=list(c("0","1"), c("aaa","bbb","ccc"))))
})

# Series of unit tests:
#  - Build documents bag-of-word
#  - Build avg. prop. matrix
#  - Build alpha sums matrix
#  - Build domain-specific words matrix
#  - Check return type at each step
#  - Check correctness of domain-specific words matrix
docs.test2 <- dsc.build.docs.df(df.test2$text)
DTM.test2 <- dsc.build.DTM(df.test2$text)
apm.test2 <- dsc.build.avg.prop.matrix(docs.test2, df.test2$class)
alpha.matrix.test2 <- dsc.build.alpha.sums.matrix(apm.test2, df.test2$class, alpha = 1.0)
CS.test2 <- dsc.build.CS.matrix(apm.test2, alpha.matrix.test2)
CS.expected <- data.frame(bat=c(TRUE,FALSE,FALSE), black=c(FALSE,TRUE,FALSE),
                          blue=c(FALSE,TRUE,FALSE),cat=c(TRUE,FALSE,FALSE),
                          dog=c(TRUE,FALSE,FALSE),green=c(FALSE,TRUE,FALSE),
                          one=c(FALSE,FALSE,TRUE),pig=c(TRUE,FALSE,FALSE),
                          red=c(FALSE,TRUE,FALSE),three=c(FALSE,FALSE,TRUE),
                          two=c(FALSE,FALSE,TRUE),white=c(FALSE,TRUE,FALSE))
row.names(CS.expected) <- c("animal","color","number")
CS.expected <- as.matrix(CS.expected)

### UNIT TESTS: Return types
test_that("Testing the class types returned by various dsc helper functions", {
  expect_is(docs.test2, "data.frame")
  expect_is(DTM.test2, "DocumentTermMatrix")
  expect_is(apm.test2, "matrix")
  expect_is(alpha.matrix.test2, "matrix")
  expect_is(CS.test2, "matrix")
})
test_that("Testing dsc.build.CS.matrix function", {
  expect_equal(CS.test2, CS.expected)
})

### Classification
sentence <- paste("red cat red dog red bat two",
                  "red pig one two three black",
                  "green blue black black bat")
# We know that CS.test2 is given by:
#        bat black blue cat dog green one pig red three two white
# animal   1     0    0   1   1     0   0   1   0     0   0     0
# color    0     1    1   0   0     1   0   0   1     0   0     1
# number   0     0    0   0   0     0   1   0   0     1   1     0
# 
newDoc.test2 <- tail(dsc.build.docs.df(c(df.test2$text, sentence)), 1)
rel.freq.vec <- dsc.compute.total.rel.freq.vec(CS.test2,
                                               df.test2$class,
                                               newDoc.test2)
# Given a new document d, as above, i.e.:
#
#     bat black blue cat dog green one pig red three two white
# d     2     3    1   1   1     1   1   1   4     1   2     0
# 
# If we compute w[CS_j] = (1 / |d|) * sum_{t \in CS_j} c(t, d)
# then we should get:
#
# |d| = 18
# w[CS_animal] = (1/18) * [2 + 1 + 1 + 1]     = 5/18
# w[CS_color]  = (1/18) * [3 + 1 + 1 + 4 + 0] = 9/18
# w[CS_number] = (1/18) * [1 + 1 + 2]         = 4/18

rel.freq.vec.expected <- c(animal=5/18, color=9/18, number=4/18)

### UNIT TEST: dsc.compute.total.rel.freq
test_that("Testing dsc.compute.total.rel.freq", {
  expect_equal(rel.freq.vec, rel.freq.vec.expected)
})

# Next, compute |CS_j|, i.e. the number of domain-specific words
# for each category j.
#
# It should be:
# |CS_animal| = |{bat, cat, dog, pig}| = 4
# |CS_color| = |{black, blue, green, red, white}| = 5
# |CS_number| = |{one, two, three}| = 3
CS.lengths.test2 <- dsc.compute.CS.lengths.vec(CS.test2)
CS.lengths.expected <- c(animal=4, color=5, number=3)

### UNIT TEST: dsc.compute.CS.lengths.vec
test_that("Testing dsc.compute.CS.lengths.vec", {
  expect_equal(CS.lengths.test2,
               CS.lengths.expected)
})

# Next, compute the classification ratios for 'newDoc.test2'.
# It corresponds to the formula:
#
#   w[CS_j] / |CS_j|^(1/p)
#
# So, we should expect:
#
# For p=1:
#   j=animal, ratio = (5/18) / 4 = 5/72
#   j=color,  ratio = (9/18) / 5 = 1/10
#   j=number, ratio = (4/18) / 3 = 2/27
# For p=2:
#   j=animal, ratio = (5/18) / 4^0.5 = 5/36
#   j=color,  ratio = (9/18) / 5^0.5
#   j=number, ratio = (4/18) / 3^0.5
# For p=Inf:
#   j=animal, ratio = (5/18) / 1 = 5/18
#   j=color,  ratio = (9/18) / 1 = 1/2
#   j=number, ratio = (4/18) / 1 = 2/9

### UNIT TESTS: dsc.compute.classification.ratios
test_that("Testing dsc.compute.classification.ratios", {
  expect_equal(dsc.compute.classification.ratios(CS.test2, df.test2$class,
                                                 newDoc.test2, p = 1),
               matrix(c(5/72, 1/10, 2/27), 1, 3,
                      dimnames=list(9, c("animal", "color", "number"))))
  expect_equal(dsc.compute.classification.ratios(CS.test2, df.test2$class,
                                                 newDoc.test2, p = 2),
               matrix(c(5/36, (9/18)/5^0.5, (4/18)/3^0.5), 1, 3,
                      dimnames=list(9, c("animal", "color", "number"))))
  expect_equal(dsc.compute.classification.ratios(CS.test2, df.test2$class,
                                                 newDoc.test2, p = Inf),
               matrix(c(5/18, 1/2, 2/9), 1, 3,
                      dimnames=list(9, c("animal", "color", "number"))))
})

# UNIT TESTS: dsc
model.test2 <- dsc(df.test2$text, df.test2$class, alpha = 1.0, p = 2)
test_that("Testing dsc", {
  expect_equal(model.test2$alpha, 1.0)
  expect_equal(model.test2$p, 2)
  expect_equal(model.test2$CS, CS.expected)
})

### Predicting new data with 'predict.dsc'
### Predicting a single document at a time
test_that("Testing predict.dsc on single documents", {
  expect_equivalent(predict(model.test2, "red"),
                    categories.test2[which(categories.test2 == "color")])
  expect_equivalent(predict(model.test2, "pig"),
                    categories.test2[which(categories.test2 == "animal")])
  expect_equivalent(predict(model.test2, "one"),
                    categories.test2[which(categories.test2 == "number")])
  expect_equivalent(predict(model.test2, "dog pig red white one two three one two three"),
                    categories.test2[which(categories.test2 == "number")])
  expect_equivalent(predict(model.test2, "dog dog cat cat pig pig bat bat one two white blue red"),
                    categories.test2[which(categories.test2 == "animal")])
  expect_equivalent(predict(model.test2, "white pig two black red blue"),
                    categories.test2[which(categories.test2 == "color")])
})

### Predicting multiple new documents at a time
newDocs.df.test2 <- data.frame(text=c("bat bat dog cat white",
                                      "cat cat cat red pig",
                                      "green dog dog dog dog two",
                                      "red green green blue blue one cat",
                                      "white white black black white white dog",
                                      "pig blue blue blue bat",
                                      "one two three three two one white",
                                      "black dog one three one one one two two"),
                              class=factor(c("animal", "animal", "animal",
                                             "color", "color", "color",
                                             "number", "number")),
                              stringsAsFactors = FALSE)

### UNIT TESTS: predict.dsc
test_that("Testing predict.dsc on multiple documents", {
  expect_equal(predict(model.test2, newDocs.df.test2$text),
               newDocs.df.test2$class)
  expect_equal(predict(model.test2, newDocs.df.test2[, "text", drop=FALSE]),
               newDocs.df.test2$class)
  newDocs.DTM.test2 <- dsc.build.DTM(c(df.test2$text, newDocs.df.test2$text))
  newDocs.DTM.test2 <- newDocs.DTM.test2[(nrow(df.test2)+1):(nrow(df.test2)+nrow(newDocs.df.test2)), ]
  expect_equal(predict(model.test2, newDocs.DTM.test2),
               newDocs.df.test2$class)
  expect_equal(predict(model.test2, as.data.frame(as.matrix(newDocs.DTM.test2))),
               newDocs.df.test2$class)
})


# TESTING FOR:
#   - Typical coding errors
#   - Parameters handling (sanity checks, types check, range checks)
#   - Invalid input handling (words outside of dictionary, adverserial input)

sentence1 <- "red cat black pig HELLO FRIEND ALOHA"
sentence2 <- "Tell me -- quickly! -- what is the password!?!?"    
newDoc.test3 <- tail(dsc.build.docs.df(c(df.test2$text, sentence1)), 1)
newDoc.test4 <- tail(dsc.build.docs.df(c(df.test2$text, sentence2)), 1)

# Split into train and test (70% for training)
# library(caTools)
# split <- sample.split(docs.enews15$class, SplitRatio=0.7)
# train1 <- subset(docs.enews15, split==TRUE)
# test1 <- subset(docs.enews15, split==FALSE)
# 
# table(train1$class)
# table(test1$class)


# TESTING DATA

# # Set working directory
# dir.datasets <- "/proj/data/cardoso-datasets"
# file.r8.train <- paste(dir.datasets, "r8-train-no-stop.csv", sep="/")
# file.r8.test <- paste(dir.datasets, "r8-test-no-stop.csv", sep="/")
# file.enews.train <- "/proj/data/cdmc-2012/enews_train_data.csv"
# file.enews.test <- "/proj/data/cdmc-2012/enews_test_data.csv"
# file.enews15.train <- "/proj/cdmc-2015/data/task1/EnewsTrain.csv"
# file.enews15.test <- "/proj/cdmc-2015/data/task1/EnewsTest.csv"
# 
# # Load datasets
# r8.train <- read.csv(file.r8.train,
#                      header = FALSE,
#                      colClasses = c("factor", "character"),
#                      col.names = c("class", "text"))
# r8.test <- read.csv(file.r8.test,
#                     header = FALSE,
#                     colClasses = c("factor", "character"),
#                     col.names = c("class", "text"))
# enews.train <- read.csv(file.enews.train,
#                         header = FALSE,
#                         colClasses = c("factor", "character"),
#                         col.names = c("class", "text"))
# enews.train <- read.csv(file.enews.train,
#                         header = FALSE,
#                         colClasses = c("factor", "character"),
#                         col.names = c("class", "text"))
# enews15.train <- read.csv(file.enews15.train,
#                         header = FALSE,
#                         colClasses = c("character", "factor"),
#                         col.names = c("text", "class"))
# enews15.test <- read.csv(file.enews15.test,
#                         header = FALSE,
#                         colClasses = c("character", "factor"),
#                         col.names = c("text", "class"))
# 
# #
# enews15 <- list(docs = dsc.build.docs.df(enews15.train$text),
#                 labels = enews15.train$class)
# 
# # Build documents matrix
# corpus.train <- Corpus(VectorSource(r8.train$text))
# corpus.test <- Corpus(VectorSource(r8.test$text))
# corpus.enews <- Corpus(VectorSource(enews.train$text))
# corpus.enews15 <- Corpus(VectorSource(enews15.train$text))
# docTerm.train <- DocumentTermMatrix(corpus.train)
# docTerm.test <- DocumentTermMatrix(corpus.test)
# docTerm.enews <- DocumentTermMatrix(corpus.enews)
# docTerm.enews15 <- DocumentTermMatrix(corpus.enews15)
# # If desired, drop infrequent words from dictionary
# # Here 0.99 means to only keep terms that appear in 1% or more documents
# sparseDocTerm.enews15 <- removeSparseTerms(docTerm.enews15, 0.99)
# 
# # Convert sparse matrix into a data frame for prediction
# docs.train <- as.data.frame(as.matrix(docTerm.train))
# docs.test <- as.data.frame(as.matrix(docTerm.test))
# docs.enews <- as.data.frame(as.matrix(docTerm.enews))
# docs.enews15 <- as.data.frame(as.matrix(sparseDocTerm.enews15))
# # Make sure column names are words
# colnames(docs.train) <- make.names(colnames(docs.train))
# colnames(docs.test) <- make.names(colnames(docs.test))
# colnames(docs.enews) <- make.names(colnames(docs.enews))
# colnames(docs.enews15) <- make.names(colnames(docs.enews15))
# # Add class labels to the corpus
# docs.train$class <- r8.train$class
# docs.test$class <- r8.test$class
# docs.enews$class <- enews.train$class
# docs.enews15$class <- enews15.train$class
# 
# # Print summary of labels for training and testing datasets
# table(docs.train$class)
# table(docs.test$class)
# table(docs.enews$class)
# table(docs.enews15$class)
# 
# 
# # Split into train and test (70% for training)
# library(caTools)
# split <- sample.split(docs.enews15$class, SplitRatio=0.7)
# train1 <- subset(docs.enews15, split==TRUE)
# test1 <- subset(docs.enews15, split==FALSE)
# 
# table(train1$class)
# table(test1$class)
# 
# # CLASSIFIERS
# # Baseline: Predict the most frequent class
# most.freq.label <- which.max(table(train1$class))[[1]]
# cat(paste0("Baseline training accuracy: ",
#            max(table(train1$class)) / nrow(train1)))
# cat(paste0("Baseline testing accuracy:  ",
#            table(test1$class)[[most.freq.label]] / nrow(test1)))
# 
# # Linear SVM
# 
# # Split into training and testing sets (70% for training)
# # library(caTools)
# # split <-sample.split(trainSparse$class, SplitRatio=0.7)
# # train1 <- subset(trainSparse, split==TRUE)
# # test1 <- subset(trainSparse, split==FALSE)
# 
# library(e1071)
# library(caret)
# model <- svm(class ~ ., data = train1, kernel = "linear")
# pred <- predict(model, test1)
# confusionMatrix(pred, test1$class)
# 
# # Classification tree (rpart)
# library(rpart)
# library(rpart.plot)
# model <- rpart(class ~ ., data = train1, method = "class")
# #prp(cart)
# pred <- predict(model, test1)
# pred.labels <- apply(pred, 1, which.max)
# confusionMatrix(pred.labels, test1$class)
# 
# # Random forest
# library(randomForest)
# model <- randomForest(class ~ ., data = train1)
# pred <- predict(model, test1)
# confusionMatrix(pred, test1$class)