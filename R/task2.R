# Description:
#
# Classification experiment for task 1 of CDMC 2015 using the
# Domain-Specific Classifier.

# Directories
setwd("~/h/proj/cdmc-2015/R/")
file.b.train <- "~/h/proj/cdmc-2015/data/task2/BuyerTrain.csv"
file.s.train <- "~/h/proj/cdmc-2015/data/task2/SellerTrain.csv"

# Load DSC implementation
source("dsc.R")

# Load R libraries
library(caTools)
library(caret)
library(randomForest)
library(e1071)
library(testthat)
library(data.table)

# Load datasets
df.buyers <- read.csv(file.b.train,
                      header = FALSE,
                      colClasses = c(rep("integer", 25), "factor"),
                      col.names = c(paste0("X", 1:25), "class"))
df.sellers <- read.csv(file.s.train,
                       header = FALSE,
                       colClasses = c(rep("integer", 25), "factor"),
                       col.names = c(paste0("X", 1:25), "class"))

df.buyers$keywords <- as.matrix(df.buyers[, 1:25])
df.buyers <- df.buyers[, c("keywords", "class")]
df.sellers$keywords <- as.matrix(df.sellers[, 1:25])
df.sellers <- df.sellers[, c("keywords", "class")]

### UNIT TESTS: dsc.build.DTM, dsc.build.docs.df
test_that("Testing statistics of BuyerTrain.csv and SellerTrain.csv", {
  expect_equal(c(table(df.buyers$class)),
               c("1"=27739, "2"=2671, "3"=4291))
  expect_equal(c(table(df.sellers$class)),
               c("4"=22191, "5"=3745, "6"=3442))
})

### Check for "inconsistent labeling"
### I.e. identical sentences with distinct labels
### In each such case, count how many errors would be made by relabeling
### each sentence with the majority class' label.
# Confirm 2450 inconsistent labels when using only 1 word / sentence
# Check if there is also inconsistent labels with more than 1 words / sentence


# Confirm in total 4266 incorrect labels

has.1word <- rowSums(df.buyers$keywords) == 1
is.positive <- df.buyers$class == 1
is.neutral <- df.buyers$class == 2
is.negative <- df.buyers$class == 3

has.1word.positive <- df.buyers[has.1word & is.positive, "keywords"]
has.1word.neutral <- df.buyers[has.1word & is.neutral, "keywords"]
has.1word.negative <- df.buyers[has.1word & is.negative, "keywords"]

df.1wordsentence <- rbind(colSums(cbind(has.1word.positive, data.frame(class.count=1))),
                          colSums(cbind(has.1word.neutral, data.frame(class.count=1))),
                          colSums(cbind(has.1word.negative, data.frame(class.count=1))))

bwhichmax <- apply(df.1wordsentence, 2, which.max)
# In each class of df.1wordsentence, replace the class with the maximum entries by 0
zeroed.max.class <- mapply(function(col, max.ind) { col[max.ind] <- 0; return(col) },
                           split(df.1wordsentence, col(df.1wordsentence)), bwhichmax)
relabeling.mistakes <-  colSums(zeroed.max.class)[1:25]
names(relabeling.mistakes) <- colnames(df.1wordsentence[0, 1:25])
(total.1word.mistakes <- sum(relabeling.mistakes))
# Found: total of 2204 relabeling mistakes

### Check for "inconsistent labeling" with 2 or more words / sentence

num.words <- 2

has.num.words <- rowSums(df.buyers$keywords) == 2

has.num.words.positive <- df.buyers[has.num.words & is.positive, "keywords"]
has.num.words.neutral <- df.buyers[has.num.words & is.neutral, "keywords"]
has.num.words.negative <- df.buyers[has.num.words & is.negative, "keywords"]

features.names <- colnames(df.buyers[0, "keywords"])
cross.join.features <- CJ(features.names, features.names, sorted = FALSE)
cross.join.indices <- CJ(1:25, 1:25, sorted = FALSE)
cartesian.product.features <- paste0(cross.join.features$V1, ".", cross.join.features$V2)
cartesian.product.indices <- mapply(c, cross.join.indices$V1, cross.join.indices$V2)

df.buyers.matrix <- as.matrix(data.frame(df.buyers$keywords, class=as.integer(df.buyers$class)))

cartesian.num.features <- 25^num.words
empty.df <- c(rep(0, cartesian.num.features), 0)
names(empty.df) <- c(cartesian.product.features, "class")

# Transform keywords matrix to a logical matrix with all pairs of
# keywords as its columns
cartesian.has.num.words <- 
  t(apply(df.buyers.matrix[has.num.words, ], 1, function(row) {
    indices <- c()
    for (j in 1:num.words) {
      indices <- c(indices, rep(which(row[1:25] == j), j))
    }
    newrow <- empty.df
    newrow[(indices[1]-1)*25 + indices[2]] <- TRUE
    newrow["class"] <- row[26]
    newrow
  }))

cartesian.pos <- cartesian.has.num.words[cartesian.has.num.words[, "class"] == 1, 1:cartesian.num.features]
cartesian.neu <- cartesian.has.num.words[cartesian.has.num.words[, "class"] == 2, 1:cartesian.num.features]
cartesian.neg <- cartesian.has.num.words[cartesian.has.num.words[, "class"] == 3, 1:cartesian.num.features]

df.has.num.words.sentence <- rbind(colSums(cbind(cartesian.pos, data.frame(class.count=1))))
df.1wordsentence <- rbind(colSums(cbind(has.1word.positive, data.frame(class.count=1))),
                          colSums(cbind(has.1word.neutral, data.frame(class.count=1))),
                          colSums(cbind(has.1word.negative, data.frame(class.count=1))))
df.1wordsentence <- rbind(colSums(cbind(has.1word.positive, data.frame(class.count=1))),
                          colSums(cbind(has.1word.neutral, data.frame(class.count=1))),
                          colSums(cbind(has.1word.negative, data.frame(class.count=1))))

###############

library(gtools)
num.features <- 25
num.words <- 2

# Generate all combinations of 'num.words' word tuples
comb.num.features <- choose(num.features+num.words-1, num.words)
comb.words <- combinations(num.features, num.words, repeats.allowed = TRUE)
comb.words.names <- paste0("X", comb.words[,1], ".X", comb.words[,2])

stopifnot(nrow(comb.words) == comb.num.features) # assert

# Turn df.buyers into an integers matrix (including class number as an integer)
df.buyers.matrix <- as.matrix(data.frame(df.buyers$keywords, class = as.integer(df.buyers$class)))

# 'empty.row' is a helper variable for the next block of code
empty.row <- c(rep(0, comb.num.features), 0)
names(empty.row) <- c(comb.words.names, "class")

get.comb.linear.index <- function(indices) {
  linear.index <- 0
  for (a in indices) {
    # Skip
    skip.times <- a-1
    if (skip.times > 0)
      linear.index <- linear.index + sum(25 : (25 - skip.times + 1))
  }
}

# Transform keywords matrix to a matrix containing the value 0 or 1 for
# each combination of words (word tuples), indicating to what words combination
# each sentence belongs to
df.buyers.has.num.words <- rowSums(df.buyers.matrix[, 1:num.features]) == num.words
comb.has.num.words <- 
  t(apply(df.buyers.matrix[df.buyers.has.num.words, ], 1, function(row) {
    indices <- c()
    for (j in 1:num.words) {
      indices <- c(indices, rep(which(row[1:25] == j), j))
    }
    newrow <- empty.row
    newrow[(indices[1]-1)*25 + indices[2]] <- TRUE
    newrow["class"] <- row[26]
    newrow
  }))

comb.pos <- comb.has.num.words[comb.has.num.words[, "class"] == 1, 1:comb.num.features]
comb.neu <- comb.has.num.words[comb.has.num.words[, "class"] == 2, 1:comb.num.features]
comb.neg <- comb.has.num.words[comb.has.num.words[, "class"] == 3, 1:comb.num.features]

df.has.num.words.sentence <- rbind(colSums(cbind(cartesian.pos, data.frame(class.count=1))))

#######################

###
### Distance-based approach
###

# Turn df.buyers into an integers matrix (including class number as an integer)
matrix.buyers <- as.matrix(data.frame(df.buyers$keywords, class = as.integer(df.buyers$class)))

# Look at: X1 > 0 & X6 > 0 & X21 > 0
# This set of sentences is almost exclusively in the "positive" class
special.indices <- df.buyers.matrix[, "X1"] > 0 & df.buyers.matrix[, "X6"] > 0 & df.buyers.matrix[, "X21"] > 0
matrix.buyers.special <- matrix.buyers[special.indices, ]
table(matrix.buyers.special[, "class"])

# We can remove those rows from further processing
matrix.buyers <- matrix.buyers[!special.indices, ]

# Global parameter
num.features <- 25

# Loop over num.words
for (num.words in 1:4) {
  
  # Generate all cartesian n-fold products of words, with n = 'num.words'
  cart.num.features <- num.features^num.words
  cart.words <- permutations(num.features, num.words, repeats.allowed = TRUE)
  cart.words.names <- apply(cart.words, 1, function(row) paste0("X", row, collapse="."))
  
  stopifnot(nrow(cart.words) == cart.num.features) # assert
  
  # 'empty.row' is a helper variable for the next block of code
  empty.row <- c(rep(0, cart.num.features), 0)
  names(empty.row) <- c(cart.words.names, "class")
  
  # Get all sentences that have 'num.words' words in total
  indices.has.num.words <- rowSums(matrix.buyers[, 1:num.features]) == num.words
  matrix.buyers.has.num.words <- matrix.buyers[indices.has.num.words, ]
  
  # Helper function
  compute.linear.index <- function(max.dim, indices, dim = max.dim) {
    if (dim > 1) {
      return((indices[max.dim - dim + 1] - 1) * num.features^(dim - 1) +
               compute.linear.index(max.dim, indices, dim = dim - 1))
    } else {
      return(indices[max.dim])
    }
  }
  
  cart.has.num.words <- 
    t(apply(matrix.buyers.has.num.words, 1, function(row) {
      indices <- c()
      for (j in 1:num.words) {
        indices <- c(indices, rep(which(row[1:num.features] == j), j))
      }
      newrow <- empty.row
      linear.index <- compute.linear.index(num.words, indices)
      newrow[linear.index] <- TRUE
      newrow["class"] <- row[num.features + 1]
      newrow
    }))
  
  cart.pos <- cart.has.num.words[cart.has.num.words[, "class"] == 1, 1:cart.num.features]
  cart.neu <- cart.has.num.words[cart.has.num.words[, "class"] == 2, 1:cart.num.features]
  cart.neg <- cart.has.num.words[cart.has.num.words[, "class"] == 3, 1:cart.num.features]
  
  cart.all.labels <- rbind(colSums(cbind(cart.pos, data.frame(class.count=1))),
                          colSums(cbind(cart.neu, data.frame(class.count=1))),
                          colSums(cbind(cart.neg, data.frame(class.count=1))))
  rownames(cart.all.labels) <- c("positive", "neutral", "negative")
  
  # Count number of relabeling mistakes, assuming we relabel all sentences in
  # the majority class
  max.per.col <- apply(cart.all.labels, 2, max)
  mistake.per.col <- colSums(rbind(cart.all.labels, -max.per.col))
  (total.mistakes <- sum(mistake.per.col[1:cart.num.features]))

  # Print results
  cat(paste0("For sentences with exactly ", num.words, " words, relabeling ",
             "all sentences with the label of the majority class would ",
             "produce a total of ", total.mistakes, " labeling mistakes!\n"))
}

# Distance idea:
library(flexclust)

d2.mink <- dist2(matrix.buyers.has.num.words, matrix.buyers.has.num.words, method = "minkowski", p=1)
d2.mink <- (d2.mink == 0)*1


#######################

bwhichmax <- apply(df.1wordsentence, 2, which.max)
# In each class of df.1wordsentence, replace the class with the maximum entries by 0
zeroed.max.class <- mapply(function(col, max.ind) { col[max.ind] <- 0; return(col) },
                           split(df.1wordsentence, col(df.1wordsentence)), bwhichmax)
relabeling.mistakes <-  colSums(zeroed.max.class)[1:25]
names(relabeling.mistakes) <- colnames(df.1wordsentence[0, 1:25])
(total.1word.mistakes <- sum(relabeling.mistakes))
# Found: total of 2204 relabeling mistakes
for (num.words in 1:10) {
  
  has.num.words <- rowSums(df.buyers$keywords) == num.words
  
  has.num.word.positive <- df.buyers[has.num.words & is.positive, "keywords"]
  has.num.word.neutral <- df.buyers[has.num.words & is.neutral, "keywords"]
  has.num.word.negative <- df.buyers[has.num.words & is.negative, "keywords"]
  
  df.1wordsentence <- rbind(colSums(cbind(has.1word.positive, data.frame(class.count=1))),
                            colSums(cbind(has.1word.neutral, data.frame(class.count=1))),
                            colSums(cbind(has.1word.negative, data.frame(class.count=1))))
  
  bwhichmax <- apply(df.1wordsentence, 2, which.max)
  # In each class of df.1wordsentence, replace the class with the maximum entries by 0
  zeroed.max.class <- mapply(function(col, max.ind) { col[max.ind] <- 0; return(col) },
                             split(df.1wordsentence, col(df.1wordsentence)), bwhichmax)
  relabeling.mistakes <-  colSums(zeroed.max.class)[1:25]
  names(relabeling.mistakes) <- colnames(df.1wordsentence[0, 1:25])
  (total.1word.mistakes <- sum(relabeling.mistakes))
  # Found: total of 2204 relabeling mistakes
}
