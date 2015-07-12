# Description:
# 
# Performance tests for the Domain-Specific Classifier implemented in "dsc.R".

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
dtm <- DocumentTermMatrix(corpus)


###
### Performance test 1
###

## Setup datasets
set.seed(0)
split <- sample.split(train$class, SplitRatio=0.5)
train1 <- dtm[split==TRUE, ]
test1 <- dtm[split==FALSE, ]
train1.labels <- subset(train$class, split==TRUE)
test1.labels <- subset(train$class, split==FALSE)

## Perform classification and time functions
system.time(model.dsc <- dsc(train1, train1.labels, alpha = 2.0, p = 1))
# elapsed (old) = 2.643 seconds
# elapsed (new) = 0.083 seconds
system.time(pred.dsc <- predict(model.dsc, test1))
# elapsed (old) = 89.677 seconds
# elapsed (new) = 5.621 seconds
cm <- confusionMatrix(test1.labels, pred.dsc)
cm$overall[1] # 0.9089552 


###
### Performance test 2
###

## Setup datasets
set.seed(0)
split <- sample.split(train$class, SplitRatio=0.75)
train2 <- dtm[split==TRUE, ]
test2 <- dtm[split==FALSE, ]
train2.labels <- subset(train$class, split==TRUE)
test2.labels <- subset(train$class, split==FALSE)

## Perform classification and time functions
system.time(model.dsc <- dsc(train2, train2.labels, alpha = 2.0, p = 1))
# elapsed (old) = 3.984 seconds
# elapsed (new) = 0.116 seconds
system.time(pred.dsc <- predict(model.dsc, test2))
# elapsed (old) = 41.315 seconds
# elapsed (new) = 2.298 seconds
cm <- confusionMatrix(test2.labels, pred.dsc)
cm$overall[1] # 0.9071856

## Perform same classification again, but this time turn on profiling
Rprof("profile2.out", line.profiling=TRUE)
system.time(model.dsc <- dsc(train2, train2.labels, alpha = 2.0, p = 1))
# elapsed = 3.984 seconds
system.time(pred.dsc <- predict(model.dsc, test2))
# elapsed = 41.315 seconds
cm <- confusionMatrix(test2.labels, pred.dsc)
cm$overall[1]
Rprof(NULL)

summaryRprof("profile2.out", lines = "show")

# $by.self
# self.time self.pct total.time total.pct
# dsc.R#140         35.02    73.20      42.72     89.30
# dsc.R#141          7.70    16.10       7.70     16.10
# dsc.R#56           4.54     9.49       4.54      9.49
# <no location>      0.44     0.92       0.44      0.92
# dsc.R#72           0.08     0.17       0.08      0.17
# dsc.R#101          0.02     0.04       0.02      0.04
# dsc.R#301          0.02     0.04       0.02      0.04
# dsc.R#87           0.02     0.04       0.02      0.04
# 
# $by.total
# total.time total.pct self.time self.pct
# dsc.R#140          42.72     89.30     35.02    73.20
# dsc.R#171          42.72     89.30      0.00     0.00
# dsc.R#293          42.72     89.30      0.00     0.00
# dsc.R#141           7.70     16.10      7.70    16.10
# dsc.R#182           4.66      9.74      0.00     0.00
# dsc.R#235           4.62      9.66      0.00     0.00
# dsc.R#56            4.54      9.49      4.54     9.49
# <no location>       0.44      0.92      0.44     0.92
# dsc.R#72            0.08      0.17      0.08     0.17
# dsc.R#101           0.02      0.04      0.02     0.04
# dsc.R#301           0.02      0.04      0.02     0.04
# dsc.R#87            0.02      0.04      0.02     0.04
# dsc.R#236           0.02      0.04      0.00     0.00
# dsc.R#238           0.02      0.04      0.00     0.00
# 
# $by.line
# self.time self.pct total.time total.pct
# <no location>      0.44     0.92       0.44      0.92
# dsc.R#56           4.54     9.49       4.54      9.49
# dsc.R#72           0.08     0.17       0.08      0.17
# dsc.R#87           0.02     0.04       0.02      0.04
# dsc.R#101          0.02     0.04       0.02      0.04
# dsc.R#140         35.02    73.20      42.72     89.30
# dsc.R#141          7.70    16.10       7.70     16.10
# dsc.R#171          0.00     0.00      42.72     89.30
# dsc.R#182          0.00     0.00       4.66      9.74
# dsc.R#235          0.00     0.00       4.62      9.66
# dsc.R#236          0.00     0.00       0.02      0.04
# dsc.R#238          0.00     0.00       0.02      0.04
# dsc.R#293          0.00     0.00      42.72     89.30
# dsc.R#301          0.02     0.04       0.02      0.04
# 
# $sample.interval
# [1] 0.02
# 
# $sampling.time
# [1] 47.84

### Okay, so line 140 was the main bottleneck, by far:
###
###  newfreqs <- (1 / newdocs.lengths) * col_sums(apply(newdocs, 1, function(row) { row * CS.j }))
###
### It seems like 'apply' is fairly inefficient over such a large datasets,
### probably because of memory-intensive it is, with a new row being created and
### appended at each step. So, I replaced the line with a straight forward
### element-wise matrix multiplication:
###
###   newfreqs <- (1 / newdocs.lengths) * row_sums(newdocs * matrix(rep(CS.j, n.newdata), n.newdata, byrow = TRUE))
###
### and it is much faster now! Here is the new profiler's output:

# $by.self
# self.time self.pct total.time total.pct
# dsc.R#56           4.56    53.15       4.56     53.15
# dsc.R#143          3.42    39.86       3.42     39.86
# <no location>      0.46     5.36       0.46      5.36
# dsc.R#72           0.08     0.93       0.08      0.93
# dsc.R#142          0.02     0.23       0.02      0.23
# dsc.R#305          0.02     0.23       0.02      0.23
# dsc.R#87           0.02     0.23       0.02      0.23
# 
# $by.total
# total.time total.pct self.time self.pct
# dsc.R#186           4.66     54.31      0.00     0.00
# dsc.R#239           4.64     54.08      0.00     0.00
# dsc.R#56            4.56     53.15      4.56    53.15
# dsc.R#175           3.44     40.09      0.00     0.00
# dsc.R#297           3.44     40.09      0.00     0.00
# dsc.R#143           3.42     39.86      3.42    39.86
# <no location>       0.46      5.36      0.46     5.36
# dsc.R#72            0.08      0.93      0.08     0.93
# dsc.R#142           0.02      0.23      0.02     0.23
# dsc.R#305           0.02      0.23      0.02     0.23
# dsc.R#87            0.02      0.23      0.02     0.23
# dsc.R#240           0.02      0.23      0.00     0.00
# 
# $by.line
# self.time self.pct total.time total.pct
# <no location>      0.46     5.36       0.46      5.36
# dsc.R#56           4.56    53.15       4.56     53.15
# dsc.R#72           0.08     0.93       0.08      0.93
# dsc.R#87           0.02     0.23       0.02      0.23
# dsc.R#142          0.02     0.23       0.02      0.23
# dsc.R#143          3.42    39.86       3.42     39.86
# dsc.R#175          0.00     0.00       3.44     40.09
# dsc.R#186          0.00     0.00       4.66     54.31
# dsc.R#239          0.00     0.00       4.64     54.08
# dsc.R#240          0.00     0.00       0.02      0.23
# dsc.R#297          0.00     0.00       3.44     40.09
# dsc.R#305          0.02     0.23       0.02      0.23
# 
# $sample.interval
# [1] 0.02
# 
# $sampling.time
# [1] 8.58

### The bottleneck, line 56, is:
###
###   apply(docs, 1, sum)
###
### So I replaced it with:
###
###   row_sums(docs)
###
### Which again was much faster! The new profiling run gives:

# $by.self
# self.time self.pct total.time total.pct
# dsc.R#143          5.18    90.56       5.18     90.56
# <no location>      0.40     6.99       0.40      6.99
# dsc.R#72           0.08     1.40       0.08      1.40
# dsc.R#314          0.02     0.35       0.02      0.35
# dsc.R#56           0.02     0.35       0.02      0.35
# dsc.R#87           0.02     0.35       0.02      0.35
# 
# $by.total
# total.time total.pct self.time self.pct
# dsc.R#143           5.18     90.56      5.18    90.56
# dsc.R#175           5.18     90.56      0.00     0.00
# dsc.R#306           5.18     90.56      0.00     0.00
# <no location>       0.40      6.99      0.40     6.99
# dsc.R#186           0.12      2.10      0.00     0.00
# dsc.R#248           0.10      1.75      0.00     0.00
# dsc.R#72            0.08      1.40      0.08     1.40
# dsc.R#314           0.02      0.35      0.02     0.35
# dsc.R#56            0.02      0.35      0.02     0.35
# dsc.R#87            0.02      0.35      0.02     0.35
# dsc.R#249           0.02      0.35      0.00     0.00
# 
# $by.line
# self.time self.pct total.time total.pct
# <no location>      0.40     6.99       0.40      6.99
# dsc.R#56           0.02     0.35       0.02      0.35
# dsc.R#72           0.08     1.40       0.08      1.40
# dsc.R#87           0.02     0.35       0.02      0.35
# dsc.R#143          5.18    90.56       5.18     90.56
# dsc.R#175          0.00     0.00       5.18     90.56
# dsc.R#186          0.00     0.00       0.12      2.10
# dsc.R#248          0.00     0.00       0.10      1.75
# dsc.R#249          0.00     0.00       0.02      0.35
# dsc.R#306          0.00     0.00       5.18     90.56
# dsc.R#314          0.02     0.35       0.02      0.35
# 
# $sample.interval
# [1] 0.02
# 
# $sampling.time
# [1] 5.72