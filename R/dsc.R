# Description:
# 
# Implementation of the Domain-Specific Classifier (DNC). WORK IN PROGRESS.
#
# Date: 8 July 2015
#
# Author: Philippe Paradis
#
# References:
#
# [1] H.H. Duan, V.G. Pestov, and V. Singla, "Text Categorization via Similarity
#     Search: An Efficient and Effective Novel Algorithm"

library(tm)
library(slam)

# HELPER FUNCTIONS

dsc.build.DTM <- function(texts)
{
  # Build a DocumentTermMatrix out of a collection of texts
  corpus <- Corpus(VectorSource(texts))
  docTermMatrix <- DocumentTermMatrix(corpus)
  docTermMatrix
}

dsc.build.docs.df <- function(texts)
{
  # Build a DocumentTermMatrix out of a collection of texts and
  # converts it to a data.frame
  corpus <- Corpus(VectorSource(texts))
  docTermMatrix <- DocumentTermMatrix(corpus)
  # Convert DTM into a data frame
  docs.df <- as.data.frame(as.matrix(docTermMatrix))
  # Make sure column names are words
  colnames(docs.df) <- make.names(colnames(docs.df))
  docs.df
}

dsc.build.avg.prop.matrix <- function(docs, labels)
{
  # Train the Domain-Specific Classifier.
  #
  # Args:
  #   docs: 
  #         A data.frame representing the documents (under the Vector Space 
  #         Model), i.e. each row is a document and each column is a word in the
  #         dictionary.
  #   labels:
  #           Labels corresponding to the rows of 'docs'.
  #
  # Returns:
  #   A
 
  # Compute length (i.e. total num. of words) |d_i| of each document d_i
  docs.length <- apply(docs, 1, sum)
  
  # Normalize the word counts to word frequencies by dividing each row by the
  # number of words in that row
  docs.normalized <- docs / docs.length
  
  # Loop over each label:
  # Given label j and word t, compute the formula for f_j(t):
  #
  #   f_j(t) = (1 / |D^j|) * sum_{d \in D^j} c(t, d) / |d|
  #
  # Reference: Formula (4) in [1].
  
  levels <- levels(factor(labels))
  avg.prop.matrix <- matrix(nrow = 0, ncol = ncol(docs))
  for (j in levels) {
    new.row <- (1 / sum(labels == j)) * col_sums(docs.normalized[labels == j, ])
    avg.prop.matrix <- rbind(avg.prop.matrix, new.row)
  }
  
  # Set column names to words
  colnames(avg.prop.matrix) <- colnames(docs)
  row.names(avg.prop.matrix) <- levels
  avg.prop.matrix
}

dsc.build.alpha.sums.matrix <- function(avg.prop.matrix, labels, alpha)
{
  levels <- levels(factor(labels))
  alpha.sums.matrix <- matrix(nrow=0, ncol=ncol(avg.prop.matrix))
  for (j in 1:length(levels)) {
    alpha.sums.matrix <- rbind(alpha.sums.matrix,
                               alpha * colSums(avg.prop.matrix[-j, , drop=FALSE]))
  }
  row.names(alpha.sums.matrix) <- levels
  alpha.sums.matrix
}

dsc.build.CS.matrix <- function(avg.prop.matrix, alpha.sums.matrix)
{
  # This functions computes and returns the CS = CS_j = CS_{j,alpha} matrix.
  # In other words, it builds the matrix of domain-specific words, where
  # each row corresponds to a label (category) and each value is either
  # TRUE or FALSE, meaning that the corresponding word is either domain-j
  # specific or not for the category j.
  CS <- avg.prop.matrix > alpha.sums.matrix
  row.names(CS) <- row.names(avg.prop.matrix)
  CS
}

dsc.compute.total.rel.freq.vec <- function(CS, labels, newdoc)
{
  # For each label j, compute the total relative frequency of domain j-specific
  # words found in 'newdoc'. Denote 'newdoc' by d, the label by j and the set
  # of domain j-specific words by CS_j. In other words, CS_j is the set of
  # words in 'CS' whose value is TRUE in row j.
  # 
  # The formula for the total relative frequency of domain j-specific words
  # found in d is given by:
  #
  #   w[CS_j] = (1 / |d|) * sum_{t \in CS_j} c(t, d)
  #
  # Reference: Formula (7) in [1].
  
  # TODO: Note that the (1 / |d|) factor is essentially meaningless for 
  # classification purposes. Its purpose is entirely theoretical. We might want
  # to remove it for the sake of computational efficiency (but I suspect the
  # speedup would be negligible anyway).
  
  newdoc.length <- sum(newdoc)
  levels <- levels(factor(labels))
  wCS <- c()
  for (j in 1:length(levels)) {
    newfreq <- (1 / newdoc.length) * sum(CS[j, ] * newdoc)
    wCS <- c(wCS, newfreq)
  }
  names(wCS) <- rownames(CS)
  wCS
}

dsc.compute.total.rel.freq.matrix <- function(CS, labels, newdocs)
{
  newdocs.lengths <- row_sums(newdocs)
  levels <- levels(factor(labels))
  wCS <- c()
  for (j in 1:length(levels)) {
    newfreq <- (1 / newdoc.length) * sum(CS[j, ] * newdoc)
    wCS <- c(wCS, newfreq)
  }
  names(wCS) <- rownames(CS)
  wCS
}

dsc.compute.CS.lengths.vec <- function(CS)
{
  # Compute |CS_j|, i.e. the number of domain-specific words in each
  # category j
  rowSums(CS)
}

dsc.compute.classification.ratios <- function(CS, labels, newdoc, p)
{
  # For each label j, compute the total relative frequency of domain j-specific
  # words found in 'newdoc'. Denote 'newdoc' by d, the label by j and the set
  # of domain j-specific words by CS_j. In other words, CS_j is the set of
  # words in 'CS' whose value is TRUE in row j.
  # 
  # The formula for the total relative frequency of domain j-specific words
  # found in d is given by:
  #
  #   w[CS_j] = (1 / |d|) * sum_{t \in CS_j} c(t, d)
  #
  # Reference: Formula (7) in [1].
  
  CS.lengths <- dsc.compute.CS.lengths.vec(CS)
  CS.lengths.p <- CS.lengths ^ (1/p)
  wCS <- dsc.compute.total.rel.freq.vec(CS, labels, newdoc)
  classification.ratios.vec <- wCS / CS.lengths.p
  classification.ratios.vec
}

dsc.new.document <- function(newdoc)
{
  # Normalize newdoc
  num.words <- sum(newdoc)
}

dsc <- function(x, ...)
  UseMethod("dsc")

dsc.default <- function(
    x,
    y,
    alpha,
    p,
    verbose = FALSE)
{
  # Train the Domain-Specific Classifier.
  #
  # Args:
  #   x: The training dataset.
  #   y: The labels corresponding to the dataset 'x'.
  #   alpha: The threshold parameter for determining domain-specific words. It
  #          is a real-number and must satisfy alpha >= 0. An optimal choice for
  #          alpha is typically determined through cross-vaidation using the
  #          training data.
  #   p: The normalization parameter. It is a real-number and must satisfy p > 0
  #      or p = Inf. Note that different datasets require different normalizing
  #      parameters and that the optimial nomrlization depends on the sizes of
  #      the document categories. When the categories are highly unbalanced, p =
  #      Inf should be used to avoid over-emphasizing the smaller categories;
  #      and small values of p should be used when the categories have roughly
  #      the same number of documents.
  #
  # Returns:
  #   An object with class 'dsc'.
  
  if (inherits(x, "character")) {
    text <- x
    DTM <- NULL
  } else if (inherits(x, "DocumentTermMatrix")) {
    text <- NULL
    DTM <- x
  } else if (inherits(x, "data.frame")) {
    stop("Training dataset 'x' has an unrecognized format.")
  }
  
  if (!is.factor(y)) {
    y <- factor(y)

  # Object of class 'dsc' to return
  my.model <- list(text=text,
                   labels=y,
                   alpha=alpha,
                   p=p,
                   categories=factor(levels(y)),
                   DTM=DTM)
  
  if (is.null(DTM))
    DTM <- dsc.build.DTM(my.model$text)
  apm <- dsc.build.avg.prop.matrix(DTM, my.model$labels)
  alpha.matrix <- dsc.build.alpha.sums.matrix(apm, my.model$labels,
                                              alpha = my.model$alpha)
  CS <- dsc.build.CS.matrix(apm, alpha.matrix)

  my.model$DTM <- DTM
  my.model$CS <- CS
  class(my.model) <- "dsc"
  return(my.model)
}

# c method for factors
c.factor <-  function(...)
{
  y <- do.call(c, lapply(list(...), as.character))
  factor(y, unique(unlist(lapply(list(...), levels))))
}

predict.dsc <- function(model, newdata, prob = FALSE, ...)
{
  # TODO: Documentation...
  
  if (!inherits(model, "dsc"))
    stop("method is only for dsc objects")
  
  n.train <- nrow(model$DTM)
  n.words <- ncol(model$DTM)
  
  if (inherits(newdata, "character")) {
    n.newdata <- length(newdata)
    dtm <- dsc.build.DTM(c(model$text, newdata))
    newdocs <- dtm[(n.train+1):(n.train+n.newdata), ]
  } else if (inherits(newdata, "DocumentTermMatrix") 
             || inherits(newdata, "data.frame")) {
    n.newdata <- nrow(newdata)
    if (ncol(newdata) == 1) {
      dtm <- dsc.build.DTM(c(model$text, newdata[[1]]))
      newdocs <- dtm[(n.train+1):(n.train+n.newdata), ]
    } else if (n.words == ncol(newdata)){
      newdocs <- newdata
    } else  {
      stop(paste0("Number of columns in newdata does not match expected number of words: ", n.words))
    }
  } else {
    stop("newdata type is unrecognized")
  }

  categories <- factor(levels(factor(model$labels)))
  
  newdocs.df <- as.data.frame(as.matrix(newdocs))
  predictions <- c.factor()
  for (i in 1:nrow(newdocs.df)) {
    newdoc <- newdocs.df[i, ]
    rel.freq.vec <- dsc.compute.total.rel.freq.vec(model$CS,
                                                   model$labels,
                                                   newdoc)
    ratios <- dsc.compute.classification.ratios(model$CS, model$labels,
                                               newdoc, p = model$p)
    if (prob == TRUE) {
      predictions <- c(predictions, ratios)
    } else {
      predictions <- c(predictions, categories[which.max(ratios)])
    }
  }
  predictions
}

print.dsc <- function(x, ...)
{
  # TODO: Documentation / improve this?
  cat("Domain-Specific Classifier.\n\n")
  cat("Parameters:\n")
  cat(paste0("alpha = ", x$alpha, ", p = ", x$p, "\n\n"))
  cat("Training set:\n")
  print(x$DTM)
  cat("\nMatrix of domain-specific words for all categories (CS_j):\n")
  print(x$CS)
}

summary.dsc <- function(x, ...)
{
  structure(x, class="summary.scnn")
}

print.summary.dsc <- function(x, ...)
{
  print.dsc(x)
  # TODO: ...
}

# dsc.formula <- function (formula,
#                          data = NULL,
#                          ...,
#                          subset,
#                          na.action = na.omit)
# {
#   call <- match.call()
#   if (!inherits(formula, "formula"))
#     stop("method is only for formula objects")
#   m <- match.call(expand.dots = FALSE)
#   if (identical(class(eval.parent(m$data)), "matrix"))
#     m$data <- as.data.frame(eval.parent(m$data))
#   m$... <- NULL
#   m[[1]] <- as.name("model.frame")
#   m$na.action <- na.action
#   m <- eval(m, parent.frame())
#   Terms <- attr(m, "terms")
#   attr(Terms, "intercept") <- 0
#   x <- model.matrix(Terms, m)
#   y <- model.extract(m, "response")
#   attr(x, "na.action") <- attr(y, "na.action") <- attr(m, "na.action")
#   if (length(scale) == 1)
#     scale <- rep(scale, ncol(x))
#   if (any(scale)) {
#     remove <- unique(c(which(labels(Terms) %in%
#                                names(attr(x, "contrasts"))),
#                        which(!scale)
#     )
#     )
#     scale <- !attr(x, "assign") %in% remove
#   }
#   ret <- svm.default (x, y, scale = scale, ..., na.action = na.action)
#   ret$call <- call
#   ret$call[[1]] <- as.name("svm")
#   ret$terms <- Terms
#   if (!is.null(attr(m, "na.action")))
#     ret$na.action <- attr(m, "na.action")
#   class(ret) <- c("scnn.formula", class(ret))
#   return (ret)
# }
