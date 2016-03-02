# Pre Process Ice Cream Data for LDA

icecream <- as.vector(data[-removeIndex, "FLAVOR"])

# Pre-processing
# Pre-processing
# read in some stopwords:
library(tm)
stop_words <- stopwords("SMART")

# pre-processing:
icecream <- gsub("'", "", icecream)  # remove apostrophes
icecream <- gsub("[[:punct:]]", " ", icecream)  # replace punctuation with space
icecream <- gsub("[[:cntrl:]]", " ", icecream)  # replace control characters with space
icecream <- gsub("^[[:space:]]+", "", icecream) # remove whitespace at beginning of documents
icecream <- gsub("[[:space:]]+$", "", icecream) # remove whitespace at end of documents
icecream <- tolower(icecream)  # force to lowercase

# tokenize on space and output as a list:
doc.list <- strsplit(icecream, "[[:space:]]+")

# compute the table of terms:
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)

# remove terms that are stop words or occur fewer than 15 times:
del <- names(term.table) %in% stop_words | term.table < 15
term.table <- term.table[!del]
vocab <- names(term.table)

# now put the documents into the format required by the lda package:
get.terms <- function(x) {
        index <- match(x, vocab)
        index <- index[!is.na(index)]
        rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)

# Compute some statistics related to the data set:
D <- length(documents)  # number of documents (51,994)
W <- length(vocab)  # number of terms in the vocab (341)
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document
term.frequency <- as.integer(term.table)  

#####################################
# BACK TO TOP
# remove data with no entries
# make sure to get doc.length with full data
removeIndex <- which(doc.length == 0)
#####################################

# Tuning the model
# MCMC and model tuning parameters:
K <- 15
G <- 5000
alpha <- 0.02
eta <- 0.02

# Fit the model:
library(lda)
set.seed(357)
t1 <- Sys.time()
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, 
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)
t2 <- Sys.time()
t2 - t1  # about 13 minutes on laptop

# Visualize the fitted model with LDAvis
theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))

IceCreams <- list(phi = phi,
                     theta = theta,
                     doc.length = doc.length,
                     vocab = vocab,
                     term.frequency = term.frequency)

library(LDAvis)
# create the JSON object to feed the visualization:
json <- createJSON(phi = IceCreams$phi, 
                   theta = IceCreams$theta, 
                   doc.length = IceCreams$doc.length, 
                   vocab = IceCreams$vocab, 
                   term.frequency = IceCreams$term.frequency)

serVis(json, out.dir = 'vis15', open.browser = TRUE)