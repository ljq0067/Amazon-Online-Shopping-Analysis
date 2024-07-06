#########################################################
##
##          Tutorial: Text Mining and NLP             
##
##           Topics:
##             - Tokenization
##             - Vectorization
##             - Normalization
##             - classification/Clustering
##             - Visualization
##
##     THE DATA CORPUS IS HERE: 
##  https://drive.google.com/drive/folders/1J_8BDiOttPvEYW4-JxrReKGP1wN40ccy?usp=sharing    
#########################################################
## Gates
#########################################################


library(tm)
#install.packages("tm")
library(stringr)
library(wordcloud)
# ONCE: install.packages("Snowball")
## NOTE Snowball is not yet available for R v 3.5.x
## So I cannot use it  - yet...
##library("Snowball")
##set working directory
## ONCE: install.packages("slam")
library(slam)
library(quanteda)
## ONCE: install.packages("quanteda")
## Note - this includes SnowballC
library(SnowballC)
library(arules)
##ONCE: install.packages('proxy')
library(proxy)
library(cluster)
library(stringi)
library(proxy)
library(Matrix)
library(tidytext) # convert DTM to DF
library(plyr) ## for adply
library(ggplot2)
library(factoextra) # for fviz
library(mclust) # for Mclust EM clustering



####### USE YOUR OWN PATH ############
setwd("D:/Jieqian Liu/Data Science and Analystics/ANLY501 Data Science & Analytics/Module2/Data Cleaning in R TEXT and Record with VIS")

## Next, load in the documents (the corpus)
NovelsCorpus <- Corpus(DirSource("Novels_Corpus"))
(getTransformations()) ## These work with library tm
(ndocs<-length(NovelsCorpus))


##The following will show you that you read in all the documents
(summary(NovelsCorpus))  ## This will list the docs in the corpus
(meta(NovelsCorpus[[1]])) ## meta data are data hidden within a doc - like id
(meta(NovelsCorpus[[1]],5))

###################################################################
#######       Change the COrpus into a DTM, a DF, and  Matrix
#######
####################################################################
## There are OPTIONS. This is NOT what you should do - but rather
## things you can do, consider, and learn more about.

# You can ignore extremely rare words i.e. terms that appear in less
# then 1% of the documents. The following is an EXAMPLE not a set method
##(minTermFreq <- ndocs * 0.01) ## Because we only have 13 docs - this will not matter
# You can ignore overly common words i.e. terms that appear in more than
## 50% of the documents
##(maxTermFreq <- ndocs * .50)

## You can create your own Stopwords
## A Wordcloud is good to determine
## if there are odd words you want to remove
#(STOPS <- c("aaron","maggi", "maggie", "philip", "tom", "glegg", "deane", "stephen","tulliver"))

Novels_dtm <- DocumentTermMatrix(NovelsCorpus,
                         control = list(
                           #stopwords = TRUE, ## remove normal stopwords
                           wordLengths=c(4, 10), ## get rid of words of len 3 or smaller or larger than 15
                           removePunctuation = TRUE,
                           removeNumbers = TRUE,
                           tolower=TRUE,
                           #stemming = TRUE,
                           remove_separators = TRUE
                           #stopwords = MyStopwords,
                
                           #removeWords(MyStopwords),
                           #bounds = list(global = c(minTermFreq, maxTermFreq))
                         ))
########################################################
################### Have a look #######################
################## and create formats #################
########################################################
#(inspect(Novels_dtm))  ## This takes a look at a subset - a peak
DTM_mat <- as.matrix(Novels_dtm)
(DTM_mat[1:13,1:10])

#########################################################
######### OK - Pause - now the data is vectorized ######
## Its current formats are:
## (1) Novels_dtm is a DocumentTermMatrix R object
## (2) DTM_mat is a matrix
#########################################################

#Novels_dtm <- weightTfIdf(Novels_dtm, normalize = TRUE)
#Novels_dtm <- weightTfIdf(Novels_dtm, normalize = FALSE)

## Look at word freuqncies out of interest
(WordFreq <- colSums(as.matrix(Novels_dtm)))

(head(WordFreq))
(length(WordFreq))
ord <- order(WordFreq)
(WordFreq[head(ord)])
(WordFreq[tail(ord)])
## Row Sums
(Row_Sum_Per_doc <- rowSums((as.matrix(Novels_dtm))))

## I want to divide each element in each row by the sum of the elements
## in that row. I will test this on a small matrix first to make 
## sure that it is doing what I want. YOU should always test ideas
## on small cases.
#############################################################
########### Creating and testing a small function ###########
#############################################################
## Create a small pretend matrix
## Using 1 in apply does rows, using a 2 does columns
(mymat = matrix(1:12,3,4))
freqs2 <- apply(mymat, 1, function(i) i/sum(i))  ## this normalizes
## Oddly, this re-organizes the matrix - so I need to transpose back
(t(freqs2))
## OK - so this works. Now I can use this to control the normalization of
## my matrix...
#############################################################

## Copy of a matrix format of the data
Novels_M <- as.matrix(Novels_dtm)
(Novels_M[1:13,1:5])

## Normalized Matrix of the data
Novels_M_N1 <- apply(Novels_M, 1, function(i) round(i/sum(i),3))
(Novels_M_N1[1:13,1:5])
## NOTICE: Applying this function flips the data...see above.
## So, we need to TRANSPOSE IT (flip it back)  The "t" means transpose
Novels_Matrix_Norm <- t(Novels_M_N1)
(Novels_Matrix_Norm[1:13,1:10])

############## Always look at what you have created ##################
## Have a look at the original and the norm to make sure
(Novels_M[1:13,1:10])
(Novels_Matrix_Norm[1:13,1:10])

######################### NOTE #####################################
## WHen you make calculations - always check your work....
## Sometimes it is better to normalize your own matrix so that
## YOU have control over the normalization. For example
## scale used diectly may not work - why?

##################################################################
###############   Convert to dataframe     #######################
##################################################################

## It is important to be able to convert between format.
## Different models require or use different formats.
## First - you can convert a DTM object into a DF...
(inspect(Novels_dtm))
Novels_DF <- as.data.frame(as.matrix(Novels_dtm))
#(head(Novels_DF))
str(Novels_DF)
(Novels_DF$aunt)
(nrow(Novels_DF))  ## Each row is a novel
## Fox DF format

######### Next - you can convert a matrix (or normalized matrix) to a DF
Novels_DF_From_Matrix_N<-as.data.frame(Novels_Matrix_Norm)

#######################################################################
#############   Making Word Clouds ####################################
#######################################################################
## This requires a matrix - I will use Novels_M from above. 
## It is NOT mornalized as I want the frequency counts!
## Let's look at the matrix first
(Novels_M[c(1:13),c(3850:3900)])
wordcloud(colnames(Novels_M), Novels_M[13, ], max.words = 100)

############### Look at most frequent words by sorting ###############
(head(sort(Novels_M[13,], decreasing = TRUE), n=20))

#######################################################################
##############        Distance Measures          ######################
#######################################################################
## Each row of data is a novel in this case
## The data in each row are the number of time that each word occurs
## The words are the columns
## So, distances can be measured between each pair of rows (or each novel)
## We can determine which novels (rows of numeric word frequencies) are "closer" 
########################################################################
## 1) I need a matrix format
## 2) I will use the matrix above that I created and normalized:
##    Novels_Matrix_Norm
## Let's look at it
(Novels_Matrix_Norm[c(1:6),c(3850:3900)])
## 3) For fun, let's also do this for a non-normalized matrix
##    I will use Novels_M from above
## Let's look at it
(Novels_M[c(1:6),c(3850:3900)])

## I am going to make copies here. 
m  <- Novels_M
m_norm <-Novels_Matrix_Norm
(str(m_norm))

###############################################################################
################# Build distance MEASURES using the dist function #############
###############################################################################
## Make sure these distance matrices make sense.
distMatrix_E <- dist(m, method="euclidean")
print(distMatrix_E)
distMatrix_C <- dist(m, method="cosine")
print("cos sim matrix is :\n")
print(distMatrix_C)
print("L2 matrix is :\n")
print(distMatrix_E)
distMatrix_C_norm <- dist(m_norm, method="cosine")
print("The norm cos sim matrix is :\n")
print(distMatrix_C_norm)
###########################################################################

############# Clustering #############################
## Hierarchical

## Euclidean
groups_E <- hclust(distMatrix_E,method="ward.D")
plot(groups_E, cex=0.9, hang=-1)
rect.hclust(groups_E, k=4)

## Cosine Similarity
groups_C <- hclust(distMatrix_C,method="ward.D")
plot(groups_C, cex=0.9, hang=-1)
rect.hclust(groups_C, k=4)

## Cosine Similarity for Normalized Matrix
groups_C_n <- hclust(distMatrix_C_norm,method="ward.D")
plot(groups_C_n, cex=0.9, hang=-1)
rect.hclust(groups_C_n, k=4)

### NOTES: Cosine Sim works the best. Norm and not norm is about
## the same because the size of the novels are not sig diff.

####################   k means clustering -----------------------------
## Remember that kmeans uses a matrix of ONLY NUMBERS
## We have this so we are OK.
## Manhattan gives the best vis results!

distance0 <- get_dist(m_norm,method = "euclidean")
fviz_dist(distance0, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))
distance1 <- get_dist(m_norm,method = "manhattan")
fviz_dist(distance1, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))
distance2 <- get_dist(m_norm,method = "pearson")
fviz_dist(distance2, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))
distance3 <- get_dist(m_norm,method = "canberra")
fviz_dist(distance3, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))
distance4 <- get_dist(m_norm,method = "spearman")
fviz_dist(distance4, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))

## Next, our current matrix does NOT have the columns as the docs
## so we need to transpose it first....
## Run the following twice...
(nrow(m))
(ncol(m))
#str(m_norm)
## k means
kmeansFIT_1 <- kmeans(m,centers=4)
#(kmeansFIT_1)
#print("Kmeans details:")
(summary(kmeansFIT_1))
(kmeansFIT_1$cluster)
#(kmeansFIT_1$centers)

###############NOTE
## One issue here is that kmeans does not
## allow us to use cosine sim
## This is creating results that are not as good. 
####################

### This is a cluster vis
fviz_cluster(kmeansFIT_1, m)
## --------------------------------------------
#########################################################

################# Expectation Maximization ---------
## When Clustering, there are many options. 
## I cannot run this as it requires more than 18 Gigs...

#ClusFI <- Mclust(X,G=6)
#(ClusFI)
#summary(ClusFI)
#plot(ClusFI, what = "classification")


########### Frequencies and Associations ###################

## FInd frequenct words...
(findFreqTerms(Novels_dtm, 2500))
## Find assocations with aselected conf
(findAssocs(Novels_dtm, 'aunt', 0.95))

##############  NOTE ############################
## The following is an alternative method
## This code can take a long time to run.
## It is commented out for now.
#################################################

##Next, there are several steps needed to prepare the texts
## You will need to remove punctuation, make everything lowercase
## normalize, remove common and useless words like "and", "the", "or"
## Uselses words are called "Stopwords"
## Don't forget to remove numbers as well. 

## The function : getTransformations() will show all the functions
## that process the data - such as removeNumbers, removePunctuation, etc
## run getTransformations() to see this.
## Also note that tolower() will change all case to lowercase.

## The tm_map function allows you to perform the same 
## transformations on all of your texts at once
#tm_corpus <- tm_map(tm_corpus, (meta(NovelsCorpus[[1]],5)))
#CleanNovelsCorpus <- tm_map(NovelsCorpus, content_transformer(removePunctuation))
#(meta(CleanNovelsCorpus[[1]]))  ## Now the metadata is gone - the id is lost

## Remove all Stop Words
#CleanNovelsCorpus <- tm_map(CleanNovelsCorpus, removeWords, stopwords("english"))

## You can also remove words that you do not want
#MyStopWords <- c("like", "very", "can", "I", "also", "lot")
#CleanNovelsCorpus <- tm_map(CleanNovelsCorpus, removeWords, MyStopWords)
## NOTE: If you have many words that you do not want to include
## you can create a file/list
## MyList <- unlist(read.table("PATH TO YOUR STOPWORD FILE", stringsAsFactors=FALSE)
## MyStopWords <- c(MyList)

##Make everything lowercase
#CleanNovelsCorpus <- tm_map(CleanNovelsCorpus, content_transformer(tolower))
## Next, we can apply lemmitization
## In other words, we can combine variations on words such as
## sing, sings, singing, singer, etc.
## I will not do this - but it is an option
#CleanNovelsCorpus <- tm_map(CleanNovelsCorpus, stemDocument)
#inspect(CleanNovelsCorpus)

## Let's see where we are so far...
## This will be large - so I am commenting it out.
## It is a good idea to inspect it once. 
## WHen you do- you will see many \n that you may
## have to deal with...
#inspect(CleanNovelsCorpus)

## You can use this view/information to add Stopwords and then re-run.
## In other words, I see from inspection that the word "can" is all over
## the place. But it does not mean anything. So I added it to my MyStopWords

## Next, I will write all cleaned docs  - the entire cleaned and prepped corpus
## to a file - in case I want to use it for something else.

## This will be commented out unless it is needed....
#(Novelsdataframe <- data.frame(text=sapply(CleanNovelsCorpus, identity), 
 #                         stringsAsFactors=F))
#write.csv(Novelsdataframe, "NovelsCorpusoutput.csv")

## Note: There are several other functions that also clean/prep text data
## stripWhitespace and
## myCorpus <- tm_map(myCorpus, content_transformer(removeURL)) 

###################################################################
######################### NEXT STEPS ##############################
###################################################################

## After you complete the above, you do not need to run those lines
## again, as they take a long time.

## The next steps are to tokenize the documents and vectorize
## each into record data such that the words are the variables
## (column names)

## Make the Term Document Matrix
## TMD stands for Term Document Matrix
#(Novels_TDM <- TermDocumentMatrix(CleanNovelsCorpus))
#inspect(Novels_TDM)
## In the DTM - doc term matrix, the words are the vars
#(Novels_DocTM <- DocumentTermMatrix(CleanNovelsCorpus))
#inspect(Novels_DocTM)
## FInd frequenct words...
#(findFreqTerms(Novels_DocTM, 1000))
## Find assocations with aselected conf
#(findAssocs(Novels_DocTM, 'world', 0.60))

## VISUALIZE
#Novels_DocTM_DF <- as.data.frame(inspect(Novels_DocTM))
#NovelsDFScale <- scale(Novels_DocTM_DF) # normalize
#d <- dist(NovelsDFScale,method="euclidean")
#d <- dist(NovelsDFScale,method="cosine")
#fit <- hclust(d, method="ward.D2")
#plot(fit)

