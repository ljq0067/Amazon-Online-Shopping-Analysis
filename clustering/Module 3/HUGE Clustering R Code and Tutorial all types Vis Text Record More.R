###
##
### Document Similarity Using Measures
##
## Gates
## ANother good resource:
## https://rstudio-pubs-static.s3.amazonaws.com/66739_c4422a1761bd4ee0b0bb8821d7780e12.html
## http://www.minerazzi.com/tutorials/cosine-similarity-tutorial.pdf
## Book: Text Mining in R
## https://www.tidytextmining.com/
######## Example 1 ----------------------
##
## Whenever you learn something new, always create a very small
## example that you can practice with. 

## I have created a small "Corpus" (collections of documents or books)
## They are called, Doc1, Doc2, ..., Doc5.
## The documents are in sentence format.

## The goal is to see how similar the documents are.

## First, we must read in the documents and convert them to 
## a format that we can evaluate.

##If you install from the source....
#Sys.setenv(NOAWT=TRUE)
## ONCE: install.packages("wordcloud")
library(wordcloud)
## ONCE: install.packages("tm")
library(tm)
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

setwd("C:\\Users\\profa\\Documents\\R\\RStudioFolder_1\\DrGExamples\\SYR\\IST707\\Week4")
## Next, load in the documents (the corpus)

### !!!!!!!!!
## Make your own corpus with 5 docs
## Make some docs similar to others so that they cluster!
##
## !!!!!!!!!!!!!!!!!!!!!!!!!!!
TheCorpus <- Corpus(DirSource("Corpus"))
##The following will show you that you read in 5 documents
(TheCorpus)

##Next, there are several steps needed to prepare the texts
## You will need to remove punctuation, make everything lowercase
## normalize, remove common and useless words like "and", "the", "or"
## Uselses words are called "Stop Words"
## Don't forget to remove numbers as well. 

## The function : getTransformations() will show all the functions
## that process the data - such as removeNumbers, removePunctuation, etc
## run getTransformations() to see this.
## Also note that tolower() will change all case to lowercase.

## The tm_map function allows you to perform the same 
## transformations on all of your texts at once
CleanCorpus <- tm_map(TheCorpus, removePunctuation)

## Remove all Stop Words
CleanCorpus <- tm_map(CleanCorpus, removeWords, stopwords("english"))

## You can also remove words that you do not want
MyStopWords <- c("and","like", "very", "can", "I", "also", "lot")
CleanCorpus <- tm_map(CleanCorpus, removeWords, MyStopWords)

## NOTE: If you have many words that you do not want to include
## you can create a file/list
## MyList <- unlist(read.table("PATH TO YOUR STOPWORD FILE", stringsAsFactors=FALSE)
## MyStopWords <- c(MyList)

##Make everything lowercase
CleanCorpus <- tm_map(CleanCorpus, content_transformer(tolower))

## Next, we can apply lemmitization
## In other words, we can combine variations on words such as
## sing, sings, singing, singer, etc.
## NOTE: This will NOT WORK for R version 3.5.x yet - so its
## just for FYI. This required package Snowball which does not yet
## run under the new version of R
#CleanCorpus <- tm_map(CleanCorpus, stemDocument)
#inspect(CleanCorpus)



## Let's see where we are so far...
inspect(CleanCorpus)
## You can use this view/information to add Stopwords and then re-run.
## In other words, I see from inspection that the word "can" is all over
## the place. But it does not mean anything. So I added it to my MyStopWords

## Next, I will write all cleaned docs  - the entire cleaned and prepped corpus
## to a file - in case I want to use it for something else.

(Cdataframe <- data.frame(text=sapply(CleanCorpus, identity), 
                        stringsAsFactors=F))
write.csv(Cdataframe, "Corpusoutput2.csv")

## Note: There are several other functions that also clean/prep text data
## stripWhitespace and
## myCorpus <- tm_map(myCorpus, content_transformer(removeURL)) 

## ------------------------------------------------------------------
## Now, we are ready to move forward.....
##-------------------------------------------------------------------

## View corpus as a document matrix
## TMD stands for Term Document Matrix
(MyTDM <- TermDocumentMatrix(CleanCorpus))
(MyDTM2 <- DocumentTermMatrix(CleanCorpus))
inspect(MyTDM)
inspect(MyDTM2)


## By inspecting this matrix, I see that the words "also" and "lot" is there, but not useful
## I will add these to my MyStopWords and will re-run the above code....
##--------------NOTE
## ABOUT DocumentTermMatrix vs. TermDocumentMatrix - yes these are NOT the same :)
##TermDocument means that the terms are on the vertical axis and the documents are 
## along the horizontal axis. DocumentTerm is the reverse

## Before we normalize, we can look at the overall frequencies of words 
## This will find words that occur 3 or more times in the entire corpus
(findFreqTerms(MyDTM2, 3))
## Find assocations via correlation
## https://www.rdocumentation.org/packages/tm/versions/0.7-6/topics/findAssocs
findAssocs(MyDTM2, 'coffee', 0.20)
findAssocs(MyDTM2, 'dog', 0.20)
findAssocs(MyDTM2, 'hiking', 0.20)

## VISUALIZE XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
## For Document Term Matrix...........
CleanDF <- as.data.frame(inspect(MyTDM))
(CleanDF)

CleanDFScale <- scale(CleanDF)
(d_TDM_E <- dist(CleanDFScale,method="euclidean"))
(d_TDM_M <- dist(CleanDFScale,method="minkowski", p=1))

## For Term Doc Matrix...................
CleanDF2 <- as.data.frame(inspect(MyDTM2))
(CleanDF2)
CleanDFScale2 <- scale(CleanDF2)

################ Distance Metrics...############
# https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/dist
###########################################################
(d2_DT_E <- dist(CleanDFScale2,method="euclidean"))
str(d2_DT_E)
(d2_DT_M2 <- dist(CleanDFScale2,method="minkowski", p=2))  ##same as Euclidean
(d2_DT_Man <- dist(CleanDFScale2,method="manhattan"))
(d2_DT_M1 <- dist(CleanDFScale2,method="minkowski", p=1)) ## same as Manhattan
(d2_DT_M4 <- dist(CleanDFScale2,method="minkowski", p=4))

#################
## Create hierarchical clustering and dendrograms......................
#https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/hclust
################

## Term Doc  - to look at words......
fit_TD1 <- hclust(d_TDM_E, method="ward.D2")
plot(fit_TD1)
fit_TD2 <- hclust(d_TDM_M, method="ward.D2")
plot(fit_TD2)

## Doc Term - to look at documents....
fit_DT1 <- hclust(d2_DT_E, method="ward.D")  
plot(fit_DT1)

fit_DT2 <- hclust(d2_DT_Man, method="average")
plot(fit_DT2)

fit_DT3 <- hclust(d2_DT_M4, method="ward.D2")
plot(fit_DT3)


## NOw I have agood matrix that allows me to see all the key words of interest 
## and their frequency in each document
## HOWEVER - I still need to normalize!
## Even though this example is very small and all docs in this example are about the
## same size, this will not always be the case. If a document has 10,000 words, it
## will easily have a greater frequency of words than a doc with 1000 words.


head(CleanDF2)
str(CleanDF2)

inspect(MyDTM2)
str(MyDTM2)

## Visualize normalized DTM
## The dendrogram:
## Terms higher in the plot appear more frequently within the corpus
## Terms grouped near to each other are more frequently found together
CleanDF_N <- as.data.frame(inspect(MyDTM2))
CleanDFScale_N <- scale(CleanDF_N)
(d <- dist(CleanDFScale_N,method="euclidean"))
fit <- hclust(d, method="ward.D2")
#rect.hclust(fit, k = 4) # cut tree into 4 clusters 
plot(fit)

###################################
## frequency Wordcloud
##################################################
inspect(MyTDM)  ## term doc (not doc term!)

m <- as.matrix(MyTDM)   ## You can use this or the next for m
(m)
##(t(m))
#m <- as.matrix(CleanDF_N)
# calculate the frequency of words and sort it by frequency
word.freq <- sort(rowSums(m), decreasing = T)
wordcloud(words = names(word.freq), freq = word.freq*2, min.freq = 2,
          random.order = F)


###############################################
##  kmeans 
##
###################################################################

#ClusterM <- t(m) # transpose the matrix to cluster documents 
#(ClusterM)
#set.seed(100) # set a fixed random seed
k <- 3 # number of clusters
#(kmeansResult <- kmeans(ClusterM, k))
(kmeansResult_tf_scaled <- kmeans(CleanDFScale_N, k))
(kmeansResult <- kmeans(MyDTM2, k)) 

#round(kmeansResult$centers, digits = 3) # cluster centers

## See the clusters  - this shows the similar documents
## This does not always work well and can also depend on the
## starting centroids
(kmeansResult$cluster)
plot(kmeansResult$cluster)


#############----------------> Silhouette with fviz
#https://www.rdocumentation.org/packages/factoextra/versions/1.0.7/topics/fviz_nbclust
(MyDF<-as.data.frame(as.matrix(MyDTM2), stringsAsFactors=False))
fviz_nbclust(MyDF, kmeans, method='silhouette', k.max=5)


inspect(MyDTM2)
library("factoextra")
(fviz_cluster(kmeansResult, data = MyDTM2,
             ellipse.type = "convex",
             #ellipse.type = "concave",
             palette = "jco",
             axes = c(1, 4), # num axes = num docs (num rows)
             ggtheme = theme_minimal()))
#, color=TRUE, shade=TRUE,
         #labels=2, lines=0)

## Let's try to find similarity using cosine similarity
## Let's look at our matrix

DT1<-MyDTM2
inspect(DT1)

DT_t <- t(MyDTM2) ## for docs
inspect(DT_t)

cosine_dist_mat1 <- 
  1 - crossprod_simple_triplet_matrix(DT1)/
  (sqrt(col_sums(DT1^2) %*% t(col_sums(DT1^2))))

(cosine_dist_mat1)

cosine_dist_mat_t <- 
  1 - crossprod_simple_triplet_matrix(DT_t)/
  (sqrt(col_sums(DT_t^2) %*% t(col_sums(DT_t^2))))

(cosine_dist_mat_t)
#heatmap https://www.rdocumentation.org/packages/stats/versions/3.5.0/topics/heatmap
## Simiarity between words
heatmap(cosine_dist_mat1)
(heatmap(cosine_dist_mat_t))

## Simiarity between docs

heatmap(t(cosine_dist_mat1),cexRow=.4, cexCol = .4 )

##############----------------------------------------
#install.packages('heatmaply')
#install.packages('yaml')
library(heatmaply)
library(htmlwidgets)
library(yaml)
mat<-t(cosine_dist_mat1)
p <- heatmaply(mat, 
               #dendrogram = "row",
               xlab = "", ylab = "", 
               main = "",
               scale = "column",
               margins = c(60,100,40,20),
               grid_color = "white",
               grid_width = 0.00001,
               titleX = FALSE,
               hide_colorbar = TRUE,
               branches_lwd = 0.1,
               label_names = c("A", "B", "C"),
               fontsize_row = 5, fontsize_col = 5,
               labCol = colnames(mat),
               labRow = rownames(mat),
               heatmap_layers = theme(axis.line=element_blank())
            )


# save the widget
# 
saveWidget(p, file= "heatmaplyExample.html")

## For the Docs
## You can use any of the distance metrics...
(mat<-as.matrix(d2_DT_E ))

p2 <- heatmaply(mat, 
               #dendrogram = "row",
               xlab = "", ylab = "", 
               main = "",
               scale = "column",
               margins = c(60,100,40,20),
               grid_color = "white",
               grid_width = 0.00001,
               titleX = FALSE,
               hide_colorbar = TRUE,
               branches_lwd = 0.2,
               label_names = c("A", "B", "C"),
               fontsize_row = 5, fontsize_col = 5,
               labCol = colnames(mat),
               labRow = rownames(mat),
               heatmap_layers = theme(axis.line=element_blank())
              )


# save the widget
# 
saveWidget(p2, file= "heatmaplyExample2.html")


#################################################################
## This is a small example of cosine similarity so you can see how it works
## I will comment it out...
######  m3 <- matrix(1:9, nrow = 3, ncol = 3)
######   (m3)
######   ((crossprod(m3))/(  sqrt(col_sums(m3^2) %*% t(col_sums(m3^2))   )))
####################################################################################
 
##########################################################3
##
##   Silhouette and Elbow - choosing k
##
#########################################################################

fviz_nbclust(
  as.matrix(MyDTM2), 
  kmeans, 
  k.max = 5,
  method = "wss",
  diss = get_dist(as.matrix(MyDTM2), method = "spearman")
)

##
## https://bradleyboehmke.github.io/HOML/kmeans.html#determine-k
#install.packages('cluster')
library(cluster)
dist_mat<-as.matrix(d2_DT_E)

silhouette_score_function <- function(k){
  km <- kmeans(MyDTM2, centers = k, nstart=25)
  ss <- cluster::silhouette(km$cluster, dist_mat)
  cat(mean(ss[, 3]))
}


k <- 2:4
avg_sil <- sapply(k, silhouette_score_function)
plot(k, type='b', avg_sil, 
     xlab='Number of clusters', ylab='Average Silhouette Scores', frame=FALSE)


#######################------------>

##############################################
##  Other Forms of CLustering.........
##
#########################################################
## HIERARCHICAL

# ----
library(dplyr)       # for data manipulation
library(ggplot2)     # for data visualization

# ---
#install.packages("cluster")
library(cluster)     # for general clustering algorithms
library(factoextra)  # for visualizing cluster results

## DATA
###https://drive.google.com/file/d/1x3PVxYAmx7CdxB6N5bF-30Z0tq9guvja/view?usp=sharing

filename="C:/Users/profa/Documents/R/RStudioFolder_1/DrGExamples/ANLY503/HeartRiskData_Outliers.csv"
HeartDF2<-read.csv(filename)
head(HeartDF2)
str(HeartDF2)
summary(HeartDF2)

HeartDF2$StressLevel<-as.ordered(HeartDF2$StressLevel)
HeartDF2$Weight<-as.numeric(HeartDF2$Weight)
HeartDF2$Height<-as.numeric(HeartDF2$Height)
str(HeartDF2)
head(HeartDF2)

## !!!!!!!!!!!!!!!!!
## You CANNOT use distance metrics on non-numeric data
## Before we can proceed - we need to REMOVE
## all non-numeric columns

HeartDF2_num <- HeartDF[,c(3,5,6)]
head(HeartDF2_num)


# Dissimilarity matrix with Euclidean
## dist in R
##  "euclidean", "maximum", "manhattan", 
## "canberra", "binary" or "minkowski" with p
(dE <- dist(HeartDF2_num, method = "euclidean"))
(dM <- dist(HeartDF2_num, method = "manhattan"))
(dMp2 <- dist(HeartDF2_num, method = "minkowski", p=2))

# Hierarchical clustering using Complete Linkage
hc_C <- hclust(dM, method = "complete" )
## RE:
#https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/hclust
# Hierarchical clustering with Ward
# ward.D2" = Ward's minimum variance method -
# however dissimilarities are **squared before clustering. 
# "single" = Nearest neighbours method. 
# "complete" = distance between two clusters is defined 
# as the maximum distance between an observation in one.
hc_D <- hclust(dE, method = "ward.D" )
hc_D2 <- hclust(dMp2, method = "ward.D2" )

######## Have a look ..........................
plot(hc_D2)
plot(hc_D)
plot(hc_C)


##################################################
##
## Which methods to use??
##
## Method with stronger clustering structures??
######################################################
library(purrr)
#install.packages("cluster")

library(cluster)

methods <- c( "average", "single", "complete", "ward")
names(methods) <- c( "average", "single", "complete", "ward")
                     

# function to compute coefficient
MethodMeasures <- function(x) {
  cluster::agnes(HeartDF2_num, method = x)$ac
}

# The agnes() function will get the agglomerative coefficient (AC), 
# which measures the amount of clustering structure found.
# Get agglomerative coefficient for each linkage method
(purrr::map_dbl(methods, MethodMeasures))
#average    single  complete      ward 
#0.9629655 0.9642673 0.9623190 0.9645178 
# We can see that single is best in this case


############################################
## More on Determining optimal clusters
#######################################################
library("factoextra")
# Look at optimal cluster numbers using silh, elbow, gap
WSS <- fviz_nbclust(HeartDF2_num, FUN = hcut, method = "wss", 
                   k.max = 5) +
  ggtitle("WSS:Elbow")
SIL <- fviz_nbclust(HeartDF2_num, FUN = hcut, method = "silhouette", 
                   k.max = 5) +
  ggtitle("Silhouette")
GAP <- fviz_nbclust(HeartDF2_num, FUN = hcut, method = "gap_stat", 
                   k.max = 5) +
  ggtitle("Gap Stat")

# Display plots side by side
gridExtra::grid.arrange(WSS, SIL, GAP, nrow = 1)

############ and ...............
library(factoextra)
file2<-"C:/Users/profa/Documents/R/RStudioFolder_1/DrGExamples/ANLY503/HeartRisk.csv"
#data
# https://drive.google.com/file/d/1pt-ouIQXH-SQzUMSqbl6Z6UWZrY3i4qu/view?usp=sharing
HeartDF_no_outliers<-read.csv(file2)
head(HeartDF_no_outliers)
## Remove non-numbers
HeartDF_no_outliers_num<-HeartDF_no_outliers[,c(3,5,6)]
head(HeartDF_no_outliers_num)

## Use a distance metric
Dist_E<-dist(HeartDF_no_outliers_num, method = "euclidean" )
fviz_dist(Dist_E)

## If we change the row numbers to labels - we can SEE the clusters...
head(HeartDF_no_outliers)
## Save the first column of labels as names....
(names<-HeartDF_no_outliers[,c(1)])
str(names)
(names<-as.character(names))
## Here is an issue - row names need to be unique. 
## So - we need to append numbers to each to make them unique...
(names<-make.unique(names, sep = ""))
## set the row names of the HeartDF_no_outliers_num as these label names...
## What are they now?
row.names(HeartDF_no_outliers_num)
## Change them-->
(.rowNamesDF(HeartDF_no_outliers_num, make.names=FALSE) <- names)
##check
row.names(HeartDF_no_outliers_num)

## OK!! Fun tricks! Now - let's cluster again....
Dist_E<-dist(HeartDF_no_outliers_num, method = "euclidean" )
fviz_dist(Dist_E)

## Better!
## NOw we can understand the clusters
## Normally - this is NOT possible
## because many datasets do not have labels

###############################################
##
##  Density Based Clustering
##  -  BDSCAN - 
##
####################################################
## Example 1: Trying to use k means for data that
## is NOT in concave clusters...this WILL NOT work....
##----------------------------
library(factoextra)
data("multishapes")
df <- multishapes[, 1:2]
set.seed(123)
km.res <- kmeans(df, 5, nstart = 25)
fviz_cluster(km.res, df, frame = FALSE, geom = "point")

## Example 2: Using Density Clustering........
#install.packages("fpc")
#install.packages("dbscan")
library(fpc)
library(dbscan)

data("multishapes", package = "factoextra")
df <- multishapes[, 1:2]
db <- fpc::dbscan(df, eps = 0.15, MinPts = 5)
# Plot DBSCAN results
plot(db, df, main = "DBSCAN", frame = FALSE)

## REF: http://www.sthda.com/english/wiki/wiki.php?id_contents=7940
