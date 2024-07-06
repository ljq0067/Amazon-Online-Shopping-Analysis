library(stats) 
library(NbClust)
library(cluster)
library(mclust)
library(amap)
library(factoextra)
library(purrr)
library(stylo) 
library(philentropy)
library(SnowballC)
library(caTools)
library(dplyr)
library(textstem)
library(stringr)
library(wordcloud)
library(tm)

setwd("D:/Jieqian Liu/Data Science and Analystics/ANLY501 Data Science & Analytics/Individual Project Profolio/clustering")
(df<-read.csv("category.csv"))
rownames(df) = df$name
Record_3D_Pretend = df[,-1]

library(dplyr)
Record_3D_Pretend <- Record_3D_Pretend %>%
  mutate_all(as.numeric)

str(Record_3D_Pretend)

## Distance Metric Matrices using dist
(M2_Eucl <- dist(Record_3D_Pretend,method="minkowski", p=2))  ##same as Euclidean
(M1_Man <- dist(Record_3D_Pretend,method="manhattan"))
(M4 <- dist(Record_3D_Pretend,method="minkowski", p=4))
#(CosSim<- dist(Record_3D_Pretend,method="cosine")) ## same as below
(CosSim <- stylo::dist.cosine(as.matrix(Record_3D_Pretend)))

str(M2_Eucl)

## Using a histogram to see the clusters
## and to choose k

Hist1 <- hclust(M2_Eucl, method="ward.D2")
plot(Hist1)

Hist2 <- hclust(M1_Man, method="ward.D2")
plot(Hist2)

Hist3 <- hclust(CosSim, method="ward.D2")
plot(Hist3)

Hist4 <- hclust(M4, method="ward.D2")
plot(Hist4)


##################  k - means----------------

k <- 3 # number of clusters
(kmeansResult1 <- kmeans(Record_3D_Pretend, k)) ## uses Euclidean
kmeansResult1$centers


###To use a different sim metric-- akmeans

library(akmeans)

akmeans(Record_3D_Pretend, min.k=2, max.k=4, verbose = TRUE)
##d.metric = 1  is for Euclidean else it uses Cos Sim



################ Cluster vis-------------------
(fviz_cluster(kmeansResult1, data = Record_3D_Pretend,
              ellipse.type = "convex",
              #ellipse.type = "concave",
              palette = "jco",
              #axes = c(1, 4), # num axes = num docs (num rows)
              ggtheme = theme_minimal()))


fviz_nbclust(
  as.matrix(Record_3D_Pretend), 
  kmeans, 
  k.max = 5,
  method = "wss", ##Within-Cluster-Sum of Squared Errors 
  diss = get_dist(as.matrix(Record_3D_Pretend), method = "euclidean")
)

## Silhouette........................
fviz_nbclust(Record_3D_Pretend, method = "silhouette", 
             FUN = hcut, k.max = 5)