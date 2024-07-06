# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:33:28 2020

@author: profa
"""

########################################
##
## Clustering Record and Text Data
##
####################################################
## Gates
####################################################

import nltk
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import re   ## for regular expressions
from mpl_toolkits.mplot3d import Axes3D
#from nltk.stem.porter import PorterStemmer

####################################################
##
##  Clustering Text Data from a Corpus 
##
####################################################
## My data and code is here - YOURS IS DIFFERENT
## DATA LINK
# https://drive.google.com/drive/folders/1VSofcdX6g86hjnofMDQJwYVveT544Oy4?usp=sharing
path="C:/Users/profa/Documents/Python Scripts/TextMining/DATA/ClusterCorpus"

## Get the text data first
print("calling os...")
FileNameList=os.listdir(path)
## check the TYPE
print(type(FileNameList))
print(FileNameList)

##-----------
## I need an empty list to start with to build a list of complete paths to files
## Notice that I defined path above. I also need a list of file names.
ListOfCompleteFilePaths=[]
ListOfJustFileNames=[]

for name in os.listdir(path):
    ## BUILD the names dynamically....
    name=name.lower()
    print(path+ "/" + name)
    next=path+ "/" + name
    
    nextnameL=[re.findall(r'[a-z]+', name)[0]]  
    nextname=nextnameL[0]   ## Keep just the name
    print(nextname)  ## ALWAYS check yourself
    
    ListOfCompleteFilePaths.append(next)
    ListOfJustFileNames.append(nextname)

#print("DONE...")
print("full list...")
print(ListOfCompleteFilePaths)
print(ListOfJustFileNames)

####################################################
##  Create the Stemmer Function.........
######################################################
## Instantiate it
A_STEMMER=PorterStemmer()
## test it
print(A_STEMMER.stem("fishers"))
#----------------------------------------
# Use NLTK's PorterStemmer in a function - DEFINE THE FUNCTION
#-------------------------------------------------------
def MY_STEMMER(str_input):
    ## Only use letters, no punct, no nums, make lowercase...
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [A_STEMMER.stem(word) for word in words] ## Use the Stemmer...
    return words


##################################################################
## CountVectorizers be set as 'content', 'file', or 'filename'
        #If set as ‘filename’, the **sequence passed as an argument to fit**
        #is expected to be a list of filenames 
        #https://scikit-learn.org/stable/modules/generated/
        ##sklearn.feature_extraction.text.CountVectorizer.html#
        ##examples-using-sklearn-feature-extraction-text-countvectorizer

##################################################################
## Tokenize and Vectorize the text data from the corpus...
##############################################################
## Instantiate three Vectorizers.....
## NOrmal CV
MyVectCount=CountVectorizer(input='filename',
                        stop_words='english',
                        max_features=100
                        )
## Tf-idf vectorizer
MyVectTFIdf=TfidfVectorizer(input='filename',
                        stop_words='english',
                        max_features=100
                        )

## Create a CountVectorizer object that you can use with the Stemmer
MyCV_Stem = CountVectorizer(input="filename", 
                        stop_words='english', 
                        tokenizer=MY_STEMMER,
                        lowercase=True)


## NOw I can vectorize using my list of complete paths to my files
DTM_Count=MyVectCount.fit_transform(ListOfCompleteFilePaths)
DTM_TF=MyVectTFIdf.fit_transform(ListOfCompleteFilePaths)
DTM_stem=MyCV_Stem.fit_transform(ListOfCompleteFilePaths)

#####################
## Get the complete vocab - the column names
## !!!!!!!!! FOr TF and CV - but NOT for stemmed...!!!
##################
ColumnNames=MyVectCount.get_feature_names()
print("The vocab is: ", ColumnNames, "\n\n")
ColNamesStem=MyCV_Stem.get_feature_names()
print("The stemmed vocab is\n", ColNamesStem)

## Use pandas to create data frames
DF_Count=pd.DataFrame(DTM_Count.toarray(),columns=ColumnNames)
DF_TF=pd.DataFrame(DTM_TF.toarray(),columns=ColumnNames)
DF_stem=pd.DataFrame(DTM_stem.toarray(),columns=ColNamesStem)
print(DF_Count)
print(DF_TF)
print(DF_stem)

############ --------------->
## OK - now we have vectorized the data - and removed punct, numbers, etc.
## From here, we can update the names of the rows without adding labels 
## to the data.
## We CANNOT have labels in the data because:
## 1) Labels are not numeric and (2) Labels are NOT data - they are labels.
#############

## Now update the row names
MyDict={}
for i in range(0, len(ListOfJustFileNames)):
    MyDict[i] = ListOfJustFileNames[i]

print("MY DICT:", MyDict)
        
DF_Count=DF_Count.rename(MyDict, axis="index")
print(DF_Count)

DF_TF=DF_TF.rename(MyDict, axis="index")
print(DF_TF)
## That's pretty!

################################################
##           Let's Cluster........
################################################
# Using sklearn
## you will need
## from sklearn.cluster import KMeans
## import numpy as np
kmeans_object_Count = sklearn.cluster.KMeans(n_clusters=2)
#print(kmeans_object)
kmeans_object_Count.fit(DF_Count)
# Get cluster assignment labels
labels = kmeans_object_Count.labels_
prediction_kmeans = kmeans_object_Count.predict(DF_Count)
#print(labels)
print(prediction_kmeans)
# Format results as a DataFrame
Myresults = pd.DataFrame([DF_Count.index,labels]).T
print(Myresults)

############# ---> ALWAYS USE VIS! ----------
print(DF_Count)
print(DF_Count["chocolate"]) 
x=DF_Count["chocolate"]  ## col 1  starting from 0
y=DF_Count["hike"]    ## col 14  starting from 0
z=DF_Count["coffee"]  ## col 2  starting from 0
colnames=DF_Count.columns
print(colnames)
#print(x,y,z)
fig1 = plt.figure(figsize=(12, 12))
ax1 = Axes3D(fig1, rect=[0, 0, .90, 1], elev=48, azim=134)

ax1.scatter(x,y,z, cmap="RdYlGn", edgecolor='k', s=200,c=prediction_kmeans)
ax1.w_xaxis.set_ticklabels([])
ax1.w_yaxis.set_ticklabels([])
ax1.w_zaxis.set_ticklabels([])

ax1.set_xlabel('Chocolate', fontsize=25)
ax1.set_ylabel('Hike', fontsize=25)
ax1.set_zlabel('Coffee', fontsize=25)
#plt.show()
        
centers = kmeans_object_Count.cluster_centers_
print(centers)
#print(centers)
C1=centers[0,(1,2,14)]
print(C1)
C2=centers[1,(1,2,14)]
print(C2)
xs=C1[0],C2[0]
print(xs)
ys=C1[1],C2[1]
zs=C1[2],C2[2]


ax1.scatter(xs,ys,zs, c='black', s=2000, alpha=0.2)
plt.show()
#plt.cla()

#---------------- end of choc, dog, hike, example....

#########################################################
##
##     kmeans with record data - NEW DATA SETS....
##
##########################################################

##DATA
## https://drive.google.com/file/d/1QtuJO1S-03zDN4f8JgR7cZ1fA3wTZ_m4/view?usp=sharing
##and
## https://drive.google.com/file/d/1sSFzvxkp4wTbna8xAcPBCvInlA_MjNdj/view?usp=sharing

Dataset1="C:/Users/profa/Documents/Python Scripts/TextMining/DATA/ClusterSmallDataset5D.csv"
Dataset2="C:/Users/profa/Documents/Python Scripts/TextMining/DATA/ClusterSmallDataset.csv"

DF5D=pd.read_csv(Dataset1)
DF3D=pd.read_csv(Dataset2)

print(DF3D.head())
print(DF5D.head())



## !!!!!!!!!!!!! This dataset has a label
## We MUST REMOVE IT before we can proceed
TrueLabel3D=DF3D["Label"]
TrueLabel5D=DF5D["Label"]
print(TrueLabel3D)

DF3D=DF3D.drop(['Label'], axis=1) #drop Label, axis = 1 is for columns
DF5D=DF5D.drop(['Label'], axis=1)

print(DF3D.head())

kmeans_object3D = sklearn.cluster.KMeans(n_clusters=2)
kmeans_object5D = sklearn.cluster.KMeans(n_clusters=2)

#print(kmeans_object)
kmeans_3D=kmeans_object3D.fit(DF3D)
kmeans_5D=kmeans_object5D.fit(DF5D)
# Get cluster assignment labels
labels3D =kmeans_3D.labels_
labels5D =kmeans_5D.labels_

prediction_kmeans_3D = kmeans_object3D.predict(DF3D)
prediction_kmeans_5D = kmeans_object5D.predict(DF5D)

print("Prediction 3D\n")
print(prediction_kmeans_3D)
print("Actual\n")
print(TrueLabel3D)

print("Prediction 5D\n")
print(prediction_kmeans_5D)
print("Actual\n")
print(TrueLabel5D)

##---------------------
## Convert True Labels from text to numeric labels...
##-----------------------
print(TrueLabel3D)
data_classes = ["BBallPlayer", "NonPlayer"]
dc = dict(zip(data_classes, range(0,2)))
print(dc)
TrueLabel3D_num=TrueLabel3D.map(dc, na_action='ignore')
print(TrueLabel3D_num)


############# ---> ALWAYS USE VIS! ----------

fig2 = plt.figure(figsize=(12, 12))
ax2 = Axes3D(fig2, rect=[0, 0, .90, 1], elev=48, azim=134)
print(DF3D)
x=DF3D.iloc[:,0] ## Height
y=DF3D.iloc[:,1] ## Weight
z=DF3D.iloc[:,2] ## Age
print(x,y,z)

ax2.scatter(x,y,z, cmap="RdYlGn", edgecolor='k', s=200,c=prediction_kmeans_3D)
ax2.w_xaxis.set_ticklabels([])
ax2.w_yaxis.set_ticklabels([])
ax2.w_zaxis.set_ticklabels([])

ax2.set_xlabel('Height', fontsize=25)
ax2.set_ylabel('Weight', fontsize=25)
ax2.set_zlabel('Age', fontsize=25)
plt.show()

## These centers should make sense. Notice the actual values....
## The BBPlayers will be taller, higher weight, higher age     
centers3D = kmeans_3D.cluster_centers_
print(centers3D)
print(centers3D[0,0])
xs=(centers3D[0,0], centers3D[1,0])
ys=(centers3D[0,1], centers3D[1,1])
zs=(centers3D[0,2], centers3D[1,2])


ax2.scatter(xs,ys,zs, c='black', s=2000, alpha=0.2)
plt.show()

###########################################
## Looking at distances
##############################################
DF3D.head()

## Let's find the distances between each PAIR
## of vectors. What is a vector? It is a data row.
## For example:  [84       250         17]
## Where, in this case, 84 is the value for height
## 250 is weight, and 17 is age.

X=DF3D

from sklearn.metrics.pairwise import euclidean_distances
## Distance between each pair of rows (vectors)
Euc_dist=euclidean_distances(X, X)

from sklearn.metrics.pairwise import manhattan_distances
Man_dist=manhattan_distances(X,X)

from sklearn.metrics.pairwise import cosine_distances
Cos_dist=cosine_distances(X,X)

from sklearn.metrics.pairwise import cosine_similarity
Cos_Sim=cosine_similarity(X,X)

#The cosine distance is equivalent to the half the squared 
## euclidean distance if each sample is normalized to unit norm

##############-------------------------->
## Visualize distances
################################################
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print(Euc_dist)
X=DF3D
#sns.set()  #back to defaults
sns.set(font_scale=3)
Z = linkage(squareform(np.around(euclidean_distances(X), 3)))

fig4 = plt.figure(figsize=(15, 15))
ax4 = fig4.add_subplot(111)
dendrogram(Z, ax=ax4)
ax4.tick_params(axis='x', which='major', labelsize=15)
ax4.tick_params(axis='y', which='major', labelsize=15)
#ax5 = fig4.add_subplot(212)
fig4.savefig('exampleSave.png')

#######################################
## Normalizing...via scaling MIN MAX
#################################################
## For the heatmap, we must normalize first
#import pandas as pd
from sklearn import preprocessing

x = X.values #returns a numpy array
print(x)
#Instantiate the min-max scaler
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
DF3D_scaled = pd.DataFrame(x_scaled)
print(DF3D.columns)
sns.clustermap(DF3D_scaled,yticklabels=TrueLabel3D, 
               xticklabels=DF3D.columns)


###############################################
##
##   Silhouette and Elbow - Optimal Clusters...
##
#############################################
from sklearn.metrics import silhouette_samples, silhouette_score

#import pandas as pd
#import numpy as np
#import seaborn as sns
#from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score
    
## The Silhouette Method helps to determine the optimal number of clusters
    ## in kmeans clustering...
    
    #Silhouette Coefficient = (x-y)/ max(x,y)

    #where, y is the mean intra cluster distance - the mean distance 
    ## to the other instances in the same cluster. 
    ## x depicts mean nearest cluster distance i.e. the mean distance 
    ## to the instances of the next closest cluster.
    ## The coefficient varies between -1 and 1. 
    ## A value close to 1 implies that the instance is close to its 
    ## cluster is a part of the right cluster. 
    ## Whereas, a value close to -1 means that the value is 
    ## assigned to the wrong cluster.

#https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
# The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    
##
    ## This example is generated from a random mixture of normal data...
    ## ref:https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c
X= np.random.rand(100,2)
print(X)
Y= 2 + np.random.rand(100,2)
Z= np.concatenate((X,Y))
Z=pd.DataFrame(Z) 
print(Z.head())

sns.scatterplot(Z[0],Z[1])

KMean= KMeans(n_clusters=2)
KMean.fit(Z)
label=KMean.predict(Z)
print(label)

sns.scatterplot(Z[0],Z[1], hue=label)
print("Silhouette Score for k=2\n",silhouette_score(Z, label))


## Now - for k = 3
KMean= KMeans(n_clusters=3)
KMean.fit(Z)
label=KMean.predict(Z)
print("Silhouette Score for k=3\n",silhouette_score(Z, label))
sns.scatterplot(Z[0],Z[1],hue=label)

## Now - for k = 4
KMean= KMeans(n_clusters=4)
KMean.fit(Z)
label=KMean.predict(Z)
print("Silhouette Score for k=4\n",silhouette_score(Z, label))
sns.scatterplot(Z[0],Z[1],hue=label)

###############################
## Silhouette Example from sklearn
###################################################
from sklearn.datasets import make_blobs
#from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_samples, silhouette_score

#import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import numpy as np


X, y = make_blobs(n_samples=500,
                  n_features=2, ## so it is 2D
                  centers=4,
                  cluster_std=1,
                  center_box=(-10.0, 10.0),
                  shuffle=True,
                  random_state=1)  # For reproducibility

range_n_clusters = [2, 3, 4, 5, 6]

print(X)

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()


## References:
#https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Hierarchical_Clustering-Dendrograms.pdf

###### Overview of distances reference....
#'minkowski', 'cityblock', 'cosine', 'correlation',
# 'hamming', 'jaccard', 'chebyshev', 'canberra', 
## 'mahalanobis', VI=None...
## RE: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist