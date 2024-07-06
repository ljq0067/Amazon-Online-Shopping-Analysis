from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import scipy.cluster.hierarchy as hc
from wordcloud import WordCloud
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd
import os
import re

# reading in data
data = pd.read_csv('review_count.csv')

# removing labels
data = data.iloc[:, 2:]

## wordcloud
wordsForCount = []
for label in set(data.columns):
    wordcount = sum(data[label])
    for i in range(wordcount):
        wordsForCount.append(str(label))

wordsForCount = " ".join(wordsForCount)  # turning list of strings into one string
wordcloud = WordCloud(collocations=False).generate(wordsForCount)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

## using the elbow method to determine optimal k for k means clustering
SS_dist = []
values_for_k = range(2, 7)

for k_val in values_for_k:
    k_means = KMeans(n_clusters=k_val)
    model = k_means.fit(data)
    SS_dist.append(k_means.inertia_)

plt.plot(values_for_k, SS_dist, 'bx-')
plt.xlabel('value')
plt.ylabel('Sum of squared distances')
plt.title('Elbow method for optimal k Choice')
plt.show()

## using the silhouette method to confirm choice in optimal clusters for kmeans
Sih = []
Cal = []
k_range = range(2, 8)

for k in k_range:
    k_means_n = KMeans(n_clusters=k)
    model = k_means_n.fit(data)
    Pred = k_means_n.predict(data)
    labels_n = k_means_n.labels_
    R1 = metrics.silhouette_score(data, labels_n, metric='euclidean')
    R2 = metrics.calinski_harabasz_score(data, labels_n)
    Sih.append(R1)
    Cal.append(R2)

print(Sih)
print(Cal)

fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
ax1.plot(k_range, Sih)
ax1.set_title("Silhouette")
ax1.set_xlabel("")
ax2.plot(k_range, Cal)
ax2.set_title("Calinski_Harabasz_Score")
ax2.set_xlabel("k values")

plt.show()

## normalizing data
dataNormalized = (data - data.mean()) / data.std()
NumCols = dataNormalized.shape[1]
My_pca = PCA(n_components=2)
dataNormalized = np.transpose(dataNormalized)


## via stackoverflow when searching for error received
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


clean_dataset(dataNormalized)
My_pca.fit(dataNormalized)

Comps = pd.DataFrame(My_pca.components_.T,
                     columns=['PC%s' % _ for _ in range(2)],
                     index=dataNormalized.columns
                     )

k = 2
## Sklearn required you to instantiate first
kmeans = KMeans(n_clusters=k)
kmeans.fit(dataNormalized)  # run kmeans

labels = kmeans.labels_
print(labels)

centroids = kmeans.cluster_centers_
print(centroids)

prediction = kmeans.predict(dataNormalized)
(countsUni, counts) = np.unique(prediction, return_counts=True)
frequencies = np.asarray((countsUni, counts)).T
y = frequencies[:, 1]
x = frequencies[:, 0]
print(prediction)
plt.bar(x, y)
plt.show()


## function that calculates and plots kmeans for n clusters
def kmeans(X, n_clusters):
    ss = StandardScaler()
    X = ss.fit_transform(X)
    km = KMeans(n_clusters=n_clusters)
    y_pred = km.fit_predict(X)
    # plt.scatter(X[:,0], X[:,1],c=y_pred, cmap='Paired')
    plt.scatter(Comps.iloc[:, 0], Comps.iloc[:, 1], s=100, alpha=0.1)
    # plt.scatter(Comps.iloc[:,0], Comps.iloc[:,1], c=y_pred, cmap = 'Paired')
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("Scatter Plot Clusters PC 1 and 2", fontsize=15)
    plt.title("K-means")
    '''
    for i, label in enumerate(labels):
        print(i)
        print(label)
        plt.annotate(label, (Comps.iloc[i,0], Comps.iloc[i,1]))
    '''
    plt.show()


def kmeansGraph(X, n_clusters):
    ss = StandardScaler()
    X = ss.fit_transform(X)
    km = KMeans(n_clusters=n_clusters)
    km.fit(X)
    y_pred = km.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='Paired')
    plt.title("K-means")
    plt.show()


## clustering using k = 3 - optimal value determined from elbow and silhouette graphs ##
kmeansGraph(dataNormalized, 3)

## clustering using k = 2 ##
kmeans(dataNormalized, 2)

## clustering using k = 4 ##
kmeans(dataNormalized, 4)

## hierarchical clustering  and visualization ##
MyHC = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
FIT = MyHC.fit(dataNormalized)
HC_labels = MyHC.labels_
print(HC_labels)

plt.figure(figsize=(12, 12))
plt.title('Hierarchical Clustering')
dendro = hc.dendrogram((hc.linkage(dataNormalized, method='ward')))
plt.show()


## using DBSCAN##
# https://medium.com/@plog397/functions-to-plot-kmeans-hierarchical-and-dbscan-clustering-c4146ed69744
def dbscan(X, eps, min_samples):
    ss = StandardScaler()
    X = ss.fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(X)
    y_pred = db.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='Paired')
    plt.title("DBSCAN")
    plt.show()


dbscan(dataNormalized, 6, 2)
