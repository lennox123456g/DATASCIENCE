# Machine learning methods can generally be classified into two main categories of models:
# supervised learning and unsupervised learning. Thus far, we have been working on
# supervised learning models, since we train our models with a target y or response variable.
# In other words, in the training data for our models, we know the “correct” answer.
# Unsupervised models are modeling techniques in which the “correct” answer is unknown.
# Many of these methods involve clustering, where the two main methods are k-means
# clustering and hierarchical clustering.

#K-MEANS

# The technique known as k-means works by first selecting how many clusters, k, exist in
# the data. The algorithm randomly selects k points in the data and calculates the distance
# from every data point to the initially selected k points. The closest points to each of the k
# clusters are assigned to the same cluster group. The center of each cluster is then
# designated as the new cluster centroid. The process is then repeated, with the distance of
# each point to each cluster centroid being calculated and assigned to a cluster and a new
# centroid picked. This algorithm is repeated until convergence occurs.
# Great visualizations1 and explanations2 of how k-means works can be found on the
# Internet. We’ll use data about wines for our k-means example.

import pandas as pd 
wine = pd.read_csv('data/wine.csv')

# Wewill drop the Cultivar column since it correlates too closely with the actual clusters
# in our data.

wine = wine.drop('Ciltivar', axis=1)

 # note that the data values are all numeric
 print(wine.columns)

print(wine.head())


# sklearnhasanimplementationofthek-meansalgorithmcalledKMeans.Herewewill
# setk=3,anduseall thedatainourdataset.
# Wewillcreatek=3clusterswitharandomseedof42.Youcanopttoleaveoutthe
# random_stateparameteroruseadifferentvalue; the42willensureyourresultsarethe
# sameasthoseprintedinthebook

from sklearn.cluster import KMeans
kmeans  = KMeans(n_clusters=3, random_state=42).fit(wine.values)
print(kmeans)

 import numpy as np
 print(np.unique(kmeans.labels_, return_counts=True))

kmeans_3 = pd.DataFrame(kmeans.labels_, columns=['cluster'])
print(kmeans_3)

# We can see that since we specified three clusters, there are only three unique labels

import numpy as np
print(np.unique(kmeans.labels_, return_counts=True))

kmeans_3= pd.DataFrame(kmeans.labels_, columns=['cluster'])
print(kmeans_3)

# Finally, we can visualize our clusters. Since humans can visualize things in only three
# dimensions, we need to reduce the number of dimensions for our data. Our wine data set
# has 13 columns, and we need to reduce this number to three so we can understand what is
# going on. Furthermore, since we are trying to plot the points in a book (a non-interactive
# medium), we should reduce the number of dimensions to two, if possible

#DIMENSION REDUCTION WITH PCA 

#principal components analysis is a projection technique that is used to reduce the number of dimensions for a dataset
#It works by finding a lower dimension in the data such that the variance is a maximised
#.Imagine a three-dimensional sphere of points. PCA
# essentially shines a light through these points and casts a shadow in the lower
# two-dimensional plane. Ideally, the shadows will be spread out as much as possible. While
# points that are far apart in PCA may not be cause for concern, points that are far apart in
# the original 3D sphere can have the light shine through them in such a way that the
# shadows cast are right next to one another. Be careful when trying to interpret points that
# are close to one another because it is possible that these points could be farther apart in the
# original space.
# We import PCA from sklearn.

from sklearn.decomposition import PCA

# We tell PCA how many dimensions (i.e., principal components) we want to project our
# data into. Here we are projecting our data down into two components.

#project our data into two components 
pca = PCA(n_components=2).fit(wine)

# Next, we need to transform our data into the new space and add the transformation to
# our data set.

pca_trans = pca.transform(wine)


#give our projection a name
pca_trans_df = pd.DataFrame(pca_trans, columns=['pca1', 'pca2'])

#concatenate our data 
kmeans_3 =pd.concat([kmeans_3, pca_trans_df], axis=1)

print(kmeans_3)


#now we can plot our results
import seaborn as sns
import matplotlib.pyplot as plt 
fig, ax = plt.subplots()

sns.scatterplot(
    x="pca1"
    y="pca2"
    data=kmeans_3hue="cluster",
    ax=ax
)

plt.show()

# Nowthatwe’veseenwhatk-meansdoestoourwinedata, let’s loadtheoriginaldata
# setagainandkeeptheCultivarcolumnwedropped.

wine_all = pd.read_csv('data/wine.csv')
print(wine_all.head())

#We’llrunPCAonourdata, justasbefore,andcomparetheclustersfromPCAandthe
# variablesfromCultivar.

pca_all = PCA(N_components=2).fit(wine_all)
pca_all_trans = pca_all.transform(wine_all)
pca_all_trans_df = pd.DataFrame(
    pca_all_trans, columns=["pca_all_1", "pca_all_2"]
)

kmeans_3=pd.concat(
  [kmeans_3,pca_all_trans_df,wine_all["Cultivar"]],axis=1
)

#Wecancomparethegroupingsbyfacetingourplot

 withsns.plotting_context(context="talk"):
 fig=sns.relplot(
    x="pca_all_1",
    y="pca_all_2",
    data=kmeans_3,
    row="cluster",
    col="Cultivar",
 )

 fig.figure.set_tight_layout(True)
 plt.show()

 # Alternatively,wecanlookatacross-tabulatedfrequencycount.
print(
    pd.crosstab(
        kmeans_3["cluster"],kmeans_3["Cultivar"],margins=True
    )
 )

 #HEIRARCHICAL CLUSTERING 
#  As the name suggests, hierarchical clustering aims to build a hierarchy of clusters. It can
# accomplish this with a bottom-up (agglomerative) or top-town (decisive) approach.
# We can perform this type of clustering with the scipy library.

from scipy.cluster import hierarchy

# We’ll load up a clean wine data set again, and drop the Cultivar column.

wine = pd.read_csv('data/wine.csv')
wine = wine.drop('Cultivar', axis=1)

# Many different formulations of the hierarchical clustering algorithm are possible. We
# can use matplotlib to plot the results.

import matplotlib.pyplot as plt

# Below we will cover a few clustering algorithms, they all work slightly differently, but
# they can lead to different results.
# . Complete: Tries to make the clusters as similar to one another as possible
# . Single: Creates looser and closer clusters by linking as many of them as possible
# . Average and Centroid: Some combination between complete and single
# . Ward: Minimizes the distance between the points within each cluster


#COMPLETE CLUSTERING 
# A hierarchical cluster using the complete clustering algorithm is shown in Figure 18.3.

wine_complete = hierarchy.complete(wine)
fig = plt.figure()
dn = hierarchy.dendrogram(wine_complete)
plt.show()

#SINGLE CLUSTERING
wine_complete = hierarchy.single(wine)
fig = plt.figure()
dn = hierarchy.dendrogram(wine_single)
plt.show()

#AVERAGE CLUSTERING 
wine_complete = hierarchy.average(wine)
fig = plt.figure()
dn = hierarchy.dendrogram(wine_average)
plt.show()

#CENTROID CLUSTERING 
wine_complete = hierarchy.centroid(wine)
fig = plt.figure()
dn = hierarchy.dendrogram(wine_centroid)
plt.show()

#WARD CLUSTERING 
wine_complete = hierarchy.ward(wine)
fig = plt.figure()
dn = hierarchy.dendrogram(wine_ward)
plt.show()

#MANUALLY SETTING THE THRESHOLD
# We can pass in a value for color_threshold to color the groups based on a specific
# threshold (Figure 18.8). By default, scipy uses the default MATLAB values.

wine_complete = hierarchy.complete(wine)
fig = plt.figure()
dn = hierarchy.dendrogram(
    wine_complete,
    #default MATLAB threshhold
    color_threshold=0.7 * max(wine_complete[:, 2]),
    above_threshold_color='y'
)
plt.show()

# When you are trying to find the underlying structure in a data set, you will often use
# unsupervised machine learning methods. k-Means and hierarchical clustering are two
# methods commonly used to solve this problem. The key is to tune your models either by
# specifying a value for k in k-means or a threshold value in hierarchical clustering that
# makes sense for the question you are trying to answer.
# It is also common practice to mix multiple types of analysis techniques to solve a
# problem. For example, you might use an unsupervised learning method to cluster your
# data and then use these clusters as features in another analysis method.
