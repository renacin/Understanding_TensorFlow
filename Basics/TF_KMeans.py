# Name:                                             Renacin Matadeen
# Date:                                                01/11/2019
# Title                                 SKLearn Example - K-Means Clustering
#
#
# ----------------------------------------------------------------------------------------------------------------------
"""

GENERAL NOTES:
    + What Is K-Means Clustering?
        - It is an unsupervised machine learning technique
        - The aim of this technique is to identify clusters of data within a dataset
        - It makes use of K number of clusters

    + When Clustering, You Must:
        - You are trying to minimize some cost function, in this case the distance between clusters and data points
        - Cluster centres are randomly placed initially, however they are iteratively moved
            + Gradient Descent?
            + Cost Function?

        - Maximize the between sum of squares (BSS) (Variation between clusters)
        - Minimize the within sum of squares (WSS) (Variation within clusters)
        - There are many types of  measures to use when quantifying the distance between points, however the easiest is
            Euclidean Distance
        - Make use of an elbow chart to understand the Sum Of Squares, and what value of K returns the best results
        - Remember, the Euclidean Distance is used to measure the distance between the point, and centroid
            + Standardization, and Normalization can have a benefitial impact on variables

    + Other implementations?
        - You can use SKLearn to cluster data, or even write your own code, however TensorFlow is an inbetween that is
            customizable, yet easy to make use of

    + Code From:
        - https://www.kaggle.com/thebrownviking20/cars-k-means-clustering-script

"""
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# ----------------------------------------------------------------------------------------------------------------------

# Define Hyperparametres
num_clusters = 3
max_iteration = 1000
max_elbow = 11

# ----------------------------------------------------------------------------------------------------------------------
# Get Data
df = pd.read_csv("C:/Users/renac/Documents/Programming/Python/Tensorflow/TensorFlow_KMeans/Data/Petal_Data.csv")
data = df[["Sepal_L", "Sepal_W", "Petal_L", "Petal_W"]]

# Implement Max Score Standardization
data_x_max = data.max(axis=0)
data_x_max_list = list(data_x_max.values)
data_x = (data / data_x_max)

# Create Elbow Graph
wcss = []
for i in range(1, max_elbow):
    kmeans = KMeans(n_clusters=i, max_iter= max_iteration, n_init=10, random_state=0)
    kmeans.fit(data_x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow Graph')
plt.xlabel('Num Clusters')
plt.ylabel('WCSS')
plt.show()

# Applying K Means To Petal Dataset
kmeans = KMeans(n_clusters= num_clusters, max_iter= max_iteration, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(data_x)

# Append Data To A New DF
classified_df = data
classified_df["Classifi"] = y_kmeans
print(classified_df)

# Write To CSV
classified_df.to_csv("C:/Users/renac/Documents/Programming/Python/Tensorflow/TensorFlow_KMeans/Data/Classified_Data.csv")
