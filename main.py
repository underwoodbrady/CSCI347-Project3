#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

def data_mining_function(dataset):
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=3)
    kmeans_labels = kmeans.fit_predict(dataset)

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(dataset)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(dataset)

    # Scatter plot using PCA transformed data
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=kmeans_labels, cmap='viridis', marker='o', s=50, label='K-means')
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=dbscan_labels, cmap='plasma', marker='x', s=50, label='DBSCAN')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.show()

# Example usage
dataset = pd.read_csv('Raisin_Dataset.csv')
data_mining_function(dataset)