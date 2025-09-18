"""
Clustering Utilities for Geographic Data

This module contains clustering algorithms used for processing geographic data,
particularly for separating disconnected regions in GFED data.
"""

import numpy as np
from sklearn.cluster import DBSCAN


def points_clustering(points, eps=0.5, min_samples=3):
    """
    Clusters points using DBSCAN algorithm.

    This function is particularly useful for identifying disconnected geographic regions
    such as island nations and archipelagos in global datasets.

    Args:
        points (list of tuples): List of (lat, lon) tuples representing points.
        eps (float): The maximum distance between two samples for one to be considered
                     as in the neighborhood of the other. Default is 0.5 degrees.
        min_samples (int): The number of samples in a neighborhood for a point to be
                           considered as a core point. Default is 3.

    Returns:
        dict: A dictionary where keys are cluster labels and values are lists of points 
              in each cluster. Noise points (labeled as -1 by DBSCAN) are excluded.
    """
    points_array = np.array(points)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(points_array)

    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label == -1:  # Skip noise points
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(points[i])
    
    return clusters