from typing import Any

from numpy.typing import NDArray
from kframework.distance.minkowski import MinkowskiDistance
from numpy import array, expand_dims, argmin, unique, sum as np_sum


class LloydCentroidsAlgorithm:
    def __init__(self, distance: MinkowskiDistance, reducer: Any):
        self.distance = distance
        self.reducer = reducer
        self.centroids: NDArray
        self.clusters: NDArray
        self.step_counter = array(0)
        self.variance: NDArray = array(0)

    def assign_clusters(self, X: NDArray):
        self.clusters = argmin(self.distance(X, expand_dims(self.centroids, 1)), axis=0)

    def update_centroids(self, X: NDArray):
        for cluster in unique(self.clusters):
            cluster_data = X[self.clusters == cluster]
            if len(cluster_data) > 0:
                self.centroids[cluster] = self.reducer(
                    X[self.clusters == cluster], axis=0
                )

    def compute_variation(self, X: NDArray):
        for cluster, centroid in enumerate(self.centroids):
            cluster_data = X[self.clusters == cluster]
            cluster_count = len(cluster_data) > 0
            if cluster_count > 0:
                self.variance[cluster] = (
                    np_sum(self.distance(cluster_data, centroid)) / cluster_count
                )

    def step(self, X: NDArray):
        self.assign_clusters(X)
        self.update_centroids(X)
        self.compute_variation(X)
        self.step_counter += 1
