from numpy.typing import NDArray
from kframework.distance.minkowski import MinkowskiDistance
from numpy import array, sum as np_sum, newaxis, tile, expand_dims


class FCMAlgorithm:
    def __init__(self, fuzziness: float):
        self.m: float = fuzziness
        self.distance = MinkowskiDistance(2)
        self.centroids: NDArray  # size: k x r
        self.tags: NDArray  # size: n x k
        self.step_counter = array(0)

    def assign_centroids(self, X: NDArray):
        tag_pow_m = self.tags**self.m
        self.centroids = ((X.T @ tag_pow_m) / np_sum(tag_pow_m, axis=0)).T

    def update_tags(self, X: NDArray):
        distances: NDArray = self.distance(X, expand_dims(self.centroids, 1)).T  # type: ignore
        numerator = tile(distances[:, :, newaxis], (1, 1, 3))
        denominator = tile(distances[:, newaxis, :], (1, 3, 1))
        self.tags = 1 / np_sum((numerator / denominator) ** (2 / (self.m - 1)), axis=2)

    def step(self, X: NDArray):
        self.assign_centroids(X)
        self.update_tags(X)
        self.step_counter += 1
