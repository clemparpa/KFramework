from kframework.model.lloyd_centroids import LloydCentroidsAlgorithm
from kframework.distance.minkowski import MinkowskiDistance
from numpy import median


class LloydKMedians(LloydCentroidsAlgorithm):
    def __init__(self):
        super().__init__(MinkowskiDistance(1), median)
