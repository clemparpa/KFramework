from kframework.init.kmeans_init import RandomKmeansInit
from numpy.typing import NDArray


class RandomPartitionInit(RandomKmeansInit):
    def process_init(self, X: NDArray) -> None:
        self.model.clusters = self._random_generator.choice(self.k, len(X))
        self.model.update_centroids(X)
