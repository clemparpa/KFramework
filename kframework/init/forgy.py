from numpy.typing import NDArray
from kframework.init.kmeans_init import RandomKmeansInit


class ForgyInit(RandomKmeansInit):
    def process_init(self, X: NDArray) -> None:
        self.model.centroids = X[
            self._random_generator.choice(len(X), self.k, replace=False)
        ]
