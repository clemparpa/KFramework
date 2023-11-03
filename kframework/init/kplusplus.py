from kframework.init.kmeans_init import RandomKmeansInit
from numpy.typing import NDArray
from numpy import transpose, expand_dims, sum as np_sum, isin


class KplusplusInit(RandomKmeansInit):
    def process_init(self, X: NDArray) -> None:
        center_indexes = [self._random_generator.choice(len(X), 1)[0]]
        for _i in range(2, self.k + 1):
            distances = transpose(
                self.model.distance(X, expand_dims(X[center_indexes], 1)) ** 2
            )
            proba = (distances / np_sum(distances, axis=0)).min(axis=1).squeeze()
            proba = proba / np_sum(proba)
            picks = self._random_generator.random.choice(
                len(X), size=len(center_indexes) + 1, p=proba, replace=False
            )
            center_indexes.append(picks[isin(picks, center_indexes, invert=True)][0])

        self.model.centroids = X[center_indexes]
