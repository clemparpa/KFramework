from numpy.typing import NDArray
from kframework.init.fcm_init import RandomFCMInit
from kframework.model.fcm import FCMAlgorithm
from numpy import sum as np_sum, newaxis


class RandomTagsInit(RandomFCMInit):
    def __init__(
        self,
        model: FCMAlgorithm,
        k: int,
        normalize: bool = True,
        random_state: int = 42,
    ):
        super().__init__(model, k, random_state)
        self.normalize = normalize

    def process_init(self, X: NDArray) -> None:
        tags = self._random_generator.uniform(size=(len(X), self.k))  # type: ignore
        if self.normalize:
            tags /= np_sum(tags, axis=1)[:, newaxis]
        self.model.tags = tags
        self.model.assign_centroids(X)
