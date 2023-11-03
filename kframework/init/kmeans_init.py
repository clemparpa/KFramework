from kframework.model.lloyd_centroids import LloydCentroidsAlgorithm
from numpy.typing import NDArray
from numpy import zeros
from kframework.utils.random_generator import RandomGenerator


class KmeansInit:
    def __init__(self, model: LloydCentroidsAlgorithm, k: int):
        self.k: int = k
        self.model = model

    def init_variance(self) -> None:
        self.model.variance = zeros((self.k,))

    def process_init(self, X: NDArray) -> None:
        raise NotImplementedError("KmeansInit class needs to be inherited.")

    def __call__(self, X: NDArray) -> None:
        self.init_variance()
        self.process_init(X)


class RandomKmeansInit(KmeansInit, RandomGenerator):
    def __init__(self, model: LloydCentroidsAlgorithm, k: int, random_state: int):
        KmeansInit.__init__(self, model, k)
        RandomGenerator.__init__(self, random_state)
