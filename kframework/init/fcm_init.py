from kframework.model.fcm import FCMAlgorithm
from kframework.utils.random_generator import RandomGenerator
from numpy.typing import NDArray


class FCMInit:
    def __init__(self, model: FCMAlgorithm, k: int):
        self.k: int = k
        self.model = model

    def process_init(self, X: NDArray) -> None:
        raise NotImplementedError("FCMInit class needs to be inherited.")

    def __call__(self, X: NDArray) -> None:
        self.process_init(X)


class RandomFCMInit(FCMInit, RandomGenerator):
    def __init__(self, model: FCMAlgorithm, k: int, random_state: int):
        FCMInit.__init__(self, model, k)
        RandomGenerator.__init__(self, random_state)
