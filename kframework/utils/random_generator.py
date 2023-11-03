from numpy.random import default_rng


class RandomGenerator:
    def __init__(self, seed: int = 42):
        self._random_generator = default_rng(seed)
