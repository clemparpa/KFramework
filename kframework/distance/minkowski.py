from numpy.typing import NDArray
from numpy import abs as np_abs
from numpy import sum as np_sum
from numpy import number


class MinkowskiDistance:
    def __init__(self, p: int) -> None:
        self.p: int = p

    def __call__(self, x: NDArray, y: NDArray) -> number:
        return (np_sum(np_abs(x - y) ** self.p, axis=-1)) ** (1 / self.p)
