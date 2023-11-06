from typing import Protocol
from numpy.typing import NDArray
from numpy import number


class Distance(Protocol):
    def __call__(self, x: NDArray | number, y: NDArray | number) -> NDArray | number:
        ...


class WithDistance(Protocol):
    distance: Distance
