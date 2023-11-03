from typing import Protocol
from numpy.typing import NDArray


class Counter(Protocol):
    step_counter: NDArray


class Variance(Protocol):
    variance: NDArray


class CentroidAndCounter(Counter, Protocol):
    centroids: NDArray


class VarianceAndCounter(Counter, Variance, Protocol):
    variance: NDArray


class Tags(Protocol):
    tags: NDArray


class TagsAndCounter(Tags, Counter, Protocol):
    ...
