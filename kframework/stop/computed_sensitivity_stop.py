from kframework.stop.stop_rule import StopRule
from typing import Callable
from numpy.typing import NDArray
from numpy import array, floating, inf, ones_like, sum as np_sum


class ComputedSensitivityStop(StopRule):
    def __init__(
        self, model, sensitivity: float, norm: Callable[[NDArray], NDArray | floating]
    ):
        super().__init__(model)
        self.norm = norm
        self.sensitivity: float = sensitivity
        self.value: NDArray | float = self._init_value()

    def _init_value(self) -> NDArray | float:
        raise NotImplementedError("Must be implemented by subclasses")

    def _compute_value(self, X: NDArray):
        raise NotImplementedError("Must be implemented by subclasses")

    def __call__(self, X: NDArray) -> bool:
        value = self._compute_value(X)
        stop = self.norm(self.value - value) < self.sensitivity
        self.value = value
        return stop and super().__call__()  # type: ignore


class VarianceSensitivityStop(ComputedSensitivityStop):
    def _init_value(self):
        return inf

    def _compute_value(self, X: NDArray):
        return array(
            [
                np_sum(self.model.distance(cluster_data, centroid)) / cluster_count
                if (
                    cluster_count := len(
                        cluster_data := X[self.model.clusters == cluster]
                    )
                )
                > 0
                else 0
                for cluster, centroid in enumerate(self.model.centroids)
            ]
        )
