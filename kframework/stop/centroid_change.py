from kframework.utils.protocols import CentroidAndCounter
from kframework.stop.stop_rule import StopRule
from numpy import array_equal


class CentroidChangeStop(StopRule):
    def __init__(self, model: CentroidAndCounter):
        super().__init__(model)
        self.centroids = model.centroids.copy()

    def __call__(self) -> bool:
        stop = array_equal(self.model.centroids, self.centroids)  # type: ignore
        self.centroids = self.model.centroids.copy()  # type: ignore
        return stop and super().__call__()
