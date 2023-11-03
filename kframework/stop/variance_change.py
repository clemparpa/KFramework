from kframework.utils.protocols import VarianceAndCounter
from kframework.stop.stop_rule import StopRule
from numpy import sum as np_sum


class VarianceChangeStop(StopRule):
    def __init__(self, model: VarianceAndCounter, tol: float):
        super().__init__(model)
        self.tol: float = tol
        self.sum_variance: float = np_sum(model.variance)

    def __call__(self) -> bool:
        new_sum_variance = np_sum(self.model.variance)  # type: ignore
        stop = abs(self.sum_variance - new_sum_variance) < self.tol
        self.sum_variance = new_sum_variance
        return stop and super().__call__()
