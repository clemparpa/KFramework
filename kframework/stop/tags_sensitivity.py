from typing import Callable
from kframework.utils.protocols import TagsAndCounter
from kframework.stop.stop_rule import StopRule
from numpy.typing import NDArray
from numpy import floating


class NormSensitivityStop(StopRule):
    def __init__(
        self,
        model: TagsAndCounter,
        sensitivity: float,
        norm: Callable[[NDArray], floating],
    ):
        super().__init__(model)
        self.sensitivity: float = sensitivity
        self.norm = norm
        self.tags = model.tags.copy()

    def __call__(self) -> bool:
        stop = self.norm(self.model.tags - self.tags) < self.sensitivity  # type: ignore
        self.tags = self.model.tags.copy()  # type: ignore
        return stop and super().__call__()  # type: ignore
