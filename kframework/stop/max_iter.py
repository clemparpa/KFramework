from kframework.model.protocols import T_model
from kframework.stop.stop_rule import StopRule


class MaxIterStop(StopRule[T_model]):
    def __init__(self, model: T_model, max_iter: int):
        super().__init__(model)
        self.max_iter: int = max_iter

    def __call__(self) -> bool:
        return self.model.step_counter >= self.max_iter and super().__call__()  # type: ignore
