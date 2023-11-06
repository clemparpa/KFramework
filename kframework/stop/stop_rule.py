from typing import Generic
from kframework.model.protocols import T_model


class StopRule(Generic[T_model]):
    def __init__(self, model: T_model):
        self.model = model

    def __call__(self) -> bool:
        return self.model.step_counter > 0  # type: ignore
