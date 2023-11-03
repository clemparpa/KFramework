from kframework.utils.protocols import Counter


class StopRule:
    def __init__(self, model: Counter):
        self.model = model

    def __call__(self) -> bool:
        return self.model.step_counter > 0  # type: ignore
