from typing import Callable
from kframework.model.protocols import T_centroids_model, T_model, T_tags_model
from kframework.stop.stop_rule import StopRule
from numpy.typing import NDArray
from numpy import floating


class AttributeSensitivityStop(StopRule[T_model]):
    _loss_obj_name: str
    _make_loss_obj_copy: bool

    def __init__(self, model, sensitivity: float, norm: Callable[[NDArray], floating]):
        super().__init__(model)
        self.norm = norm
        self.sensitivity: float = sensitivity
        self._last_obj = getattr(self.model, self._loss_obj_name, None)
        if self._make_loss_obj_copy and self._last_obj is not None:
            self._last_obj = self._last_obj.copy()

    @property
    def last_obj(self):
        return self._last_obj

    @last_obj.setter
    def last_obj(self, value):
        self._last_obj = value.copy() if self._make_loss_obj_copy else value

    def __call__(self) -> bool:
        stop = False
        new_obj = getattr(self.model, self._loss_obj_name)
        if self.last_obj is not None:
            stop = self.norm(new_obj - self._last_obj) < self.sensitivity
        self.last_obj = new_obj
        return stop and super().__call__()  # type: ignore


class CentroidSensitivityStop(AttributeSensitivityStop[T_centroids_model]):
    _loss_obj_name: str = "centroids"
    _make_loss_obj_copy: bool = True


class TagsSensitivityStop(AttributeSensitivityStop[T_tags_model]):
    _loss_obj_name: str = "tags"
    _make_loss_obj_copy: bool = True
