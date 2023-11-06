from typing import Protocol, TypeVar
from numpy.typing import NDArray
from kframework.distance.protocols import WithDistance


class Model(Protocol):
    step_counter: NDArray


T_model = TypeVar("T_model", bound=Model)


class CentroidsModel(Model, Protocol):
    centroids: NDArray


T_centroids_model = TypeVar("T_centroids_model", bound=CentroidsModel)


class ClustersModel(Model, Protocol):
    clusters: NDArray


class TagsModel(Model, Protocol):
    Tags: NDArray


T_tags_model = TypeVar("T_tags_model", bound=TagsModel)


class VarianceModel(CentroidsModel, ClustersModel, WithDistance, Protocol):
    ...


T_variance_model = TypeVar("T_variance_model", bound=VarianceModel)
