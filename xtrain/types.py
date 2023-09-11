from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol, Union

import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from jax import Array

try:
    from jax.typing import ArrayLike
except:
    ArrayLike = Union[
        Array,  # JAX array type
        np.ndarray,  # NumPy array type
        np.bool_,
        np.number,  # NumPy scalar types
        bool,
        int,
        float,
        complex,  # Python scalar types
    ]

Params = FrozenDict

Optimizer = optax.GradientTransformation

PathLike = Union[str, Path]


class LossFunc(Protocol):
    def __call__(self, preds: Any, labels: Any, inputs: Any) -> ArrayLike:
        ...


class Metric(Protocol):
    def update(self, *args, **kwargs):
        ...

    def compute(self, *args, **kwargs) -> dict:
        ...


__all__ = [
    "Array",
    "ArrayLike",
    "Params",
    "Optimizer",
    "LossFunc",
    "Metric",
    "PathLike",
]
