from __future__ import annotations

from typing import Protocol, Sequence, Union, Any

from functools import partial

import jax.numpy as jnp
from flax import struct

from .utils import _get_name

class LossFunc_(Protocol):
    def __call__(self, batch: Any, prediction: Any) -> float:
        ...

LossFunc = LossFunc_ | str

@partial(struct.dataclass, frozen=False)
class LossLog:
    loss_fn: LossFunc = struct.field(pytree_node=False)
    weight: float = 1.0
    cnt: float = 0.0
    sum: float = 0.0

    def __post_init__(self):
        self.__name__ = _get_name(self.loss_fn)

    # update() is meant to be called in JAX transformation, where objects are immutable
    # so it returns a copy of self in order for the caller to track changes to the object.
    def update(self, batch, prediction):
        if isinstance(self.loss_fn, str):
            loss = prediction
            for k in self.loss_fn.split("/"):
                loss = loss[k]
        else:
            loss = self.loss_fn(batch, prediction)

        if loss is None:
            return 0.0, self

        loss *= self.weight
        self.cnt += 1
        self.sum += loss

        return loss, self

    def compute(self):
        if self.cnt == 0:
            return 0
        else:
            return self.sum / self.cnt

    def reset(self):
        self.cnt = 0.0
        self.sum = 0.0

    def __repr__(self) -> str:
        return self.__name__ + f": {float(self.compute()):.4f}"
