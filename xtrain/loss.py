from __future__ import annotations

from typing import Protocol, Sequence, Any

from functools import partial

import jax.numpy as jnp
from flax import struct

from .utils import _get_name, unpack_x_y_sample_weight
from jax.typing import ArrayLike

class LossFunc_(Protocol):
    def __call__(self, batch: Any, prediction: Any) -> ArrayLike | tuple[ArrayLike, ArrayLike]:
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
    
    def compute_loss(self, batch, prediction):
        if isinstance(self.loss_fn, str):
            loss = prediction
            for k in self.loss_fn.split("/"):
                loss = loss[k]
        else:
            loss = self.loss_fn(batch, prediction)

        return loss

    def update(self, batch, prediction):
        _, _, sample_weight = unpack_x_y_sample_weight(batch)

        loss = self.compute_loss(batch, prediction)

        if loss is not None:
            loss = jnp.asarray(loss)

            if sample_weight is None:
                sample_weight = jnp.ones_like(loss)
            loss = sample_weight * loss

            self.cnt += sample_weight.sum()
            self.sum += loss.sum()

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
