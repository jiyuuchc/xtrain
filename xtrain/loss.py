from __future__ import annotations

import functools
from enum import Enum

import jax.numpy as jnp
from flax import struct

from .types import *
from .utils import _get_name

from typing import Union, Sequence, Callable

IndexLike = Union[str, int]
FilterSpec = Union[IndexLike, Sequence[IndexLike]]

class LossLog(struct.PyTreeNode):
    loss_fn: Callable = struct.field(pytree_node=False)
    weight: jnp.ndarray = 1.0
    cnt: jnp.ndarray = 0.0
    sum: jnp.ndarray = 0.0

    def update(self, **kwargs):
        loss = self.loss_fn(**kwargs) * self.weight
        new_log = self.replace(cnt=self.cnt + 1, sum=self.sum + loss)
        return loss, new_log

    def compute(self):
        return self.sum / self.cnt


def loss_func_on(func, filters: FilterSpec):
    @functools.wraps(func)
    def wrapper(**kwargs):
        if "labels" in kwargs and kwargs["labels"] is not None:
            for index in filters:
                kwargs["labels"] = kwargs["labels"][index]

        if "preds" in kwargs and kwargs["preds"] is not None:
            for index in filter:
                kwargs["preds"] = kwargs["preds"][index]

        return func(**kwargs)

    return wrapper


def partial_loss_func(func, *args, **kwargs):
    new_func = functools.partial(func, *args, **kwargs)

    new_func.name = _get_name(func)

    return new_func
