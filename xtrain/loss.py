from __future__ import annotations

import functools
from enum import Enum

import jax.numpy as jnp
from flax import struct

from .types import *
from .utils import _get_name

from typing import Union, Sequence

IndexLike = Union[str, int]
FilterSpec = Union[IndexLike, Sequence[IndexLike]]

class Reduction(Enum):
    """
    Types of loss reduction.

    Contains the following values:
    * `NONE`: Weighted losses with one dimension reduced (axis=-1, or axis
        specified by loss function). When this reduction type used with built-in
        Keras training loops like `fit`/`evaluate`, the unreduced vector loss is
        passed to the optimizer but the reported loss will be a scalar value.
    * `SUM`: Scalar sum of weighted losses.
    * `SUM_OVER_BATCH_SIZE`: Scalar `SUM` divided by number of elements in losses.
    """

    # AUTO = "auto"
    NONE = "none"
    SUM = "sum"
    SUM_OVER_BATCH_SIZE = "sum_over_batch_size"

    @classmethod
    def all(cls):
        return (
            # cls.AUTO,
            cls.NONE,
            cls.SUM,
            cls.SUM_OVER_BATCH_SIZE,
        )

    @classmethod
    def validate(cls, key):
        if key not in cls.all():
            raise ValueError("Invalid Reduction Key %s." % key)


def reduce_loss(
    values: jnp.ndarray, sample_weight: tp.Optional[jnp.ndarray], weight, reduction
) -> jnp.ndarray:

    values = jnp.asarray(values)

    if sample_weight is not None:
        # expand `sample_weight` dimensions until it has the same rank as `values`
        while sample_weight.ndim < values.ndim:
            sample_weight = sample_weight[..., jnp.newaxis]

        values *= sample_weight

    if reduction == Reduction.NONE:
        loss = values
    elif reduction == Reduction.SUM:
        loss = jnp.sum(values)
    elif reduction == Reduction.SUM_OVER_BATCH_SIZE:
        loss = jnp.sum(values) / jnp.prod(jnp.array(values.shape))
    else:
        raise ValueError(f"Invalid reduction '{reduction}'")

    return loss * weight


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


def reduce_loss_func(func, reduction=Reduction.SUM_OVER_BATCH_SIZE):
    @functools.wraps(func)
    def wrapper(**kwargs):
        values = jnp.asarray(func(**kwargs))

        if reduction == Reduction.NONE:
            loss = values
        elif reduction == Reduction.SUM:
            loss = jnp.sum(values)
        elif reduction == Reduction.SUM_OVER_BATCH_SIZE:
            loss = jnp.sum(values) / jnp.prod(jnp.array(values.shape[1:]))
        else:
            raise ValueError(f"Invalid reduction '{reduction}'")

        return loss

    return wrapper


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
