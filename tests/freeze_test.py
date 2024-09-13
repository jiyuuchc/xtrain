from __future__ import annotations

import flax.linen as nn
import jax
import numpy as np
import optax

import pytest

import xtrain

key = jax.random.PRNGKey(0)
key, k1, k2 = jax.random.split(key, 3)
_X = jax.random.uniform(k1, [16, 4, 16])
_Y = jax.random.uniform(k2, [16, 4, 4])

def gen(X=_X, Y=_Y):
    for x, y in zip(X, Y):
        yield x, y

def mse(batch, prediction):
    labels = batch[1]
    return ((prediction - labels) ** 2).mean()

class M(nn.Module):
    def setup(self):
        self.sub1 = nn.Dense(4)

    def __call__(self, x):
        return self.sub1(x)

def test_freeze_fn():
    trainer = xtrain.Trainer(
        model=M(),
        losses=mse,
        optimizer=optax.adamw(0.01),
        seed=key,
        strategy=xtrain.Eager,
    )
    train_it = trainer.train(gen)

    # freeze partial params
    params_copy = jax.tree_util.tree_map(lambda x: x.copy(), train_it.parameters["sub1"]['kernel'])
    train_it.freeze("sub1/bias")
    assert train_it.frozen["sub1"]["bias"] is True
    next(train_it)
    next(train_it) # step twice to force param update
    assert jax.numpy.all(train_it.parameters["sub1"]["bias"] == 0)
    for x, y in zip(
        jax.tree_util.tree_leaves(params_copy), 
        jax.tree_util.tree_leaves(train_it.parameters["sub1"]['kernel'])
    ):
        assert not jax.numpy.allclose(x, y)

    # freeze all params
    params_copy = jax.tree_util.tree_map(lambda x: x.copy(), train_it.parameters["sub1"])
    train_it.freeze("sub1")
    next(train_it)
    next(train_it)
    for x, y in zip(
        jax.tree_util.tree_leaves(params_copy), 
        jax.tree_util.tree_leaves(train_it.parameters["sub1"])
    ):
        assert jax.numpy.allclose(x, y)
