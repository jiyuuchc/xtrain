from __future__ import annotations

import flax.linen as nn
import jax
import numpy as np
import optax
import pytest

import xtrain

def mse(batch, prediction):
    labels = batch[1]
    return ((prediction - labels) ** 2).mean()


def test_vmap_strategy():
    key = jax.random.PRNGKey(0)
    key, k1, k2 = jax.random.split(key, 3)
    _X = jax.random.uniform(k1, [16, 4, 16])
    _Y = jax.random.uniform(k2, [16, 4, 4])

    def gen(X, Y):
        for x, y in zip(X, Y):
            yield x, y

    trainer = xtrain.Trainer(
        model=nn.Dense(4),
        losses=mse,
        optimizer=optax.adam(0.01),
        seed=key,
        strategy=xtrain.Core,
    )

    train_it = trainer.train(gen(X=_X, Y=_Y))

    m = trainer.compute_metrics(
        gen(X=_X, Y=_Y),
        mse,
        dict(params=train_it.parameters),
    )

    assert isinstance(m, dict)
    assert "mse" in m
    assert np.allclose(m["mse"], 0.83978, atol=1e-4, rtol=1e-4)
