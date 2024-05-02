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

def test_trainer():
    trainer = xtrain.Trainer(
        model=nn.Dense(4),
        losses=mse,
        optimizer=optax.adam(0.01),
        seed=key,
        strategy=xtrain.Eager,
    )
    train_it = trainer.train(
        xtrain.GeneratorAdapter(gen)
    )

    last_loss = 1000
    for epoch in range(2):
        train_it.reset()

        for _ in train_it:
            pass

        loss = train_it.loss["mse"]

        assert loss < last_loss

        last_loss = loss

def test_vmap_strategy():
    key = jax.random.PRNGKey(0)
    key, k1, k2 = jax.random.split(key, 3)
    _X = jax.random.uniform(k1, [16, 4, 16])
    _Y = jax.random.uniform(k2, [16, 4, 4])

    def gen(X, Y):
        for x, y in zip(X, Y):
            yield x, y

    def _run():
        g = gen(X=_X, Y=_Y)

        for _ in (it := trainer.train(g)):
            pass
        return it.loss["mse"]

    trainer = xtrain.Trainer(
        model=nn.Dense(4),
        losses=mse,
        optimizer=optax.adam(0.01),
        seed=key,
        strategy=xtrain.Eager,
    )

    eager_loss = _run()

    # assert np.allclose(eager_loss, 0.372847)

    trainer = xtrain.Trainer(
        model=nn.Dense(4),
        losses=mse,
        optimizer=optax.adam(0.01),
        seed=key,
        strategy=xtrain.VMapped,
    )

    vmap_loss = _run()

    assert jax.numpy.allclose(eager_loss, vmap_loss)
