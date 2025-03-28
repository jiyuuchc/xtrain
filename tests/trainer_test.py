from __future__ import annotations

import flax.linen as nn
import jax
import numpy as np
import optax

import pytest

import xtrain

key = jax.random.PRNGKey(0)

def test_variable_structure(train_data, loss_fn):
    trainer = xtrain.Trainer(
        model=nn.Dense(4),
        losses=loss_fn,
        optimizer=optax.adam(0.01),
        seed=key,
        strategy=xtrain.Core,
    )
    train_it = trainer.train(train_data)

    var_struct = jax.tree_util.tree_structure(train_it.variables)

    last_loss = 1000

    for epoch in range(2):
        train_it.reset()

        for _ in train_it:
            pass

        loss = train_it.loss["mse"]

        assert loss < last_loss

        last_loss = loss

    assert var_struct == jax.tree_util.tree_structure(train_it.variables)


def test_vmap_strategy(train_data, loss_fn):
    def _run(**kwargs):
        for _ in (it := trainer.train(train_data, **kwargs)):
            pass
        return it.loss["mse"]

    trainer = xtrain.Trainer(
        model=nn.Dense(4),
        losses=loss_fn,
        optimizer=optax.adam(0.01),
        seed=key,
    )

    core_loss = _run(strategy=xtrain.Core)

    vmap_loss = _run(strategy=xtrain.VMapped)

    assert jax.numpy.allclose(core_loss, vmap_loss)
