from __future__ import annotations

import flax.linen as nn
import jax
import numpy as np
import optax
import pytest

import xtrain

key = jax.random.PRNGKey(0)

def test_metric(train_data, loss_fn):
    trainer = xtrain.Trainer(
        model=nn.Dense(4),
        losses=loss_fn,
        optimizer=optax.adam(0.01),
        seed=key,
        strategy=xtrain.Core,
    )

    train_it = trainer.train(train_data)

    m = trainer.compute_metrics(
        train_data,
        loss_fn,
        dict(params=train_it.parameters),
    )

    assert isinstance(m, dict)
    assert "mse" in m
