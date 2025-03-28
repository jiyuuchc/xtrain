from __future__ import annotations

import flax.linen as nn
import jax
import numpy as np
import optax

import pytest

import xtrain
from xtrain.utils import gf_batch

def test_pseudo_distributed(train_data, loss_fn):
    trainer = xtrain.Trainer(
        model=nn.Dense(4),
        losses=loss_fn,
        optimizer=optax.adam(0.01),
        seed=123,
    )

    for _ in (it := trainer.train(train_data, strategy=xtrain.Core)):
        pass
    core_loss = it.loss["mse"]

    it = trainer.train(
        gf_batch(train_data, batch_size=1),
        strategy=xtrain.Distributed,
    )
    for _ in it:
        pass

    distributed_loss = it.loss["mse"]

    assert jax.numpy.allclose(core_loss, distributed_loss)
