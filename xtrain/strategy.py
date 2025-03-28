from __future__ import annotations

import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState

from . import base_trainer
from .loss import LossLog
from .utils import Inputs, unpack_prediction_and_state, unpack_x_y_sample_weight

def _fold_rng(rngs, n):
    return {
        name: jax.random.fold_in(rng, n) for name, rng in rngs.items()
    }


class Eager:
    @classmethod
    def loss_fn(cls, params, train_obj, batch):
        inputs, _, sample_weight = unpack_x_y_sample_weight(batch)

        step = train_obj.train_state.step
        rngs = _fold_rng(train_obj.rngs, step)

        variables = train_obj.variables.copy()
        variables["params"] = params

        prediction = Inputs.apply(
            train_obj.train_state.apply_fn,
            variables, 
            rngs=rngs, 
        )(inputs)

        prediction, new_variables = unpack_prediction_and_state(prediction, train_obj.has_aux)
        if "params" in new_variables:
            del new_variables["params"]
        train_obj.variables.update(new_variables)

        losses = []
        for loss_fn in train_obj.loss_logs:
            losses.append(loss_fn.update(batch, prediction))

        total_loss = sum(losses)

        return total_loss, (prediction, train_obj)


    @classmethod
    def init_fn(cls, key, model, inputs, method=None):
        inputs_obj = Inputs.from_value(inputs)

        if method is None:
            fn = model.init
        else:
            fn = nn.init(method, model)

        state = fn(key, *inputs_obj.args, **inputs_obj.kwargs)

        return state


    @classmethod
    def predict(cls, apply_fn, variables, inputs):
        # print("JIT Predict")
        inputs_obj = Inputs.from_value(inputs)
        preds = apply_fn(variables, *inputs_obj.args, **inputs_obj.kwargs)
        return preds


    @classmethod
    def train_step(
        cls,
        train_obj: base_trainer.TrainIterator,
        batch: tp.Any,
    ) -> tuple[tp.Any, base_trainer.TrainIterator]:
        grads, (preds, train_obj) = jax.grad(cls.loss_fn, has_aux=True)(
            train_obj.train_state.params,
            train_obj,
            batch,
        )

        train_obj.train_state = train_obj.train_state.apply_gradients(grads=grads)

        return preds, train_obj


class Core(Eager):
    train_step = jax.jit(Eager.train_step)

JIT = Core

class VMapped(Eager):
    _transform_fn = jax.vmap

    @classmethod
    def loss_fn(cls, params, train_obj, batch):
        inputs, _, sample_weight = unpack_x_y_sample_weight(batch)
        inputs = Inputs.from_value(inputs)

        # inputs = Inputs.from_value(inputs)
        batch_size = jax.tree_util.tree_leaves(batch)[0].shape[0]
        step = train_obj.train_state.step
        rngs = {
            name: jax.random.split(jax.random.fold_in(rng, step), batch_size) 
            for name, rng in train_obj.rngs.items()
        }
        inputs = inputs.update(rngs=rngs)

        variables = train_obj.variables.copy()
        variables["params"] = params

        prediction = cls._transform_fn(Inputs.apply(
            train_obj.train_state.apply_fn,
            variables, 
        ))(inputs)

        prediction, new_variables = unpack_prediction_and_state(prediction, train_obj.has_aux)
        if "params" in new_variables:
            del new_variables["params"]

        new_variables = jax.tree_util.tree_map(lambda x: x.mean(axis=0), new_variables)
        train_obj.variables.update(new_variables)

        losses, loss_logs = [], []
        for loss_fn in train_obj.loss_logs:
            loss = cls._transform_fn(loss_fn.compute_loss)(batch, prediction)
            if sample_weight is None:
                sample_weight = jnp.ones_like(loss)

            loss = sample_weight * loss
            loss_fn = loss_fn.replace(
                cnt = loss_fn.cnt + sample_weight.sum(),
                total = loss_fn.total + loss.sum(),
            )

            losses.append(loss.sum())
            loss_logs.append(loss_fn)

        total_loss = sum(losses)
        train_obj.loss_logs = loss_logs

        return total_loss, (prediction, train_obj)

    @classmethod
    def init_fn(cls, key, model, inputs, method=None):
        inputs = jax.tree_util.tree_map(lambda v:v[0], inputs)
        return Core.init_fn(key, model, inputs, method)


    predict = jax.vmap(
        Core.predict,
        in_axes=(None, None, 0),
    )


class Distributed(VMapped):
    _transform_fn = jax.pmap

    predict = jax.pmap(
        Core.predict,
        axis_name="mapped",
        in_axes=(None, None, 0),
        static_broadcasted_argnums=0,
    )
