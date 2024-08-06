from __future__ import annotations

import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from . import base_trainer
from .loss import LossLog
from .utils import Inputs, unpack_prediction_and_state, unpack_x_y_sample_weight

class Eager:
    @classmethod
    def loss_fn(cls, params, train_obj, batch):
        inputs, _, sample_weight = unpack_x_y_sample_weight(batch)

        step = train_obj.train_state.step
        rngs = {
            name: jax.random.fold_in(rng, step) for name, rng in train_obj.rngs.items()
        }

        variables = train_obj.variables
        variables["params"] = params

        model_out = Inputs.apply(
            train_obj.train_state.apply_fn,
            variables, 
            rngs=rngs, 
        )(inputs)

        prediction, new_variables = unpack_prediction_and_state(model_out, train_obj.has_aux)

        losses, loss_logs = [], []
        for loss_fn in train_obj.loss_logs:
            loss = loss_fn.compute_loss(batch, prediction)
            if loss is None:
                loss = 0
            else:
                if sample_weight is None:
                    sample_weight = jnp.ones_like(loss)

                loss = sample_weight * loss
                loss_fn = loss_fn.replace(
                    cnt = loss_fn.cnt + sample_weight.sum(),
                    sum = loss_fn.sum + loss.sum(),
                )

            losses.append(loss.sum())
            loss_logs.append(loss_fn)

        total_loss = sum(losses)

        return total_loss, (model_out, loss_logs)


    @classmethod
    def init_fn(cls, key, model, inputs):
        inputs_obj = Inputs.from_value(inputs)

        state = model.init(key, *inputs_obj.args, **inputs_obj.kwargs)

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
    ) -> tuple[TrainState, tuple[LossLog], tp.Any]:
        try:
            axis_index = jax.lax.axis_index("mapped")
            train_obj = train_obj.replace(
                rngs={
                    name: jax.random.fold_in(key, axis_index)
                    for name, key in train_obj.rngs.items()
                }
            )
        except NameError:
            axis_index = -1

        grads, (preds, losses) = jax.grad(cls.loss_fn, has_aux=True)(
            train_obj.train_state.params,
            train_obj,
            batch,
        )

        try:
            grads = jax.lax.pmean(grads, axis_name="mapped")
            losses = jax.lax.pmean(losses, axis_name="mapped")
        except NameError:
            pass

        grads = jax.tree_util.tree_map(
            lambda x, freeze: jax.numpy.zeros_like(x) if freeze else x,
            grads, train_obj.frozen,
        )

        state = train_obj.train_state.apply_gradients(grads=grads)

        return state, losses, preds


class Core(Eager):
    predict = jax.jit(Eager.predict, static_argnames="apply_fn")

class JIT(Core):
    train_step = jax.jit(Core.train_step)


class _VMapped(Eager):
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

        variables = train_obj.variables
        variables["params"] = params

        model_out = jax.vmap(Inputs.apply(
            train_obj.train_state.apply_fn,
            variables, 
        ))(inputs)

        prediction, new_variables = unpack_prediction_and_state(model_out, train_obj.has_aux)

        losses, loss_logs = [], []
        for loss_fn in train_obj.loss_logs:
            loss = jax.vmap(loss_fn.compute_loss)(batch, prediction)
            if sample_weight is None:
                sample_weight = jnp.ones_like(loss)

            loss = sample_weight * loss
            loss_fn = loss_fn.replace(
                cnt = loss_fn.cnt + sample_weight.sum(),
                sum = loss_fn.sum + loss.sum(),
            )

            losses.append(loss.sum())
            loss_logs.append(loss_fn)

        total_loss = sum(losses)

        return total_loss, (model_out, tuple(loss_logs))


    @classmethod
    def init_fn(cls, key, model, inputs):
        inputs = jax.tree_util.tree_map(lambda v:v[0], inputs)
        return Eager.init_fn(key, model, inputs)


class VMapped(_VMapped):
    train_step = jax.jit(_VMapped.train_step)

    predict = jax.jit(
        jax.vmap(
            _VMapped.predict,
            in_axes=(None, None, 0),
        ),
        static_argnames="apply_fn",
    )


class _Distributed(Eager):
    @classmethod
    def init_fn(cls, key, model, inputs):
        inputs = jax.tree_map(lambda v: v[0], inputs)
        return Eager.init_fn(key, model, inputs)


class Distributed(_Distributed):
    train_step = jax.pmap(
        Eager.train_step,
        axis_name="mapped",
        in_axes=(None, 0),
        out_axes=(None, None, 0),
    )

    predict = jax.pmap(
        Eager.predict,
        axis_name="mapped",
        in_axes=(None, None, 0),
        static_broadcasted_argnums=0,
    )
