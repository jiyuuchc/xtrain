from __future__ import annotations

import typing as tp
from functools import partial

import jax
from flax.training.train_state import TrainState

from . import base_trainer
from .loss import LossLog
from .utils import Inputs, unpack_prediction_and_state, unpack_x_y_sample_weight

class Eager:
    @classmethod
    def loss_fn(cls, params, train_obj, batch):
        inputs, _, _ = unpack_x_y_sample_weight(batch)

        step = train_obj.train_state.step
        rngs = {
            name: jax.random.fold_in(rng, step) for name, rng in train_obj.rngs.items()
        }
        variables = train_obj.variables
        variables["params"] = params

        model_out = Inputs.apply(train_obj.train_state.apply_fn, variables, rngs=rngs)(inputs)

        prediction, _ = unpack_prediction_and_state(model_out, train_obj.has_aux)

        args = dict(
            batch=batch,
            prediction=prediction,
        )

        losses, loss_logs = zip(
            *[loss_log.update(**args) for loss_log in train_obj.loss_logs]
        )
        total_loss = sum(losses)

        return total_loss, (loss_logs, model_out)

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

        grads, (loss_logs, preds) = jax.grad(cls.loss_fn, has_aux=True)(
            train_obj.train_state.params,
            train_obj,
            batch,
        )

        if axis_index >= 0:
            grads = jax.lax.pmean(grads, axis_name="mapped")
            # aggregate logs
            loss_logs = jax.tree_map(partial(jax.lax.pmean, axis_name="mapped"), loss_logs)
        
        grads = jax.tree_util.tree_map(
            lambda x, freeze: jax.numpy.zeros_like(x) if freeze else x,
            grads, train_obj.frozen,
        )
        
        state = train_obj.train_state.apply_gradients(grads=grads)

        #         # sync batch statistics
        #         model.map(partial(jax.lax.pmean, axis_name="mapped"), tx.BatchStat, inplace=True)

        return state, loss_logs, preds


class Core(Eager):
    predict = jax.jit(Eager.predict, static_argnames="apply_fn")

    @classmethod
    def init_fn(cls, key, model, inputs):
        inputs_obj = Inputs.from_value(inputs)

        state = jax.jit(model.init)(key, *inputs_obj.args, **inputs_obj.kwargs)

        return state


class JIT(Core):
    train_step = jax.jit(Core.train_step)


class _Distributed(Eager):
    @classmethod
    def init_fn(cls, key, model, inputs):
        inputs = jax.tree_map(lambda v: v[0], inputs)
        return Eager.init_fn(key, model, inputs)


class Distributed(_Distributed):
    train_step = jax.pmap(
        _Distributed._train_step,
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


class VMapped(_Distributed):
    train_step = jax.jit(
        jax.vmap(
            _Distributed._train_step,
            axis_name="mapped",
            in_axes=(None, 0),
            out_axes=(None, None, 0),
        ),
    )

    predict = jax.jit(
        jax.vmap(
            Eager.predict,
            axis_name="mapped",
            in_axes=(None, None, 0),
        ),
        static_argnames="apply_fn",
    )
