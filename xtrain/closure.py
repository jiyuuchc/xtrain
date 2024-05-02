from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from xtrain.utils import Inputs

def _copy_tree(tree):
    return {
        k: jax.tree_util.tree_map(lambda x:x, v["student"]) for k, v in tree.items()
    }

class MeanTeacher(nn.Module):
    student: nn.Module
    teacher_fn: Callable|None = None
    momentum : float = 0.995

    def __post_init__(self):
        super().__post_init__()
        if self.teacher_fn is None:
            self.teacher_fn = lambda mdl, *args, **kwargs: mdl(*args, **kwargs)

    def __call__(self, student_input, teacher_input=None):
        student_input = Inputs.from_value(student_input)
        student_out = self.student(*student_input.args, **student_input.kwargs)

        if not self.has_variable("teacher_variables", "teacher_params"):
            self.variables["teacher_variables"] = dict(params=jax.tree_util.tree_map(lambda x:x, self.variables["params"]["student"]))

        if teacher_input is None:
            teacher_input = student_input
        teacher_input = Inputs.from_value(teacher_input)

        rngs = {k:self.scope.make_rng(k) for k in self.scope.rngs.keys()}
        teacher_vars = self.variables["teacher_variables"]
        teacher_out = nn.apply(self.teacher_fn, self.student, mutable=self.scope.mutable)(
                teacher_vars, *teacher_input.args, **teacher_input.kwargs, rngs=rngs)

        if self.scope.mutable is not False:
            teacher_out, teacher_vars = teacher_out
            # if "params" in teacher_vars:
            #     teacher_vars.pop("params")
            self.variables["teacher_variables"].update(teacher_vars)

        # update teacher parameters
        self.variables["teacher_variables"]["params"] = jax.tree_util.tree_map(
            lambda x, y: x * self.momentum + y * (1-self.momentum),
            self.variables["teacher_variables"]["params"], self.variables["params"]["student"],
        )

        return dict(
            student_prediction = student_out, 
            teacher_prediction = jax.lax.stop_gradient(teacher_out),
        )


@jax.custom_vjp
def gradient_reversal(x):
    return x

def _gr_fwd(x):
    return x, None

def _gr_bwd(_, g):
    return (jax.tree_util.tree_map(lambda v: -v, g),)

gradient_reversal.defvjp(_gr_fwd, _gr_bwd)
""" A gradient reveral op. 

    This is a no-op during inference. During training, it does nothing
    in the forward pass, but reverse the sign of gradient in backward
    pass. Typically placed right before a discriminator in adversal 
    training.
"""

class Adversal(nn.Module):
    main_module: nn.Module
    discriminator: nn.Module
    collection_name: str|None = None
    loss_reduction_fn: Callable|None = jnp.mean
    output_key: str = "output"

    def __call__(self, inputs, ref_inputs, *args, **kwargs):

        inputs_obj = Inputs.from_value(inputs)

        main_out = self.main_module(*inputs_obj.args, *args, **inputs_obj.kwargs, **kwargs)

        if self.collection_name is None:
            collections = main_out
        else:  
            collections = self.main_module.variables[self.collection_name]

        collections = gradient_reversal(collections)

        dsc_main = self.discriminator(collections)
        dsc_ref = self.discriminator(ref_inputs)

        reduction_fn = self.loss_reduction_fn or (lambda x: x)
        dsc_loss_main = jax.tree_util.tree_map(
            lambda x: reduction_fn(
                optax.sigmoid_binary_cross_entropy(x, jnp.ones_like(x))
            ),
            dsc_main,
        )
        dsc_loss_ref = jax.tree_util.tree_map(
            lambda x: reduction_fn(
                optax.sigmoid_binary_cross_entropy(x, jnp.zeros_like(x))
            ),
            dsc_ref,
        )

        output = dict(
            dsc_loss_main = dsc_loss_main,
            dsc_loss_ref = dsc_loss_ref,
        )
        output[self.output_key] = main_out

        return output
