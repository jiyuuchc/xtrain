from __future__ import annotations

import functools

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
    """ Create a mean-teacher module.

    Attributes:
        student: The student module
        teacher_fn: By default teacher execute the student's __call__ function. This behavior can be overriden by specifying an apply_fn here.
        momentum: The momentum of the mean teacher.
    """
    student: nn.Module
    teacher_fn: Callable|None = None
    momentum : float = 0.995

    def __post_init__(self):
        super().__post_init__()
        if self.teacher_fn is None:
            self.teacher_fn = lambda mdl, *args, **kwargs: mdl(*args, **kwargs)

    def __call__(self, student_input, teacher_input=None):
        """ Default apply fn of the module. If use this as the apply function, make sure the mutable set include "teacher_variables" key.

        Example:
            ``` 
            mt_module.apply(inputs, mutable="teacher_variables")
            ```

        Args:
            student_input: The input to the student module
            teacher_input: If not None, the teacher takes a different input than the student.

        returns:
            A dict with two keys:
                - student_prediction: the output of the student module
                - teacher_prediction: the output of the teacher module
        """
        student_out = Inputs.apply(self.student)(student_input)

        if not self.has_variable("teacher_variables", "params"):
            self.variables["teacher_variables"] = dict(params=jax.tree_util.tree_map(lambda x:x, self.variables["params"]["student"]))

        if teacher_input is None:
            teacher_input = student_input
        teacher_input = Inputs.from_value(teacher_input)

        rngs = {k:self.scope.make_rng(k) for k in self.scope.rngs.keys()}
        teacher_vars = self.variables["teacher_variables"]
        teacher_fn = nn.apply(self.teacher_fn, self.student, mutable=self.scope.mutable)
        teacher_out = Inputs.apply(teacher_fn, teacher_vars, rngs=rngs)(teacher_input)

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
    """ Create an adversal learning module.

    Attributes:
        main_module: The main (generator) module
        discriminator: The discriminator module
        collection_name: By default, the discriminator input is the output of the main_module. Alternatively, one can 
            store the inputs in a module variable and specify the varibale name here.
        output_key: The dict key of the adversal model outputs that corresponds to the main_module output
        loss_reduction_fn: reduction function of the discriminator loss.
    """

    main_module: nn.Module
    discriminator: nn.Module
    collection_name: str|None = None
    output_key: str = "output"
    loss_reduction_fn: Callable|None = jnp.mean

    def __call__(self, inputs, ref_inputs=None, *args, **kwargs):
        """ Adversal module main function

        Args:
            inputs: The inputs to the generator.
            ref_inputs: A "true" reference.
            args: Additional argumnets to the generator.
            kwargs: Additional keyward arguments to the genreator.  

        Returns:
            A dict with three keys:
                - dsc_loss_main: Discriminator loss regarding generator output
                - dsc_loss_main: Discriminator loss regarding the reference input
                - output: Generator output.
        """
        reduction_fn = self.loss_reduction_fn or (lambda x: x)

        main_out = Inputs.apply(self.main_module, *args, **kwargs)(inputs)

        if self.collection_name is None:
            collections = main_out
        else:  
            collections = self.main_module.variables[self.collection_name]

        collections = gradient_reversal(collections)

        discriminator = Inputs.apply(self.discriminator)
        dsc_loss_main = jax.tree_util.tree_map(
            lambda x: reduction_fn(
                optax.sigmoid_binary_cross_entropy(x, jnp.ones_like(x))
            ),
            discriminator(collections),
        )

        if ref_inputs is not None:
            dsc_loss_ref = jax.tree_util.tree_map(
                lambda x: reduction_fn(
                    optax.sigmoid_binary_cross_entropy(x, jnp.zeros_like(x))
                ),
                discriminator(ref_inputs),
            )
        else:
            dsc_loss_ref = None

        output = dict(
            dsc_loss_main = dsc_loss_main,
            dsc_loss_ref = dsc_loss_ref,
        )
        output[self.output_key] = main_out

        return output
