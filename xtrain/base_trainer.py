from __future__ import annotations

import dataclasses
import pickle
from collections.abc import Iterator
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence, Union, Any, runtime_checkable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.core.scope import CollectionFilter
from flax.training.train_state import TrainState

from .loss import LossLog, LossFunc
from .strategy import JIT
from .utils import (
    Peekable,
    _get_name,
    unpack_prediction_and_state,
    unpack_x_y_sample_weight,
    wrap_data_stream
)

@runtime_checkable
class Metric(Protocol):
    def update(self, batch: Any, prediction: Any) -> Any:
        ...

    def compute(self, *args, **kwargs) -> dict:
        ...

PathLike = Path | str
MetricLike = Metric | LossFunc
LOSSES = Union[LossFunc, Sequence[LossFunc]]
METRICS = Union[MetricLike, Sequence[MetricLike]]
RNG = jax.Array
Optimizer = Any

# so that multiple calls return the same obj
# this avoids JIT when supplying partial func as args
_cached_partial = lru_cache(partial)

@partial(struct.dataclass, frozen=False)
class TrainIterator(Iterator):
    """The iterator obj returned by Trainer.train(). Iterating this object drives the training. The object supports orbax checkpointing.

    Example:
        ```
        import orbax.checkpoint as ocp

        train_it = trainer.train(dataset)

        # make a checkpoint
        checkpointer = ocp.StandardCheckpointer()
        cp_path = cp_path.absolute() # orbax needs absolute path
        checkpointer.save(cp_path / "test_cp", train_it)

        # restore
        restored = checkpointer.restore(
            cp_path / "test_cp",
            train_it,
        )
        ```

    A caveat is that the checkpoint does not save the current state of the dataset.

    """

    ctx: Trainer = struct.field(pytree_node=False)
    data: Iterator = struct.field(pytree_node=False)
    train_state: TrainState
    rngs: dict[str, RNG]
    loss_logs: tuple[LossLog]
    variables: dict = struct.field(default_factory=dict)
    frozen: dict = struct.field(default_factory=dict, pytree_node=False)

    @property
    def loss_fns(self):
        return (loss_log.loss_fn for loss_log in self.loss_logs)

    @property
    def parameters(self):
        return self.train_state.params

    @property
    def step(self):
        return self.train_state.step

    @property
    def loss(self):
        return self._compute_loss_log()

    @property
    def has_aux(self):
        return self.ctx.mutable or self.ctx.capture_intermediates

    @property
    def vars_and_params(self):
        v = self.variables.copy()
        v["params"] = self.train_state["params"]
        return v

    def __iter__(self):
        self.data = Peekable(iter(self.ctx.dataset))
        return self

    def _compute_loss_log(self) -> dict:
        return {
            _get_name(loss_log.loss_fn): loss_log.compute()
            for loss_log in self.loss_logs
        }

    def reset(self):
        """ Restart the dataset iteration and reset loss metrics
        """
        self.data = Peekable(iter(self.ctx.dataset))
        self.reset_loss_logs()


    def reset_loss_logs(self):
        """Reset internal loss value tracking.

        Args:
            loss_weights: Optional weights of individual loss functions. If not None, the
                total loss is weighted sum.
        """
        for loss in self.loss_logs:
            loss.reset()

    def __next__(self):
        train_fn = self.ctx.strategy.train_step

        batch = next(self.data)

        train_state, loss_logs, preds = train_fn(self, batch)

        preds, variables = unpack_prediction_and_state(preds, self.has_aux)

        self.train_state = train_state
        self.variables = variables
        self.loss_logs = loss_logs

        return preds

    def save_model(self, path: PathLike, sub_module: str | None = None) -> None:
        """Save the model in a pickled file. The pickle is a tuple of
            (module, weights).

        Args:
            path: The file path.
            sub_module: Optionally only save a sub_module of the model
                by specifying the name
        """
        import cloudpickle
        module = self.ctx.model
        params = self.parameters

        if sub_module is not None:
            module = module.bind(dict(params=params))
            module = getattr(module, sub_module)
            module, variables = module.unbind()
            params = variables["params"]

        if isinstance(path, str):
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            cloudpickle.dump((module, params), f)


    def freeze(self, spec:str="", *, unfreeze=False):
        """ Freeze some parameters

        Args:
            spec: a "/" separate string indicating the path of the parameters. The parameters 
              matching the path and all its children parameters will be frozen
        
        KeywordArgs:
            unfreeze: unfreeze instead.
        """
        spec = spec.strip().strip("/").split("/")

        if spec == ['']:
            self.frozen = jax.tree_util.tree_map(lambda _: not unfreeze, self.frozen)
            return

        try:
            sub = self.parameters
            for k in spec:
                sub = sub[k]
        except NameError:
            raise ValueError(f"The key '{k}' in the supplied spec '{spec}' doesn't exist.")

        def _map_fn(path, x):
            if len(path) < len(spec):
                return x
            for k, p in enumerate(path[:len(spec)]):
                if p.key != spec[k]:
                    return x

            return not unfreeze

        self.frozen = jax.tree_util.tree_map_with_path(
            _map_fn, self.frozen
        )
    
    def unfreeze(self, spec:str=""):
        """ Unfreeze some parameters. See [freeze() fucntion](./#lacss.train.base_trainer.TrainIterator.freeze)

        Args:
            spec: a "/" separate string indicating the path of the parameters. The parameters 
              matching the path and all its children parameters will be unfrozen
        """
        return self.freeze(spec, unfreeze=True)
    

@dataclasses.dataclass
class Trainer:
    """A general purpose FLAX model trainer. Help avoiding most of the biolerplate code when trainning with FLAX.

    Attributes:
        model: A Flax module
        losses: A collection of loss function ( loss_fn(batch:Any, prediction:Any)->float ).
        optimizer: An optax optimizer
        seed: RNG seed
        strategy: a training strategy type

    Example:

        ```
        trainer = lacss.train.Trainer(my_module, my_loss_func)

        train_it = trainer.train(my_dataset)

        for k in range(train_steps):
            _ = next(train_it)
            if k % 1000 == 0:
                print(train_it.loss_logs)
                train_it.reset_loss_logs()
        ```

    """

    model: nn.Module
    losses: LOSSES
    optimizer: Optimizer
    mutable: CollectionFilter = False
    capture_intermediates: Union[bool, Callable[["Module", str], bool]] = False
    seed: int | RNG = 42
    strategy: type = JIT

    def _initialize(self, rng: RNG, data: Iterator) -> dict:
        peek = data.peek()
        inputs, _, _ = unpack_x_y_sample_weight(peek)

        return self.strategy.init_fn(rng, self.model, inputs)

    def train(
        self,
        dataset: Iterable,
        *,
        strategy: type | None = None,
        rng_cols: Sequence[str] = ["dropout"],
        init_vars: dict | None = None,
        frozen: dict | None = None,
        method: Union[Callable[..., Any], None] = None,
        **kwargs,
    ) -> TrainIterator:
        """Create the training iterator

        Args:
            dataset: An iterator or iterable to supply the training data.
                The dataset should produce ```(inputs, labels, sample_weight)```, however
                both the labels and the sample_weight are optional. The inputs is either a list 
                (not tuple) or a dict. If latter, the keys are interpreted as the names for
                keyword args of the model's __call__ function. 
            strategy: Optionally override the default strategy.
            rng_cols: Names of any RNG used by the model. Should be a list of strings.
            init_vars: optional variables to initialize model
            frozen: a bool pytree (matching model parameter tree) indicating frozen parameters.
            **kwargs: Additional keyward args passed to the model. E.g. "training=True"

        Returns:
            TrainIterator. Stepping through the iterator will train the model.

        """
        config = dataclasses.replace(self, strategy=strategy or self.strategy)

        config.dataset = wrap_data_stream(dataset)

        assert config.strategy is not None

        dataset_iter = Peekable(iter(config.dataset))

        seed = (
            self.seed
            if isinstance(self.seed, jnp.ndarray)
            else jax.random.PRNGKey(self.seed)
        )

        rng_cols = rng_cols or ["dropout"]
        if isinstance(rng_cols, str):
            rng_cols = [rng_cols]
        seed, key = jax.random.split(seed)
        keys = jax.random.split(key, len(rng_cols))
        rngs = dict(zip(rng_cols, keys))

        if init_vars is None:
            seed, key = jax.random.split(seed)
            keys = jax.random.split(seed, len(rng_cols) + 1)
            init_rngs = dict(zip(rng_cols + ["params"], keys))

            init_vars = config._initialize(init_rngs, dataset_iter)

        if frozen is None:
            frozen = jax.tree_util.tree_map(lambda _: False, init_vars["params"])
        else:
            if jax.tree_util.tree_structure(frozen) != jax.tree_util.tree_structure(init_vars["params"]):
                raise ValueError(f"Invalid frozen dict. Its tree stucture is different from that of the model parameters")

        params = init_vars.pop("params")
        train_state = TrainState.create(
            apply_fn = _cached_partial(
                self.model.apply,
                mutable=self.mutable,
                capture_intermediates=self.capture_intermediates,
                method = method,
                **kwargs,
            ),
            params=params,
            tx=self.optimizer,
        )

        losses = self.losses
        try:
            iter(losses)
        except:
            losses = (losses,)
        loss_logs = tuple(LossLog(loss) for loss in losses)

        return TrainIterator(
            ctx=config,
            data=dataset_iter,
            train_state=train_state,
            rngs=rngs,
            loss_logs=loss_logs,
            variables=init_vars,
            frozen=frozen,
        )

    def test(
        self,
        dataset: Iterable,
        metrics: METRICS,
        variables: dict,
        strategy: type | None = None,
        method: Union[Callable[..., Any], str, None] = None,
        **kwargs,
    ) -> Iterator:
        """Create test/validation iterator.

        Args:
            dataset: An iterator or iterable to supply the testing data.
                The iterator should yield a tupple of (inputs, labels).
            metrics: A list of Metric objects. They should have two functions:
                m.update(preds, **kwargs):
                    preds is the model output. the remaining kwargs are content of
                    labels.
                m.compute():
                    which should return the accumulated metric value.
            variables: Model weights etc. typically get from TrainIterator
            strategy: Optionally override the default strategy.

        Returns:
            An iterator. Stepping through it will drive the updating of each metric
                obj. The iterator itself return the list of metrics.
        """
        if strategy is None:
            strategy = self.strategy

        if isinstance(metrics, str):
            metrics = [metrics]
        try:
            iter(metrics)
        except TypeError:
            metrics = [metrics]
        metrics = [m if isinstance(m, Metric) else LossLog(m) for m in metrics]

        apply_fn=_cached_partial(
            self.model.apply,
            mutable=self.mutable,
            capture_intermediates=self.capture_intermediates,
            method=method,
            **kwargs,
        )

        predict_fn = strategy.predict

        for data in wrap_data_stream(dataset):
            inputs, _, _ = unpack_x_y_sample_weight(data)
            preds = predict_fn(apply_fn, variables, inputs)
            kwargs = dict(
                batch=data,
                prediction=preds,
            )
            for m in metrics:
                m.update(**kwargs)

            yield metrics

    def compute_metrics(self, *args, **kwargs) -> dict:
        """A convient function to compute all metrics. See [test() fucntion](./#lacss.train.base_trainer.Trainer.test)

        Returns:
            A metric dict. Keys are metric names.
        """
        for metrics in self.test(*args, **kwargs):
            pass
        return {_get_name(m): m.compute() for m in metrics}

    def predict(self, dataset: Iterable, variables: dict, strategy: type | None = None, method: Union[Callable[..., Any], str, None] = None, **kwargs):
        """Create predictor iterator.

        Args:
            dataset: An iterator or iterable to supply the input data.
            variables: Model weights etc. typically get from TrainIterator
            strategy: Optionally override the default backend.

        Returns:
            An iterator. Stepping through it will produce model predictions
        """
        if strategy is None:
            strategy = self.strategy

        apply_fn=_cached_partial(
            self.model.apply,
            mutable=self.mutable,
            capture_intermediates=self.capture_intermediates,
            method=method,
            **kwargs,
        )

        predict_fn = strategy.predict

        for inputs in dataset:
            preds = predict_fn(apply_fn, variables, inputs)
            yield preds
