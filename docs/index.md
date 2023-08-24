## Training backends

| class | description |
|---|---|
| xtrain.Core | No model JIT compiling, i.e., for debugging |
| xtrain.JIT | JIT compile the model. Default strategy |
| xtrain.VMapped | Transform the model with vmap. This allows defining a model on unbatched data but train with batched data. |
| xtrain.Distributed | Transform the model with pmap. This allows training the model on multiple devices. |

::: xtrain.Trainer
      options:
        show_root_heading: true

::: xtrain.TFDatasetAdapter
      options:
        show_root_heading: true

::: xtrain.TorchDataLoaderAdapter
      options:
        show_root_heading: true

::: xtrain.loss_func_on
      options:
        show_root_heading: true

::: xtrain.partial_loss_func
      options:
        show_root_heading: true
