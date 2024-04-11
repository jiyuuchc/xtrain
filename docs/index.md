## Training backends

| class | description |
|---|---|
| xtrain.Core | No model JIT compiling, i.e., for debugging |
| xtrain.JIT | JIT compile the model. Default strategy |
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

::: xtrain.GeneratorAdapter
      options:
        show_root_heading: true
