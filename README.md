## XTRAIN: a tiny library for training [Flax](https://github.com/google/flax) models.

Design goals:

  - Help avoiding boiler-plate code
  - Minimal functionality and dependency
  - Agnostic to hardware configuration (e.g. GPU->TPU)

### General workflow

#### Step 1: define your model

```
class MyFlaxModule(nn.Module):
  @nn.compact
  def __call__(self, x):
    ...
```

#### Step 2: define loss function

```
def my_loss_func(batch, prediction):
    x, y_true = batch
    loss = ....
    return loss
```

#### Step 3: create an iterator that supplies training data

```
my_data = zip(sequence_of_inputs, sequence_of_labels)
```

#### Step 4: train

```
# create and initialize a Trainer object
trainer = xtrain.Trainer(
  model = MyFlaxModule(),
  losses = my_loss_func,
  optimizer = optax.adam(1e-4),
)

train_iter = trainer.train(my_data) # returns a iterable object

# iterate the train_iter trains the model
for epoch in range(3):
  for model_out in train_iter:
    pass
  print(train_iter.loss_logs)
  train_iter.reset_loss_logs()
```

### Training data format

- tensowflow Dataset
- torch dataloader
- generator function
- other python iterable that produce numpy data

### Checkpointing

train_iter is orbax compatible.

```
import orbax.checkpoint as ocp
ocp.StandardCheckpointer().save(cp_path, args=ocp.args.StandardSave(train_iter))
```

### Freeze submodule
```
train_iter.freeze("submodule/Dense_0/kernel")
```

### Simple batch parallelism on multiple device
```
# Add a new batch dim to you dataset
ds = ds.batch(8)
# create trainer with the Distributed strategy
trainer_iter = xtrain.Trainer(model, losses, optimizer, strategy=xtrain.Distributed).train(ds)
```

### API documentation

https://jiyuuchc.github.io/xtrain/

