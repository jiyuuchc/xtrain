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
my_data = itertools.cycle(
    zip(sequence_of_inputs, sequence_of_labels)
)
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
for step in range(train_steps):
    model_out = next(train_iter)
    if step // 1000 == 0:
        print(train_iter.loss_logs)
        train_iter.reset_loss_logs()
```

### Full documentation

https://jiyuuchc.github.io/xtrain/

