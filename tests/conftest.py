from pathlib import Path

import pytest
import jax

MODULE_DIR = Path(__file__).parent

@pytest.fixture(scope="package")
def train_data():
    key = jax.random.PRNGKey(0)
    key, k1, k2 = jax.random.split(key, 3)
    _X = jax.random.uniform(k1, [16, 4, 16])
    _Y = jax.random.uniform(k2, [16, 4, 4])

    def gen(X=_X, Y=_Y):
        for x, y in zip(X, Y):
            yield x, y

    return gen

@pytest.fixture(scope="package")
def loss_fn():
    def mse(batch, prediction):
        labels = batch[1]
        return ((prediction - labels) ** 2).mean()

    return mse
