import pytest
import tensorflow as tf
import numpy as np

import xtrain

def test_tf_adaptor():
    ds = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    ds = xtrain.TFDatasetAdapter(ds)
    ds_it = iter(ds)

    assert next(ds_it) == 1
    assert next(ds_it) == 2

    ds_it = iter(ds)

    assert next(ds_it) == 1
    assert next(ds_it) == 2

def test_gen_adaptor():
    def gen():
        for x in range(10):
            yield x + 1
    ds = xtrain.GeneratorAdapter(gen)
    ds_it = iter(ds)
    assert next(ds_it) == 1
    assert next(ds_it) == 2

    ds = xtrain.GeneratorAdapter(gen, prefetch=1)
    ds_it = iter(ds)
    assert next(ds_it) == 1
    assert next(ds_it) == 2
    for x in ds_it:
        pass
    assert x == 10

def test_gen_adaptor_exception():
    def gen():
        for x in range(3):
            if x == 2:
                raise ValueError("Test!")
            else:
                yield x + 1

    ds = xtrain.GeneratorAdapter(gen)
    with pytest.raises(ValueError, match="Test!"):
        for _ in ds:
            pass

    ds = xtrain.GeneratorAdapter(gen, prefetch=1)
    with pytest.raises(ValueError, match="Test!"):
        for _ in ds:
            pass
