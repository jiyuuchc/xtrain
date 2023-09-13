import pytest
import tensorflow as tf

import xtrain


def test_tf_adaptor():
    ds = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    ds_it = xtrain.TFDatasetAdapter(ds)

    assert next(ds_it) == 1
    assert next(ds_it) == 2
