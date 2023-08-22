import jax
from tensorflow.data import Dataset


class TFDatasetAdapter:
    """Adapter that handles `tf.data.Dataset`."""

    def __init__(self, ds: Dataset, steps=-1):
        self._dataset = ds

    def get_dataset(self):
        def parse_tf_data_gen():
            for batch in iter(self._dataset):
                batch = jax.tree_util.tree_map(lambda x: x.numpy(), batch)
                yield batch

        return parse_tf_data_gen

    def __iter__(self):
        g = self.get_dataset()
        return g()
