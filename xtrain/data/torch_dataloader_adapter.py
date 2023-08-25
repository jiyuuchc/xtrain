import jax
from torch.utils.data import DataLoader

from .utils import list_to_tuple


class TorchDataLoaderAdapter:
    """Convert torch dataloader into a python iterable suitable for [xtrain.Trainer](./#xtrain.Trainer)

    ```
    my_dataset = TorchDataLoaderAdapter(my_torch_dataloader)
    ```

    """

    def __init__(
        self,
        x: DataLoader,
        steps: int = -1,
    ):
        self.steps = steps
        self._dataset = x
        self.current_step = 0

    def get_dataset(self):
        def parse_dataloader_gen():
            self.current_step = 0
            for batch in iter(self._dataset):
                self.current_step += 1
                batch = jax.tree_util.tree_map(
                    lambda x: x.cpu().numpy(), list_to_tuple(batch)
                )
                yield batch

        return parse_dataloader_gen

    def __iter__(self):
        g = self.get_dataset()
        return g()
