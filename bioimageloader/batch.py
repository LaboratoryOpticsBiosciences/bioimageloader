import bisect
import concurrent.futures
import warnings
from math import ceil
from typing import List

import numpy as np

from .base import DatasetInterface


class ConcatDataset:
    """Concatenate Datasets

    Todo
    ----
    - Typing datasets with covariant class
        Lose param hints because of Generic type
    - Intermediate class linking DatasetInterface and [MaskDataset, BBoxDataset,
      BBoxDataset, ...]

    References
    ----------
    .. [1] https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#ConcatDataset
    """
    def __init__(self, datasets: List[DatasetInterface]):
        self.datasets = datasets
        self.acronym = [dset.acronym for dset in self.datasets]
        self.sizes = [len(dset) for dset in self.datasets]
        self.cumulative_sizes = np.cumsum(self.sizes)
        # Check and Warn
        if any([s == 0 for s in self.sizes]):
            i = self.sizes.index(0)
            warnings.warn(f"ind={i} {self.datasets[i].acronym} is empty")
        if len(set(outputs := [dset.output for dset in self.datasets])) != 1:
            warnings.warn(f"output types do not match {outputs}")

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, ind):
        ind_dataset = bisect.bisect_right(self.cumulative_sizes, ind)
        ind_sample = ind if ind_dataset == 0 else ind - self.cumulative_sizes[ind_dataset - 1]
        return self.datasets[ind_dataset][ind_sample]


class BatchDataloader:
    def __init__(
        self,
        dataset: DatasetInterface,
        batch_size: int = 16,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return ceil(len(self.dataset) / self.batch_size)
