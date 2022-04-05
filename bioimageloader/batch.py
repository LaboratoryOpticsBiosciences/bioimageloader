"""Batch loader module
"""

import bisect
import concurrent.futures
import random
import warnings
from math import ceil
from typing import Dict, Iterator, List, Optional

import numpy as np

from .base import Dataset


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
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.acronym = [dset.acronym for dset in self.datasets]
        self.sizes = [len(dset) for dset in self.datasets]
        self.cumulative_sizes = np.cumsum(self.sizes)
        # Check and Warn
        if any([s == 0 for s in self.sizes]):
            i = self.sizes.index(0)
            warnings.warn(f"ind={i} {self.datasets[i].acronym} is empty",
                          stacklevel=2)
        if len(set(outputs := [dset.output for dset in self.datasets])) != 1:
            warnings.warn(f"output types do not match {outputs}",
                          stacklevel=2)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, ind):
        ind_dataset = bisect.bisect_right(self.cumulative_sizes, ind)
        ind_sample = ind if ind_dataset == 0 else ind - self.cumulative_sizes[ind_dataset - 1]
        return self.datasets[ind_dataset][ind_sample]


class BatchDataloader:
    """Batch loader with multi-processing
    """
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 16,
        shuffle: bool = False,
        drop_last: bool = False,
        num_workers: Optional[int] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers)

        if shuffle:
            idx = list(range(len(self.dataset)))
            random.shuffle(idx)
            dataset.file_list = [dataset.file_list[ind] for ind in idx]
            if hasattr(dataset, 'anno_dict'):
                dataset.anno_dict = dict((i, dataset.anno_dict[ind])
                                         for i, ind in enumerate(idx))

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return ceil(len(self.dataset) / self.batch_size)

    @property
    def _last_size(self):
        if self.drop_last:
            return self.batch_size
        remainder = len(self.dataset) - self.batch_size * (len(self) - 1)
        return remainder

    def __iter__(self):
        return IterBatchDataloader(self)

    def __del__(self):
        # Shutdown mp
        self.executor.shutdown(wait=True)


class IterBatchDataloader(Iterator):
    """Iterate BatchDataloader
    """
    def __init__(self, dataloader: BatchDataloader):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.batch_size = self.dataloader.batch_size
        self.drop_last = self.dataloader.drop_last
        self.num_workers = self.dataloader.num_workers
        self.executor = self.dataloader.executor
        self._last_size = self.dataloader._last_size

        self.batch_ind = 0
        self.batch_end = len(dataloader)

    def __next__(self) -> Dict[str, np.ndarray]:
        if self.batch_ind == self.batch_end:
            raise StopIteration
        if self.batch_ind == self.batch_end - 1 and not self.drop_last:
            batch_size = self._last_size
        else:
            batch_size = self.batch_size
        ind_start = self.batch_size * self.batch_ind
        jobs = [self.executor.submit(_mp_getitem, self.dataset, ind)
                for ind in range(ind_start, ind_start + batch_size)]
        concurrent.futures.wait(jobs)
        self.batch_ind += 1
        # return [j.result() for j in jobs]
        # bundle with the same keys
        d0 = jobs[0].result()
        keys = d0.keys()
        ret = {}
        for k in keys:
            ret[k] = np.stack([d0[k]] + list(map(lambda j: j.result()[k],
                                                 jobs[1:]))
                              )
        return ret


def _mp_getitem(dataset, ind):
    return dataset[ind]
