import random
from functools import cached_property
from itertools import accumulate
from pathlib import Path
from typing import List, Sequence, Union, Protocol
from copy import deepcopy

import numpy as np
import pandas as pd
from PIL import Image

from bioimageloader.base import NucleiDataset


def random_label_cmap(n=2**16, h=(0, 1), l=(.4, 1), s=(.2, .8)):
    # https://github.com/mpicbg-csbd/stardist/blob/master/stardist/plot/plot.py#L8
    import colorsys

    import matplotlib

    # cols = np.random.rand(n,3)
    # cols = np.random.uniform(0.1,1.0,(n,3))
    h,l,s = np.random.uniform(*h,n), np.random.uniform(*l,n), np.random.uniform(*s,n)
    cols = np.stack([colorsys.hls_to_rgb(_h,_l,_s) for _h,_l,_s in zip(h,l,s)],axis=0)
    cols[0] = 0
    return matplotlib.colors.ListedColormap(cols)


def imread_array(p: Path, dtype=None) -> np.ndarray:
    '''Read an image using PIL then convert it into numpy array'''
    img = np.asarray(Image.open(p), dtype=dtype)
    return img


def rle_decoding_inseg(
    size: Union[tuple, list],
    run_lengths: Union[pd.DataFrame, list]
) -> np.ndarray:
    """Decoding RLE (Run Length Encoding). Output binary mask. If you want each
    instance have different values, use `rle_decoding_inseg(), instead.`

    Parameters
    ----------
    size : list or tuple
        Shape of the original image array (height, width)
    run_lengths : list
        List of run length encodings
    val : int or float
        Constant value for all encoded pixels

    Returns
    -------
    decoded : numpy.ndarray
        Decoded image array

    """
    # #--- Prep ---# #
    # if `run_lengths` is pd.Dataframe, convert it to a list of numpy.ndarray
    if isinstance(run_lengths, pd.DataFrame):
        _run_lengths = []
        for e in run_lengths['EncodedPixels']:
            if not isinstance(e, np.ndarray):
                _run_lengths.append(np.array(e.split(sep=' '), dtype=int))
            else:
                _run_lengths.append(e)
        run_lengths = _run_lengths

    # #--- Draw canvas ---# #
    h, w = size[0], size[1]
    decoded = np.zeros(h * w, dtype=np.uint8)
    # num_objs = len(run_lengths)

    for i, rle in enumerate(run_lengths):
        for p, l in zip(rle[0::2], rle[1::2]):
            for dp in range(l):
                decoded[(p-1)+dp] = i + 1  # 0 is bg
    return decoded.reshape((w, h)).T


class NucleiDatasetProto(Protocol):
    file_list: list
    anno_dict: dict

    def __len__(self):
        ...


def subset(dataset: NucleiDatasetProto, indices: Sequence[int]):
    dset = deepcopy(dataset)
    dset.file_list = [dataset.file_list[i] for i in sorted(indices)]
    return dset


def random_split_dataset(
    dataset: NucleiDatasetProto,
    lengths: Sequence[int],
) -> List[NucleiDataset]:
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return [subset(dataset, indices[offset - length:offset])
            for offset, length in zip(accumulate(lengths), lengths)]


def definite_split_dataset_by_indices(
    dataset: NucleiDatasetProto,
    indices: Sequence[int],
) -> List[NucleiDataset]:
    return subset(dataset, indices)
