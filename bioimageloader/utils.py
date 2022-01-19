import random
from copy import deepcopy
from functools import cached_property
from itertools import accumulate
from pathlib import Path
from typing import Callable, List, Protocol, Sequence, Union, Optional

import albumentations
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


def stack_channels(
    imread_handler: Callable[[Path], np.ndarray],
    p_lst: list,
    *axis_order: int
) -> np.ndarray:
    """Take a list of multi-channel images whose channels are separated in each
    file and read them in specified order.

    The order of channels follows the order of each list by default. If
    `*axis_order` is explicitely given, the function will put channels in that
    order.

    Parameters
    ----------
    imread_handler : Callable
        Func to read images e.g.) PIL.Image.open
    p_lst : list of file path
        A list of file path. Each element refers to one channel.
    axis_order : int(s)
        Additional arguments to indicate the order of channels. It should match
        the number of channels of the return. For example, 3 arguments if
        num_channels <= 3, else `n` arguments elif num_channels=`n`

    """
    images = []
    for p in p_lst:
        images.append(imread_handler(p))
    num_channels = len(images)
    if (num_axes := len(axis_order)) != 0:
        if num_channels != num_axes:
            raise ValueError
    stacked = np.stack(images, axis=-1)
    if axis_order:
        ordered = np.zeros_like(stacked)
        for i, o in enumerate(axis_order):
            ordered[..., o] = stacked[..., i]
        return ordered
    return stacked


def stack_channels_to_rgb(
    imread_handler: Callable[[Path], np.ndarray],
    p_lst: List[Path],
    *axis_order: int
) -> np.ndarray:
    """Take a list of multi-channel images whose channels are separated in each
    file and read them in specified order. If the number of channels is less
    than or equal to 3, then array will be assumed as a RGB image. Otherwise, it
    it returns an array with the same number of channels of the input.

    The order of channels follows the order of each list by default. If
    `*axis_order` is explicitely given, the function will put channels in that
    order.

    Parameters
    ----------
    imread_handler : Callable
        Func to read images e.g.) PIL.Image.open
    p_lst : a list of Paths
        A list of Path objects. Each element refers to one channel
    axis_order : int(s)
        Additional arguments to indicate the order of channels. It should match
        the number of channels of the return. For example, 3 arguments if
        num_channels <= 3, else `n` arguments elif num_channels=`n`

    """
    images = []
    for p in p_lst:
        images.append(imread_handler(p))
    num_channels = len(images)
    stacked = np.stack(images, axis=-1)
    if num_channels < 3:
        # it happens to be only 2 channels
        stacked = np.concatenate([stacked, np.zeros_like(images[0])[...,np.newaxis]], axis=-1)
    if axis_order:
        ordered = np.zeros_like(stacked)
        for i, o in enumerate(axis_order):
            ordered[..., o] = stacked[..., i]
        return ordered
    return stacked


def bundle_list(lst: list, bundle_size: int) -> List[list]:
    """Reshape a list given the repetition step size"""
    return [list(e) for e in zip(
        *[lst[i::bundle_size] for i in range(bundle_size)]
    )]


def albumentation_gray_sum(
    image: np.ndarray,
    num_channels: int,
    **kwargs
) -> np.ndarray:
    dtype = image.dtype
    image = (image / num_channels).sum(axis=-1)
    image = image.astype(dtype)
    return image


def expand_to_rgb(
    image: np.ndarray,
    dtype: Optional[str] = None,
) -> np.ndarray:
    """Expand axis of image that has 2 channels to have 3 channels mainly for
    visualization
    """
    num_channels = image.shape[-1]
    if num_channels != 2:
        raise ValueError
    # it happens to be only 2 channels
    stacked = np.concatenate(
        [image, np.zeros_like(image[..., 0])[..., np.newaxis]],
        axis=-1,
    )
    if dtype is not None:
        return stacked.astype(dtype)
    return stacked
