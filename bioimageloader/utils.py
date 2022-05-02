"""Classic utils module
"""

import csv
import random
from copy import deepcopy
from itertools import accumulate
from pathlib import Path
from typing import Callable, List, Optional, Protocol, Sequence, TypeVar, Union

import albumentations
import numpy as np
from PIL import Image

from .base import Dataset, MaskDataset
from .types import Bundled

T = TypeVar('T')


class MaskDatasetProto(Protocol):
    """Static typing protocol for MaskDataset
    """
    file_list: list
    anno_dict: dict
    output: str

    def __len__(self):
        ...


def random_label_cmap(n=2**16, h=(0, 1), l=(.4, 1), s=(.2, .8)):
    """Random color map for labels (credit: StarDist team) [1]_

    Need matplotlib

    .. [1] https://github.com/stardist/stardist/blob/4422d1c235175a41d657009cb01075347cd14a53/stardist/plot/plot.py#L8
    """
    import colorsys

    import matplotlib

    # cols = np.random.rand(n,3)
    # cols = np.random.uniform(0.1,1.0,(n,3))
    h,l,s = np.random.uniform(*h,n), np.random.uniform(*l,n), np.random.uniform(*s,n)
    cols = np.stack([colorsys.hls_to_rgb(_h,_l,_s) for _h,_l,_s in zip(h,l,s)],axis=0)
    cols[0] = 0
    return matplotlib.colors.ListedColormap(cols)


def imread_asarray(p: Path, dtype=None) -> np.ndarray:
    '''Read an image using PIL then convert it into numpy array'''
    img = np.array(Image.open(p), dtype=dtype)
    return img


def read_csv(file: Union[str, Path], sniffer_siz=2048) -> tuple:
    with open(file, newline='') as csvfile:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(csvfile.readline())
        has_header = sniffer.has_header(csvfile.readline())
        csvfile.seek(0)
        reader = csv.reader(csvfile, dialect)
        header = None
        if has_header:
            header = next(reader)
        lines = [row for row in reader]
    return header, lines


def ordered_unique(seq: Sequence[T]) -> List[T]:
    unique = []
    v = None
    for _v in seq:
        if v != _v:
            unique.append(_v)
            v = _v
    return unique


def rle_decoding_inseg(
    size: Union[tuple, list],
    run_lengths: List[List[int]],
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
    # #--- Draw canvas ---# #
    h, w = size[0], size[1]
    decoded = np.zeros(h * w, dtype=np.uint8)
    # num_objs = len(run_lengths)

    for i, rle in enumerate(run_lengths):
        for p, l in zip(rle[0::2], rle[1::2]):
            for dp in range(l):
                decoded[(p-1)+dp] = i + 1  # 0 is bg
    return decoded.reshape((w, h)).T


def subset(dataset: MaskDatasetProto, indices: Sequence[int]):
    indices = sorted(indices)
    dset = deepcopy(dataset)
    dset.file_list = [dataset.file_list[i] for i in indices]
    if dataset.output != 'image':
        dset.anno_dict = dict((i, dataset.anno_dict[k]) for i, k in enumerate(
            indices
        ))
    return dset


def random_split_dataset(
    dataset: MaskDatasetProto,
    lengths: Sequence[int],
) -> List[MaskDataset]:
    """Randomly split dataset and return subsets
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return [subset(dataset, indices[offset - length:offset])
            for offset, length in zip(accumulate(lengths), lengths)]


def split_dataset_by_indices(
    dataset: MaskDatasetProto,
    indices: Sequence[int],
) -> List[MaskDataset]:
    """Split dataset given indices
    """
    return subset(dataset, indices)


def stack_channels(
    imread_handler: Callable[[Path], np.ndarray],
    p_lst: List[Path],
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
    """Many transforms work for either RGB or gray scale images. Having RGB is
    also helpful for visualization.

    Take a list of multi-channel images whose channels are separated in each
    file and read them in specified order. If the number of channels is less
    than or equal to 3, then array will be assumed as a RGB image. Otherwise, it
    it returns an array with the same number of channels of the input.

    The order of channels follows the order of each list by default. If
    `*axis_order` is explicitely given, the function will put channels in that
    order.

    Parameters
    ----------
    imread_handler : Callable
        Func to read images e.g.) PIL.Image.open | tifffile.imread
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
        stacked = np.concatenate([stacked, np.zeros_like(images[0])[..., np.newaxis]], axis=-1)
    if axis_order:
        ordered = np.zeros_like(stacked)
        for i, o in enumerate(axis_order):
            ordered[..., o] = stacked[..., i]
        return ordered
    return stacked


def bundle_list(lst: List[T], bundle_size: int) -> List[Bundled[T]]:
    """Reshape a list given the repetition step size"""
    return [list(e) for e in zip(
        *[lst[i::bundle_size] for i in range(bundle_size)]
    )]


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
        return stacked.astype(dtype, copy=False)
    return stacked


def get_dataset_from_directory(
    root_dir: str,
    *,
    output: Optional[str] = None,
    transforms: Optional[albumentations.Compose] = None,
    num_samples: Optional[int] = None,
    grayscale: Optional[bool] = None,
    grayscale_mode: Optional[Union[str, Sequence[float]]] = None,
) -> Dataset:
    """Construct MaskDataset by assuming the structure of given directory

    >>> case1/
    ├── image00.tif
    ├── image01.tif
    ├── image02.tif
    ├── image03.tif
    ├── image04.tif
    ├── image05.tif
    ├── image06.tif
    ├── image07.tif
    ├── image08.tif
    └── image09.tif

    """
    # works with case1
    # case1/                  *case2/                *case4/
    # ├── image00.tif         ├── image00.tif         ├── images
    # ├── image01.tif         ├── image01.tif         │   ├── 00.png
    # ├── image02.tif         ├── image02.tif         │   ├── 01.png
    # ├── image03.tif         ├── image03.tif         │   ├── 02.png
    # ├── image04.tif         ├── image04.tif         │   ├── 03.png
    # ├── image05.tif         ├── label00.tif         │   └── 04.png
    # ├── image06.tif         ├── label01.tif         └── labels
    # ├── image07.tif         ├── label02.tif             ├── 00
    # ├── image08.tif         ├── label03.tif             │   ├── 0.jpg
    # └── image09.tif         └── label04.tif             │   ├── 1.jpg
    #                                                     │   ├── 3.jpg
    # case3/                                              │   └── 4.jpg
    # ├── images                                          ├── 01
    # │   ├── 00.png                                      │   ├── 0.jpg
    # │   ├── 01.png                                      │   ├── 1.jpg
    # │   ├── 02.png                                      │   ├── 2.jpg
    # │   ├── 03.png                                      │   ├── 3.jpg
    # │   ├── 04.png                                      │   ├── 4.jpg
    # │   ├── 05.png                                      │   ├── 5.jpg
    # │   ├── 06.png                                      │   └── 6.jpg
    # │   ├── 07.png                                      ├── 02
    # │   ├── 08.png                                      │   ├── 0.jpg
    # │   └── 09.png                                      │   ├── 1.jpg
    # └── labels                                          │   └── 2.jpg
    #     ├── 00.tif                                      ├── 03
    #     ├── 01.tif                                      │   ├── 0.jpg
    #     ├── 02.tif                                      │   ├── 1.jpg
    #     ├── 03.tif                                      │   ├── 2.jpg
    #     ├── 04.tif                                      │   ├── 3.jpg
    #     ├── 05.tif                                      │   ├── 4.jpg
    #     ├── 06.tif                                      │   └── 5.jpg
    #     ├── 07.tif                                      └── 04
    #     ├── 08.tif                                          ├── 0.jpg
    #     └── 09.tif                                          └── 1.jpg

    from .common import CommonDataset
    mask_dataset = CommonDataset(
        root_dir=root_dir,
        output=output,
        transforms=transforms,
        num_samples=num_samples,
        grayscale=grayscale,
        grayscale_mode=grayscale_mode,
    )
    return mask_dataset


def get_maskdataset_from_directory(
    root_dir: str,
    *,
    image_dir: Optional[str] = None,
    label_dir: Optional[str] = None,
    output: Optional[str] = None,
    transforms: Optional[albumentations.Compose] = None,
    num_samples: Optional[int] = None,
    grayscale: Optional[bool] = None,
    grayscale_mode: Optional[Union[str, Sequence[float]]] = None,
) -> MaskDataset:
    """Construct MaskDataset by assuming the structure of given directory

    >>> case3/
    ├── images
    │   ├── 00.png
    │   ├── 01.png
    │   ├── 02.png
    │   ├── 03.png
    │   └── 04.png
    └── labels
        ├── 00.tif
        ├── 01.tif
        ├── 02.tif
        ├── 03.tif
        └── 04.tif

    """
    # work with case3
    # case1/                  *case2/                *case4/
    # ├── image00.tif         ├── image00.tif         ├── images
    # ├── image01.tif         ├── image01.tif         │   ├── 00.png
    # ├── image02.tif         ├── image02.tif         │   ├── 01.png
    # ├── image03.tif         ├── image03.tif         │   ├── 02.png
    # ├── image04.tif         ├── image04.tif         │   ├── 03.png
    # ├── image05.tif         ├── label00.tif         │   └── 04.png
    # ├── image06.tif         ├── label01.tif         └── labels
    # ├── image07.tif         ├── label02.tif             ├── 00
    # ├── image08.tif         ├── label03.tif             │   ├── 0.jpg
    # └── image09.tif         └── label04.tif             │   ├── 1.jpg
    #                                                     │   ├── 3.jpg
    # case3/                                              │   └── 4.jpg
    # ├── images                                          ├── 01
    # │   ├── 00.png                                      │   ├── 0.jpg
    # │   ├── 01.png                                      │   ├── 1.jpg
    # │   ├── 02.png                                      │   ├── 2.jpg
    # │   ├── 03.png                                      │   ├── 3.jpg
    # │   ├── 04.png                                      │   ├── 4.jpg
    # │   ├── 05.png                                      │   ├── 5.jpg
    # │   ├── 06.png                                      │   └── 6.jpg
    # │   ├── 07.png                                      ├── 02
    # │   ├── 08.png                                      │   ├── 0.jpg
    # │   └── 09.png                                      │   ├── 1.jpg
    # └── labels                                          │   └── 2.jpg
    #     ├── 00.tif                                      ├── 03
    #     ├── 01.tif                                      │   ├── 0.jpg
    #     ├── 02.tif                                      │   ├── 1.jpg
    #     ├── 03.tif                                      │   ├── 2.jpg
    #     ├── 04.tif                                      │   ├── 3.jpg
    #     ├── 05.tif                                      │   ├── 4.jpg
    #     ├── 06.tif                                      │   └── 5.jpg
    #     ├── 07.tif                                      └── 04
    #     ├── 08.tif                                          ├── 0.jpg
    #     └── 09.tif                                          └── 1.jpg

    from .common import CommonMaskDataset
    mask_dataset = CommonMaskDataset(
        root_dir=root_dir,
        output=output,
        transforms=transforms,
        num_samples=num_samples,
        grayscale=grayscale,
        grayscale_mode=grayscale_mode,
    )
    mask_dataset._setattr_ifvalue('_image_dir', image_dir)
    mask_dataset._setattr_ifvalue('_label_dir', label_dir)

    return mask_dataset
