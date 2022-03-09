"""(experimental) Easily load unknown datasets as Dataset or as MaskDataset

Common dataset is a dataset which has expected structures that
bioimageloader can easily access. Simply provide a path to root directory.

This module is experimental.

Currently it assumes two cases for MaskDataset,

>>> 1. case: only images
    case1/
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

>>> 2. case: images in "images/" and labels in "labels/"
    case3/
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

Examples
--------
Case 1:

>>> dataset = CommonDataset('./Data/case1')

Case 3:

>>> dataset = CommonMaskDataset('./Data/case3')

see also ``utils.get_maskdatasets_from_directory``

>>> datset = get_maskdataset_from_directory(
        './Data/case3',
        image_dir='images',
        labels='labels',
    )
"""

import os
from functools import cached_property
from pathlib import Path
from typing import Optional, Sequence, Union

import albumentations
import numpy as np
import tifffile

from .base import Dataset, MaskDataset
from .types import KNOWN_IMAGE_EXT, PIL_IMAGE_EXT, TIFFFILE_IMAGE_EXT
from .utils import imread_asarray


class CommonDataset(Dataset):
    """Load a dataset thas has a common structure

    Parameters
    ----------
    root_dir
    output : optional
    transforms : optional
    num_samples : optional
    grayscale : optional
    grayscale_mode : optional
    num_channels : optional

    Attributes
    ----------
    image_dir
    file_list

    Methods
    -------
    get_image
    _setattr_ifvalue
    _filter_known_ext

    See Also
    --------
    Dataset : super class
    bioimageloader.utils.get_dataset_from_directory : util

    """
    count = 0
    acronym = 'dataset'

    def __init__(
        self,
        root_dir,
        *,
        output: Optional[str] = None,
        transforms: Optional[albumentations.Compose] = None,
        num_samples: Optional[int] = None,
        grayscale: Optional[bool] = None,
        grayscale_mode: Optional[Union[str, Sequence[float]]] = None,
        **kwargs
    ):
        self.acronym = f'dataset_{CommonMaskDataset.count}'
        self._root_dir = root_dir
        # keywords
        self._setattr_ifvalue('_output', output)
        self._setattr_ifvalue('_transforms', transforms)
        self._setattr_ifvalue('_num_samples', num_samples)
        self._setattr_ifvalue('_grayscale', grayscale)
        self._setattr_ifvalue('_grayscale_mode', grayscale_mode)
        # count # of instances
        CommonDataset.count += 1

    def _setattr_ifvalue(self, attr, value=None):
        """Set attribute if value is not None"""
        if value is not None:
            setattr(self, attr, value)

    @staticmethod
    def _filter_known_ext(p: Path):
        """Filter extensions supported by PIL and tifffile"""
        return p.suffix.lower() in KNOWN_IMAGE_EXT

    @cached_property
    def file_list(self):
        return sorted(filter(self._filter_known_ext, self.root_dir.iterdir()))

    def get_image(self, p: Path) -> np.ndarray:
        if (suffix := p.suffix.lower()) in TIFFFILE_IMAGE_EXT:
            img = tifffile.imread(p)
        elif suffix in PIL_IMAGE_EXT:
            img = imread_asarray(p)
        return img


class CommonMaskDataset(CommonDataset, MaskDataset):
    """Load a dataset thas has a common structure with its mask annotation

    Call this from ``bioimageloader.utils.get_maskdataset_from_directory()``

    Parameters
    ----------
    root_dir
    output : optional
    transforms : optional
    num_samples : optional
    grayscale : optional
    grayscale_mode : optional
    num_channels : optional

    Attributes
    ----------
    image_dir
    mask_dir
    file_list
    anno_dict

    Methods
    -------
    get_image
    get_mask

    See Also
    --------
    MaskDataset : super class
    CommonDataset : super class
    bioimageloader.utils.get_maskdataset_from_directory : util

    """
    count = 0
    acronym = 'maskdataset'

    def __init__(
        self,
        root_dir,
        *,
        output: Optional[str] = None,
        transforms: Optional[albumentations.Compose] = None,
        num_samples: Optional[int] = None,
        grayscale: Optional[bool] = None,
        grayscale_mode: Optional[Union[str, Sequence[float]]] = None,
        **kwargs
    ):
        super().__init__(
            root_dir=root_dir,
            output=output,
            transforms=transforms,
            num_samples=num_samples,
            grayscale=grayscale,
            grayscale_mode=grayscale_mode,
            **kwargs
        )
        self.acronym = f'maskdataset_{CommonMaskDataset.count}'
        # count # of instances
        CommonMaskDataset.count += 1

    @property
    def image_dir(self) -> Optional[Path]:
        if hasattr(self, '_image_dir'):
            if (_image_dir := self.root_dir / self._image_dir).is_dir():
                return _image_dir
            else:
                raise NotADirectoryError
        if (n := 'images') in os.listdir(self.root_dir):
            if (_image_dir := self.root_dir / n).is_dir():
                return _image_dir
        return None

    @image_dir.setter
    def image_dir(self, val):
        self._image_dir = val

    @property
    def mask_dir(self) -> Optional[Path]:
        if hasattr(self, '_mask_dir'):
            if (_mask_dir := self.root_dir / self._mask_dir).is_dir():
                return _mask_dir
            else:
                raise NotADirectoryError
            return self.root_dir / self._mask_dir
        if (n := 'labels') in os.listdir(self.root_dir):
            if (_mask_dir := self.root_dir / n).is_dir():
                return _mask_dir
        return None

    @mask_dir.setter
    def mask_dir(self, val):
        self._mask_dir = val

    @cached_property
    def file_list(self):
        image_dir = self.image_dir if self.image_dir else self.root_dir
        return sorted(filter(self._filter_known_ext, image_dir.iterdir()))

    @cached_property
    def anno_dict(self):
        mask_dir = self.mask_dir if self.mask_dir else self.root_dir
        return sorted(filter(self._filter_known_ext, mask_dir.iterdir()))

    def get_mask(self, p: Path) -> np.ndarray:
        if (suffix := p.suffix.lower()) in TIFFFILE_IMAGE_EXT:
            img = tifffile.imread(p)
        elif suffix in PIL_IMAGE_EXT:
            img = imread_asarray(p)
        return img
