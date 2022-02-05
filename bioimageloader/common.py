import os
from functools import cached_property
from pathlib import Path
from typing import Optional, Sequence, Union

import albumentations
import numpy as np
import tifffile

from .base import MaskDataset
from .types import KNOWN_IMAGE_EXT, PIL_IMAGE_EXT, TIFFFILE_IMAGE_EXT
from .utils import imread_asarray


class CommonMaskDataset(MaskDataset):
    """Call this from ``bioimageloader.utils.get_maskdataset_from_directory()``

    Parameters
    ----------
    root_dir
    output : optional
    transforms : optional
    num_calls : optional
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
    _setattr_ifvalue
    _filter_known_ext

    See Also
    --------
    MaskDataset : super class
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
        num_calls: Optional[int] = None,
        grayscale: Optional[bool] = None,
        grayscale_mode: Optional[Union[str, Sequence[float]]] = None,
        **kwargs
    ):
        self.acronym = f'maskdataset_{CommonMaskDataset.count}'
        self._root_dir = root_dir
        # keywords
        self._setattr_ifvalue('_output', output)
        self._setattr_ifvalue('_transforms', transforms)
        self._setattr_ifvalue('_num_calls', num_calls)
        self._setattr_ifvalue('_grayscale', grayscale)
        self._setattr_ifvalue('_grayscale_mode', grayscale_mode)
        # count # of instances
        CommonMaskDataset.count += 1

    def _setattr_ifvalue(self, attr, value=None):
        """Set attribute if value is not None"""
        if value is not None:
            setattr(self, attr, value)

    @property
    def image_dir(self) -> Optional[Path]:
        if hasattr(self, '_image_dir'):
            return self.root_dir / self._image_dir
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
            return self.root_dir / self._mask_dir
        if (n := 'labels') in os.listdir(self.root_dir):
            if (_mask_dir := self.root_dir / n).is_dir():
                return _mask_dir
        return None

    @mask_dir.setter
    def mask_dir(self, val):
        self._mask_dir = val

    @staticmethod
    def _filter_known_ext(p: Path):
        """Filter extensions supported by PIL and tifffile"""
        return p.suffix.lower() in KNOWN_IMAGE_EXT

    @cached_property
    def file_list(self):
        image_dir = self.image_dir if self.image_dir else self.root_dir
        return sorted(filter(self._filter_known_ext, image_dir.iterdir()))

    @cached_property
    def anno_dict(self):
        mask_dir = self.mask_dir if self.mask_dir else self.root_dir
        return sorted(filter(self._filter_known_ext, mask_dir.iterdir()))

    def get_image(self, p: Path) -> np.ndarray:
        if (suffix := p.suffix.lower()) in TIFFFILE_IMAGE_EXT:
            img = tifffile.imread(p)
        elif suffix in PIL_IMAGE_EXT:
            img = imread_asarray(p)
        return img

    def get_mask(self, p: Path) -> np.ndarray:
        if (suffix := p.suffix.lower()) in TIFFFILE_IMAGE_EXT:
            img = tifffile.imread(p)
        elif suffix in PIL_IMAGE_EXT:
            img = imread_asarray(p)
        return img
