import os
from functools import cached_property
from pathlib import Path
from typing import Optional, Sequence, Union

import albumentations
import numpy as np
import tifffile
from PIL import Image

from .base import NucleiDataset
from .types import KNOWN_IMAGE_EXT, PIL_IMAGE_EXT, TIFFFILE_IMAGE_EXT
from .utils import imread_asarray


class GenericNucleiDataset(NucleiDataset):
    """Call this from .utils.get_nucleidataset_from_directory()

    Attributes
    ----------
    root_dir
    output (optional)
    transforms
    num_calls
    grayscale (optional)
    grayscale_mode (optional)
    num_channels (optional)
    file_list
    anno_dict (optional)
    # overview_table (not implemented yet)

    See Also
    --------
    bioimageloader.utils.get_nucleidataset_from_directory

    """
    count = 0
    acronym = 'nucleidataset'

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
        self.acronym = f'nucleidataset_{GenericNucleiDataset.count}'
        self._root_dir = root_dir
        # keywords
        self._setattr_ifvalue('_output', output)
        self._setattr_ifvalue('_transforms', transforms)
        self._setattr_ifvalue('_num_calls', num_calls)
        self._setattr_ifvalue('_grayscale', grayscale)
        self._setattr_ifvalue('_grayscale_mode', grayscale_mode)
        # count # of instances
        GenericNucleiDataset.count += 1

    def _setattr_ifvalue(self, attr, value=None):
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
    def label_dir(self) -> Optional[Path]:
        if hasattr(self, '_label_dir'):
            return self.root_dir / self._label_dir
        if (n := 'labels') in os.listdir(self.root_dir):
            if (_label_dir := self.root_dir / n).is_dir():
                return _label_dir
        return None

    @label_dir.setter
    def label_dir(self, val):
        self._label_dir = val

    @staticmethod
    def _filter_known_ext(p: Path):
        return p.suffix.lower() in KNOWN_IMAGE_EXT

    @cached_property
    def file_list(self):
        """Overwrite it"""
        image_dir = self.image_dir if self.image_dir else self.root_dir
        return sorted(filter(self._filter_known_ext, image_dir.iterdir()))

    @cached_property
    def anno_dict(self):
        """Overwrite it"""
        label_dir = self.label_dir if self.label_dir else self.root_dir
        return sorted(filter(self._filter_known_ext, label_dir.iterdir()))

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
