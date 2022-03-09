from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import albumentations
import numpy as np

from bioimageloader.base import Dataset, MaskDataset
from bioimageloader.types import BundledPath


class DatasetTemplate(Dataset):
    """Template for Dataset

    Parameters
    ----------
    root_dir : str
        Path to root directory
    transforms : albumentations.Compose, optional
        An instance of Compose (albumentations pkg) that defines augmentation in
        sequence.
    num_samples : int, optional
        Useful when ``transforms`` is set. Define the total length of the
        dataset. If it is set, it overwrites ``__len__``.
    grayscale : bool, default: False
        Convert images to grayscale
    grayscale_mode : {'cv2', 'equal', Sequence[float]}, default: 'cv2'
        How to convert to grayscale. If set to 'cv2', it follows opencv
        implementation. Else if set to 'equal', it sums up values along channel
        axis, then divides it by the number of expected channels.

    See Also
    --------
    MaskDataset : Super class
    DatasetInterface : Interface

    """
    # Set acronym
    acronym = ''

    def __init__(
        self,
        root_dir: str,
        *,  # only keyword param
        transforms: Optional[albumentations.Compose] = None,
        num_samples: Optional[int] = None,
        grayscale: bool = False,  # optional
        grayscale_mode: Union[str, Sequence[float]] = 'cv2',  # optional
        # specific to this dataset
        **kwargs
    ):
        self._root_dir = root_dir
        self._transforms = transforms
        self._num_samples = num_samples
        self._grayscale = grayscale   # optional
        self._grayscale_mode = grayscale_mode  # optional
        # specific to this one here

    def get_image(self, p: Union[Path, List[BundledPath]]) -> np.ndarray:
        ...

    @cached_property
    def file_list(self) -> Union[List[Path], List[BundledPath]]:
        # Important to decorate with `cached_property` in general
        ...


class MaskTemplate(MaskDataset):
    """Template for MaskDataset

    Parameters
    ----------
    root_dir : str
        Path to root directory
    output : {'both', 'image', 'mask'}, default: 'both'
        Change outputs. 'both' returns {'image': image, 'mask': mask}.
    transforms : albumentations.Compose, optional
        An instance of Compose (albumentations pkg) that defines augmentation in
        sequence.
    num_samples : int, optional
        Useful when ``transforms`` is set. Define the total length of the
        dataset. If it is set, it overwrites ``__len__``.
    grayscale : bool, default: False
        Convert images to grayscale
    grayscale_mode : {'cv2', 'equal', Sequence[float]}, default: 'cv2'
        How to convert to grayscale. If set to 'cv2', it follows opencv
        implementation. Else if set to 'equal', it sums up values along channel
        axis, then divides it by the number of expected channels.

    See Also
    --------
    MaskDataset : Super class
    DatasetInterface : Interface

    """
    # Set acronym
    acronym = ''

    def __init__(
        self,
        root_dir: str,
        *,  # only keyword param
        output: str = 'both',
        transforms: Optional[albumentations.Compose] = None,
        num_samples: Optional[int] = None,
        grayscale: bool = False,  # optional
        grayscale_mode: Union[str, Sequence[float]] = 'cv2',  # optional
        # specific to this dataset
        **kwargs
    ):
        self._root_dir = root_dir
        self._output = output
        self._transforms = transforms
        self._num_samples = num_samples
        self._grayscale = grayscale   # optional
        self._grayscale_mode = grayscale_mode  # optional
        # specific to this one here

    def get_image(self, p: Union[Path, List[BundledPath]]) -> np.ndarray:
        ...

    def get_mask(self, p: Union[Path, List[BundledPath]]) -> np.ndarray:
        ...

    @cached_property
    def file_list(self) -> Union[List[Path], List[BundledPath]]:
        # Important to decorate with `cached_property` in general
        ...

    @cached_property
    def anno_dict(self) -> Union[Dict[int, Path], Dict[int, BundledPath]]:
        # Important to decorate with `cached_property` in general
        ...
