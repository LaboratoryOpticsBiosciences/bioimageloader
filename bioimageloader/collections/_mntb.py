from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union, overload

import albumentations
import cv2
import numpy as np
import zarr

from ..base import ZarrDataset

class MNTB(ZarrDataset):
    """Medium Nucleus of the Trapezoidal Body

    Dataset from the Laboratory of Optics and Biosciences.

    Parameters
    ----------
    root_dir : str
        Path to root directory
    transforms : albumentations.Compose, optional
        An instance of Compose (albumentations pkg) that defines augmentation in
        sequence.
    num_calls : int, optional
        Useful when ``transforms`` is set. Define the total length of the
        dataset. If it is set, it overwrites ``__len__``.
    grayscale : bool, default: False
        Convert images to grayscale
    grayscale_mode : {'equal', 'cv2', Sequence[float]}, default: 'equal'
        How to convert to grayscale. If set to 'cv2', it follows opencv
        implementation. Else if set to 'equal', it sums up values along channel
        axis, then divides it by the number of expected channels.

    Notes
    -----
    - axis order : CZYX

    References
    ----------
    .. to come

    See Also
    --------
    ZarrDataset : Base class
    DatasetInterface : Interface

    """
    # Dataset's acronym
    acronym = 'mntb'

    def __init__(
        self,
        root_dir: str,
        *,
        transforms: Optional[albumentations.Compose] = None,
        num_calls: Optional[int] = None,
        grayscale: bool = False,
        grayscale_mode: Union[str, Sequence[float]] = 'equal',
        # specific to this dataset
        slice_shape: Tuple[int] = (3,10,100,100),
        **kwargs
    ):
        self._root_dir = root_dir
        self._transforms = transforms
        self._num_calls = num_calls
        self._grayscale = grayscale
        self._grayscale_mode = grayscale_mode
        self._slice_shape = slice_shape

    def get_image(self, p: Union[str, tuple]) -> np.ndarray:
        """Get a slice from an array located in path/to/array

        Parameters
        ----------
        p : Union[str, tuple]
            If None, will call ``get_random_slice_in_array()`` using ``slice_shape``.

        Returns
        -------
        np.ndarray
            slice of the path/to/array
        """
        array_path, slices = p
        zarr_array = zarr.open(array_path)
        img = zarr_array[slices]
        return img

    @cached_property
    def file_list(self) -> List[Union[str, tuple]]:
        """file_list for zarr format is a path to an array and slices in this array"""
        root_dir = self.root_dir
        file_list = [(root_dir/"0", None)] # None will call get_random_bbox_in_array()
        return file_list