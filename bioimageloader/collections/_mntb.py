from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union, overload

import albumentations
import cv2
import numpy as np
import zarr

from ..base import MaskDataset

class MNTB(MaskDataset):
    """Medium Nucleus of the Trapezoidal Body

    Dataset from the Laboratory of Optics and Biosciences.

    Parameters
    ----------
    root_dir : str
        Path to root directory
    output : {'image'}
        Change outputs.
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
    image_ch : {'F0', 'F1', 'F2'}, default: ('F0', 'F1', 'F2')
        Which channel(s) to load as image. Make sure to give it as a Sequence
        when choose a single channel. Name matches to `anno_ch`.

    Notes
    -----
    - F0, F1, F2 to be replaced by real fluorophores names

    References
    ----------
    .. to come

    See Also
    --------
    Dataset : Base class
    DatasetInterface : Interface

    """
    # Dataset's acronym
    acronym = 'mntb'

    def __init__(
        self,
        root_dir: str,
        *,
        output: str = 'image',
        transforms: Optional[albumentations.Compose] = None,
        num_calls: Optional[int] = None,
        grayscale: bool = False,
        grayscale_mode: Union[str, Sequence[float]] = 'equal',
        # specific to this dataset
        image_ch: Sequence[str] = ('F0', 'F1', 'F2',),
        **kwargs
    ):
        self._root_dir = root_dir
        self._output = output
        self._transforms = transforms
        self._num_calls = num_calls
        self._grayscale = grayscale
        self._grayscale_mode = grayscale_mode
        self.image_ch = image_ch

    def get_image(self, p: Path) -> np.ndarray:
        zarr_group = zarr.open(p)
        img = zarr_group["0"][:]
        return img

    @cached_property
    def file_list(self) -> List[Path]:
        root_dir = self.root_dir
        file_list = sorted(root_dir.glob(f'*.zarr'))
        return file_list