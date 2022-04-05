from functools import cached_property
from pathlib import Path
from typing import List, Optional, Sequence, Union

import albumentations
import numpy as np
from PIL import Image


from ..base import MaskDataset

class BBBC030(MaskDataset):
    """Chinese Hamster Ovary Cells
    
    The image set consists of 60 Differential Interference Contrast (DIC) images of Chinese Hamster Ovary (CHO) cells. 
    The images are taken on an Olympus Cell-R microscope with a 20x lens at the time when the cell initiated their attachment to the bottom of the dish. 
    
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
    
    References
    ----------
    .. [1] https://bbbc.broadinstitute.org/BBBC030
    
    """
    # Set acronym
    acronym = 'BBBC030'

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

    def get_image(self, p: Path) -> np.ndarray:
        img = Image.open(p)
        return np.asarray(img)

    def get_mask(self, p: Path) -> np.ndarray:
        mask = Image.open(p).convert('1')
        # dtype=bool originally and bool is not well handled by albumentations
        return  255 * np.asarray(mask)

    @cached_property
    def file_list(self) -> List[Path]:
        # Important to decorate with `cached_property` in general
        root_dir = self.root_dir
        parent = 'images'
        file_list = sorted(root_dir.glob(f'{parent}/*.png'))
        return file_list

    @cached_property
    def anno_dict(self) -> List[Path]:
        # Important to decorate with `cached_property` in general
        root_dir = self.root_dir
        parent = 'ground_truth'
        file_list = sorted(root_dir.glob(f'{parent}/*.png'))
        return file_list