from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

import numpy as np
from PIL import Image
from skimage.util import img_as_float32

from ..base import MaskDataset
from ..utils import imread_asarray

if TYPE_CHECKING:
    import albumentations

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
        transforms: Optional['albumentations.Compose'] = None,
        num_samples: Optional[int] = None,
        # specific to this dataset
        **kwargs
    ):
        self._root_dir = root_dir
        self._output = output
        self._transforms = transforms
        self._num_samples = num_samples
        # specific to this one here

    def get_image(self, p: Path) -> np.ndarray:
        img = imread_asarray(p)
        return img_as_float32(img)

    def get_mask(self, p: Path) -> np.ndarray:
        mask = imread_asarray(p)
        return mask

    @cached_property
    def file_list(self) -> List[Path]:
        # Important to decorate with `cached_property` in general
        root_dir = self.root_dir
        parent = 'images'
        file_list = sorted(root_dir.glob(f'{parent}/*.png'))
        return file_list

    @cached_property
    def anno_dict(self) -> Dict[int, Path]:
        # Important to decorate with `cached_property` in general
        root_dir = self.root_dir
        parent = 'ground_truth'
        anno_list = sorted(root_dir.glob(f'{parent}/*.png'))
        anno_dict = dict((k, v) for k, v in enumerate(anno_list))
        return anno_dict
