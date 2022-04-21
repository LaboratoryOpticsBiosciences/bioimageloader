from functools import cached_property
from pathlib import Path
from typing import List, Optional, Dict

import albumentations
import cv2
import numpy as np
import tifffile


from ..base import MaskDataset

class BBBC009(MaskDataset):
    """Human red blood cells

    This image set consists of five differential interference contrast (DIC) images of red bood cells.


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

    References
    ----------
    .. [1] https://bbbc.broadinstitute.org/BBBC009

    See Also
    --------
    MaskDataset : Super class
    DatasetInterface : Interface
    """
    # Set acronym
    acronym = 'BBBC009'

    def __init__(
        self,
        root_dir: str,
        *,  # only keyword param
        output: str = 'both',
        transforms: Optional[albumentations.Compose] = None,
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
        img = tifffile.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def get_mask(self, p: Path) -> np.ndarray:
        mask = tifffile.imread(p)
        # dtype=bool originally and bool is not well handled by albumentations
        return mask.astype(np.uint8)

    @cached_property
    def file_list(self) -> List[Path]:
        # Important to decorate with `cached_property` in general
        #file_list: Union[List[Path], List[List[Path]]]
        root_dir = self.root_dir
        parent = 'human_rbc_dic_images'
        file_list = sorted(root_dir.glob(f'{parent}/*.tif'))
        return file_list

    @cached_property
    def anno_dict(self) -> Dict[int, Path]:
        # Important to decorate with `cached_property` in general
        #file_list: Union[List[Path], List[List[Path]]]
        root_dir = self.root_dir
        parent = 'human_rbc_dic_outlines'
        anno_list = sorted(root_dir.glob(f'{parent}/*.tif'))
        anno_dict = dict((k, v) for k, v in enumerate(anno_list))
        return anno_dict
