import os.path
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import albumentations
import numpy as np
from PIL import Image

from ..base import NucleiDataset


class TNBC(NucleiDataset):
    """TNBC Nuclei Segmentation Dataset

    References
    ----------
    Segmentation of Nuclei in Histopathology Images by Deep Regression of
    the Distance Map [1]_

    .. [1] https://ieeexplore.ieee.org/document/8438559

    See Also
    --------
    NucleiDataset : Super class
    NucleiDatasetInterface : Interface of the super class
    """

    # Dataset's acronym
    acronym = 'TNBC'

    def __init__(
        self,
        # Interface requirement
        root_dir: str,
        *,
        output: str = 'both',
        transforms: Optional[albumentations.Compose] = None,
        num_calls: Optional[int] = None,
        grayscale: bool = False,
        grayscale_mode: Union[str, Sequence[float]] = 'cv2',
        **kwargs
    ):
        """
        Parameters
        ---------
        root_dir : str
            Path to root directory
        output : {'image','mask','both'} (default: 'both')
            Change outputs. 'both' returns {'image': image, 'mask': mask}.
        transforms : albumentations.Compose, optional
            An instance of Compose (albumentations pkg) that defines
            augmentation in sequence.
        num_calls : int, optional
            Useful when `transforms` is set. Define the total length of the
            dataset. If it is set, it overrides __len__.
        grayscale : bool (default: False)
            Convert images to grayscale
        grayscale_mode : {'cv2', 'equal', Sequence[float]} (default: 'cv2')
            How to convert to grayscale. If set to 'cv2', it follows opencv
            implementation. Else if set to 'equal', it sums up values along
            channel axis, then divides it by the number of expected channels.

        See Also
        --------
        NucleiDataset : Super class
        DatasetInterface : Interface
        """
        # Interface and super-class arguments
        self._root_dir = os.path.join(root_dir, 'TNBC_NucleiSegmentation')
        self._output = output
        self._transforms = transforms
        self._num_calls = num_calls
        self._grayscale = grayscale
        self._grayscale_mode = grayscale_mode

    def get_image(self, p: Path) -> np.ndarray:
        img = Image.open(p)
        if img.mode == 'RGBA':
            img = img.convert(mode='RGB')
        return np.asarray(img)

    def get_mask(self, p: Path) -> np.ndarray:
        mask = Image.open(p)
        return np.asarray(mask)

    @cached_property
    def file_list(self) -> List[Path]:
        # Call NucleiDataset.root_dir
        root_dir = self.root_dir
        parent = 'Slide_*'
        file_list = sorted(root_dir.glob(f'{parent}/*.png'))
        return file_list
        # if self.indices:
        #     return [file_list[i] for i in self.indices]

    @cached_property
    def anno_dict(self) -> Dict[int, Path]:
        """anno_dict[ind] = <file>"""
        root_dir = self.root_dir
        parent = 'GT_*'
        anno_dict = dict((k, v) for k, v in enumerate(
            sorted(root_dir.glob(f'{parent}/*.png'))
            ))
        return anno_dict
        # if self.indices:
        #     return dict((i, anno_dict[k]) for i, k in enumerate(
        #         self.indices
        #         ))
