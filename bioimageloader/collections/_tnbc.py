import os.path
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional

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
    TNBC : Super class
    NucleiDataset : Super class
    NucleiDatasetInterface : Interface of the super class
    """

    # Dataset's acronym
    acronym = 'TNBC'

    def __init__(
        self,
        # Interface requirement
        root_dir,
        output: str = 'both',
        transforms: Optional[albumentations.Compose] = None,
        num_calls: Optional[int] = None,
        *args, **kwargs
    ):
        """
        Parameters
        ---------
        root_dir : str or pathlib.Path
            Path to root directory
        output : {'image','mask','both'}
            Change outputs. 'both' returns {'image': image, 'mask': mask}.
            (default: 'both')
        transforms : albumentations.Compose, optional
            An instance of Compose (albumentations pkg) that defines
            augmentation in sequence.
        num_calls : int, optional
            Useful when `augmenters` is set. Define the total length of the
            dataset. If it is set, it overrides __len__.

        indices : list of int, optional
            This dataset does not provide training/testing split, one can
            provide a list of indices to load.
        contrast_inversion : bool or list, optional
            If True, pixel values of all images will be inverted within their
            data type. e.g. in case of uint8, 0 -> 255, 254 -> 1. Optionally, it
            can take a list of indicies to invert the given indices selectively.

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

    def get_image(self, p: Path) -> np.ndarray:
        img = Image.open(p)
        if img.mode == 'RGBA':
            img = img.convert(mode='RGB')
        return np.array(img)
        # if self.gray and (self.contrast_inversion is not None):
        #     img = np.array(img)
        #     if isinstance(self.contrast_inversion, (list, tuple)):
        #         ind = self.file_list.index(p)
        #         if ind in self.contrast_inversion:
        #             img = ~img
        #     else:
        #         if self.contrast_inversion:
        #             img = ~img

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
