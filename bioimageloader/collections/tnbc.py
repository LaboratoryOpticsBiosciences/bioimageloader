import os.path
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imgaug.augmenters as iaa
import numpy as np
from PIL import Image

from bioimageloader.base import NucleiDataset


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
        resize: Optional[Tuple[int, int]] = None,
        # Specific to this dataset
        indices: Optional[list] = None,
        gray: bool = True,
        contrast_inversion: Optional[Union[bool, list]] = None,
        # Always good to have
        augmenters: iaa.Sequential = None,
        num_calls: int = None,
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
        resize : tuple(int,int), optional
            Resize output image array to the givne (height, width). Image output
            will be interpolated in the second order and mask output will be not
            be interpolated.
        indices : list of int, optional
            This dataset does not provide training/testing split, one can
            provide a list of indices to load.
        gray : bool
            Convert images to gray scale if True, default is False
        contrast_inversion : bool or list, optional
            If True, pixel values of all images will be inverted within their
            data type. e.g. in case of uint8, 0 -> 255, 254 -> 1. Optionally, it
            can take a list of indicies to invert the given indices selectively.
        augmenters : imgaug.augmenters.Sequential, optional
            An instance of Sequential object (imgaug pkg) that contains all
            augmentation.
        num_calls : int, optional
            Useful when `augmenters` is set. Define the total length of the
            dataset. If it is set, it overrides __len__.

        """
        # Interface and super-class arguments
        super().__init__(*args, **kwargs)
        self._root_dir = os.path.join(root_dir, 'TNBC_NucleiSegmentation')
        self._output = output
        self._resize = resize
        # Parameters specific this dataset
        self._indices = indices
        self.gray = gray
        self.contrast_inversion = contrast_inversion
        # Always good to have
        self.augmenters = augmenters
        self.num_calls = num_calls

    def get_image(self, p: Path) -> np.ndarray:
        img = Image.open(p)
        if self.gray:
            img = img.convert(mode='L')
        else:
            if img.mode == 'RGBA':
                img = img.convert(mode='RGB')
        if self.gray and (self.contrast_inversion is not None):
            img = np.array(img)
            if isinstance(self.contrast_inversion, (list, tuple)):
                ind = self.file_list.index(p)
                if ind in self.contrast_inversion:
                    img = ~img
            else:
                if self.contrast_inversion:
                    img = ~img
        if self.resize:
            img = self._resize_arr(img, 1)
        return np.array(img)

    def get_mask(self, p: Path) -> np.ndarray:
        mask = Image.open(p)
        if self.resize:
            mask = self._resize_arr(mask, 0)
        return np.array(mask)

    def __len__(self):
        if self.num_calls:
            return self.num_calls
        return len(self.file_list)

    @cached_property
    def file_list(self) -> List[Path]:
        # Call NucleiDataset.root_dir
        root_dir = self.root_dir
        parent = 'Slide_*'
        file_list = sorted(root_dir.glob(f'{parent}/*.png'))
        if self.indices:
            return [file_list[i] for i in self.indices]
        return file_list

    @cached_property
    def anno_dict(self) -> Dict[int, Path]:
        """anno_dict[ind] = <file>"""
        root_dir = self.root_dir
        parent = 'GT_*'
        anno_dict = dict((k, v) for k, v in enumerate(
            sorted(root_dir.glob(f'{parent}/*.png'))
            ))
        if self.indices:
            return dict((i, anno_dict[k]) for i, k in enumerate(
                self.indices
                ))
        return anno_dict
