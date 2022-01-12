from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
from PIL import Image

from bioimageloader.base import NucleiDataset
from bioimageloader.utils import imread_array, rle_decoding_inseg

class DSB2018(NucleiDataset):
    """Data Science Bowl 2018

    Each entry is a pair of an image and a mask.
    By default, it applies `base_transform`, which makes an image Tensor and
    have data range of uint8 [0, 255].

    Returns a dictionary, whose key is determined by `output` argument.
    """

    # Dataset's acronym
    acronym = 'DSB2018'

    def __init__(
        self,
        # Interface requirement
        root_dir: Union[str, Path],
        output: str = 'both',
        resize: Optional[Tuple[int, int]] = None,
        # Specific to this dataset
        training: bool = True,
        gray: bool = True,
        contrast_inversion: Optional[Union[bool, list]] = None,
        # Always good to have
        augmenters: Optional[iaa.Sequential] = None,
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
        resize : tuple(int,int), optional
            Resize output image array to the givne (height, width). Image output
            will be interpolated in the second order and mask output will be not
            be interpolated.
        training : bool or list of int
            Load training data if True, else load testing data.
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

        See Also
        --------
        NucleiDataset : Super class
        NucleiDatasetInterface : Interface of the super class

        References
        ----------
        Kaggle 2018 Data Science Bowl [1]_

        .. [1] https://www.kaggle.com/c/data-science-bowl-2018

        """
        # Interface and super-class arguments
        super().__init__(*args, **kwargs)
        self._root_dir = root_dir
        self._output = output
        self._resize = resize
        # Parameters specific this dataset
        self.training = training
        self.gray = gray
        self.contrast_inversion = contrast_inversion
        # Always good to have
        self.augmenters = augmenters
        self.num_calls = num_calls

    def get_image(self, p: Path) -> np.ndarray:
        """Should be called from `__getitem__()`"""
        img = Image.open(p)
        # Resizing, to Tensor, mul(255)
        if self.gray:
            img = img.convert(mode='L')
        else:
            # It's RGBA
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

    def get_mask(self, p_lst: Union[List[Path], pd.DataFrame]) -> np.ndarray:
        """Should be called from `__getitem__()`"""
        if not self.training and isinstance(p_lst, pd.DataFrame):
            run_lengths = p_lst['EncodedPixels']
            h, w = p_lst.iloc[0][['Height', 'Width']]
            mask = rle_decoding_inseg((h, w), run_lengths)
            if self.resize:
                mask = self._resize_arr(mask, 0)
            return mask
        # p_lst[0]
        p = p_lst[0]
        val = 1
        m0 = imread_array(p) > 0
        mask = np.zeros_like(m0, dtype=np.uint8)  # uint8 is enough
        mask[m0] = val
        # p_lst[1:]
        for p in p_lst[1:]:
            val += 1
            m = imread_array(p) > 0
            # Does not allow overlapping, but since it's semantic segmentation
            # task, it's okay
            mask[m] = val
        if self.resize:
            mask = self._resize_arr(mask, 0)
        return mask

    def __len__(self):
        if self.num_calls:
            return self.num_calls
        return len(self.file_list)

    @cached_property
    def file_list(self) -> List[Path]:
        # Call NucleiDataset.root_dir
        root_dir = self.root_dir
        parent = 'stage1_train'
        if not self.training:
            parent = 'stage1_test'
        return sorted(root_dir.glob(f'{parent}/*/images/*.png'))

    @cached_property
    def anno_dict(self) -> Dict[int, Union[List[Path], pd.DataFrame]]:
        """anno_dict[ind] = <file>"""
        anno_dict = {}
        if self.training:
            for i, p in enumerate(self.file_list):
                anno_dict[i] = list(p.parents[1].glob('masks/*.png'))
        else:
            solution = pd.read_csv(
                self.root_dir / 'stage1_solution.csv',
                index_col=0
            )
            solution['EncodedPixels'] = solution['EncodedPixels'].apply(
                lambda x: np.array(x.split(' '), dtype=int)
            )
            for i, p in enumerate(self.file_list):
                ind = p.parents[1].stem
                anno_dict[i] = solution.loc[ind]
        return dict(anno_dict)
