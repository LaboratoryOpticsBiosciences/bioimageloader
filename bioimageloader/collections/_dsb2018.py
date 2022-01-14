from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Union

import albumentations
import numpy as np
import pandas as pd
from PIL import Image

from ..base import NucleiDataset
from ..utils import imread_array, rle_decoding_inseg


class DSB2018(NucleiDataset):
    """Data Science Bowl 2018

    Each entry is a pair of an image and a mask.
    By default, it applies `base_transform`, which makes an image Tensor and
    have data range of uint8 [0, 255].

    Returns a dictionary, whose key is determined by `output` argument.
    """

    # Set acronym
    acronym = 'DSB2018'

    def __init__(
        self,
        root_dir: str,
        # optional
        output: str = 'both',
        transforms: Optional[albumentations.Compose] = None,
        num_calls: Optional[int] = None,
        # specific to this onem here
        training: bool = True,
        *args, **kwargs
    ):
        """
        Parameters
        ----------
        root_dir : str or pathlib.Path
            Path to root directory
        training : bool
            Load training set if True, else load testing one
        output : {'image', 'mask', 'both'}
            Change outputs. 'both' returns {'image': image, 'mask': mask}.
            (default: 'both')
        transforms : albumentations.Compose, optional
            An instance of Compose (albumentations pkg) that defines
            augmentation in sequence.
        num_calls : int
            Useful when `transforms` is set. Define the total length of the
            dataset. If it is set, it overrides __len__.

        See Also
        --------
        NucleiDataset : Super class
        DatasetInterface : Interface

        """
        self._root_dir = root_dir
        # optional
        self._output = output
        self._transforms = transforms
        self._num_calls = num_calls
        # specific to this one here
        self.training = training

    def get_image(self, p: Path) -> np.ndarray:
        img = Image.open(p)
        img = img.convert(mode='RGB')
        return np.asarray(img)

    def get_mask(self, p_lst: Union[List[Path], pd.DataFrame]) -> np.ndarray:
        """Should be called from `__getitem__()`"""
        if not self.training and isinstance(p_lst, pd.DataFrame):
            run_lengths = p_lst['EncodedPixels']
            h, w = p_lst.iloc[0][['Height', 'Width']]
            mask = rle_decoding_inseg((h, w), run_lengths)
            return mask
        p = p_lst[0]
        val = 1
        m0 = imread_array(p) > 0
        mask = np.zeros_like(m0, dtype=np.uint8)  # uint8 is enough
        mask[m0] = val
        for p in p_lst[1:]:
            val += 1
            m = imread_array(p) > 0
            # Does not allow overlapping, but since it's semantic segmentation
            # task, it's okay
            mask[m] = val
        return mask

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
