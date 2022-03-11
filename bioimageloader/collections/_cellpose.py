from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import albumentations
import cv2
import numpy as np

from bioimageloader.base import MaskDataset
from bioimageloader.utils import imread_asarray


class Cellpose(MaskDataset):
    """Dataset for Cellpose [1]_, [2]_

    Cellpose: a generalist algorithm for cellular segmentation

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
    training : bool, default: True
        Load training set if True, else load testing one.
    gray_is_not_green : bool, default: True
        Proper grayscale. Green channel value will be broadcast to all channels.
    specialized_data : bool, default: False
        Load "specialized data" mentioned in the paper [1]_.

    Notes
    -----
    - It is a complete dataset by itself, meaning that it is not intended to be
      mixed or concatenated with others. It consists of various sources of
      images, not only bioimages but also images of fruits, rocks and etc.
    - All images have 3 channels, but technically they are not RGB. Every images
      have values on the second channel and if there is more signal, then it
      goes to the first one. There is no image that has values on the last
      channel. As a result, when visualized in RGB, they look all green and red.
    - Built-in grayscale conversion methods are not correct for this dataset.
      The conversion should be channel-agnostic.
    - Download link is hard to find [3]_
    - Currently, ``gray_is_not_green=False`` and ``grayscale=True`` will reduce
      values of single channel images 1/3 times.

    References
    ----------
    .. [1] C. Stringer, M. Michaelos, and M. Pachitariu, “Cellpose: a generalist
       algorithm for cellular segmentation,” bioRxiv, p. 2020.02.02.931238, Feb.
       2020, doi: 10.1101/2020.02.02.931238.
    .. [2] https://github.com/mouseLand/cellpose
    .. [3] https://www.cellpose.org/dataset


    See Also
    --------
    MaskDataset : Super class
    DatasetInterface : Interface

    """
    # Set acronym
    acronym = 'Cellpose'

    names_rg_training = ['000_img.png', '001_img.png', '002_img.png',
                         '003_img.png', '004_img.png', '005_img.png',
                         '006_img.png', '007_img.png', '008_img.png',
                         '009_img.png', '010_img.png', '011_img.png',
                         '012_img.png', '013_img.png', '014_img.png',
                         '015_img.png', '016_img.png', '017_img.png',
                         '018_img.png', '019_img.png', '020_img.png',
                         '021_img.png', '022_img.png', '023_img.png',
                         '024_img.png', '025_img.png', '026_img.png',
                         '027_img.png', '028_img.png', '029_img.png',
                         '030_img.png', '031_img.png', '032_img.png',
                         '033_img.png', '034_img.png', '035_img.png',
                         '036_img.png', '037_img.png', '038_img.png',
                         '039_img.png', '040_img.png', '041_img.png',
                         '042_img.png', '043_img.png', '044_img.png',
                         '045_img.png', '046_img.png', '047_img.png',
                         '048_img.png', '049_img.png', '050_img.png',
                         '051_img.png', '052_img.png', '053_img.png',
                         '054_img.png', '055_img.png', '056_img.png',
                         '057_img.png', '058_img.png', '059_img.png',
                         '060_img.png', '061_img.png', '062_img.png',
                         '063_img.png', '064_img.png', '065_img.png',
                         '066_img.png', '067_img.png', '068_img.png',
                         '069_img.png', '070_img.png', '071_img.png',
                         '072_img.png', '073_img.png', '074_img.png',
                         '075_img.png', '076_img.png', '077_img.png',
                         '078_img.png', '079_img.png', '080_img.png',
                         '081_img.png', '082_img.png', '083_img.png',
                         '084_img.png', '085_img.png', '086_img.png',
                         '087_img.png', '088_img.png', '089_img.png',
                         '090_img.png', '091_img.png', '092_img.png',
                         '093_img.png', '094_img.png', '095_img.png',
                         '096_img.png', '097_img.png', '098_img.png',
                         '099_img.png', '100_img.png', '101_img.png',
                         '102_img.png', '103_img.png', '104_img.png',
                         '105_img.png', '106_img.png', '107_img.png',
                         '108_img.png', '109_img.png', '110_img.png',
                         '111_img.png', '112_img.png', '113_img.png',
                         '114_img.png', '115_img.png', '116_img.png',
                         '117_img.png', '118_img.png', '119_img.png',
                         '120_img.png', '121_img.png', '122_img.png',
                         '144_img.png', '145_img.png', '146_img.png',
                         '147_img.png', '148_img.png', '149_img.png',
                         '150_img.png', '151_img.png', '152_img.png',
                         '153_img.png', '154_img.png', '156_img.png',
                         '157_img.png', '161_img.png', '162_img.png',
                         '167_img.png', '169_img.png', '177_img.png',
                         '178_img.png', '180_img.png', '181_img.png',
                         '182_img.png', '183_img.png', '185_img.png',
                         '186_img.png', '187_img.png', '191_img.png',
                         '192_img.png', '193_img.png', '195_img.png',
                         '197_img.png', '198_img.png', '199_img.png',
                         '200_img.png', '201_img.png', '203_img.png',
                         '205_img.png', '206_img.png', '207_img.png',
                         '209_img.png', '210_img.png', '213_img.png',
                         '215_img.png', '218_img.png', '222_img.png',
                         '223_img.png', '225_img.png', '226_img.png',
                         '227_img.png', '228_img.png', '229_img.png',
                         '230_img.png', '231_img.png', '232_img.png',
                         '233_img.png', '234_img.png', '235_img.png',
                         '236_img.png', '237_img.png', '238_img.png',
                         '239_img.png', '240_img.png', '243_img.png',
                         '244_img.png', '246_img.png', '250_img.png',
                         '261_img.png', '264_img.png', '269_img.png',
                         '270_img.png', '271_img.png', '272_img.png',
                         '273_img.png', '274_img.png', '280_img.png',
                         '283_img.png', '284_img.png', '285_img.png',
                         '286_img.png', '287_img.png', '332_img.png',
                         '337_img.png', '340_img.png', '439_img.png']

    names_rg_test = ['000_img.png', '001_img.png', '002_img.png', '003_img.png',
                     '004_img.png', '005_img.png', '006_img.png', '007_img.png',
                     '008_img.png', '009_img.png', '010_img.png', '011_img.png',
                     '017_img.png', '019_img.png', '021_img.png', '022_img.png']

    idx_sp_train = list(range(89))
    idx_sp_test = list(range(11))

    # num_channels = 2

    def __init__(
        self,
        root_dir: str,
        *,  # only keyword param
        output: str = 'both',
        transforms: Optional[albumentations.Compose] = None,
        num_samples: Optional[int] = None,
        grayscale: bool = False,  # optional
        grayscale_mode: Union[str, Sequence[float]] = 'equal',
        # specific to this dataset
        training: bool = True,
        gray_is_not_green: bool = True,
        specialized_data: bool = False,
        **kwargs
    ):
        self._root_dir = root_dir
        self._output = output
        self._transforms = transforms
        self._num_samples = num_samples
        self._grayscale = grayscale
        self._grayscale_mode = grayscale_mode
        # specific to this one here
        self.training = training
        self.gray_is_not_green = gray_is_not_green
        self.specialized_data = specialized_data

        self.names_rg = self.names_rg_training if training else self.names_rg_test
        if specialized_data:
            self.specialized_idx = self.idx_sp_train if training else self.idx_sp_test

    def get_image(self, p: Path) -> np.ndarray:
        img = imread_asarray(p)
        if p.name in self.names_rg:
            return img
        if self.gray_is_not_green:
            return cv2.cvtColor(img[..., 1], cv2.COLOR_GRAY2RGB)
        return img

    def get_mask(self, p: Path) -> np.ndarray:
        mask = imread_asarray(p)
        return mask

    @cached_property
    def file_list(self) -> List[Path]:
        # Important to decorate with `cached_property` in general
        parent = 'train' if self.training else 'test'
        _file_list = sorted((self.root_dir / parent).glob('*_img.png'))
        if self.specialized_data:
            return [_file_list[i] for i in self.specialized_idx]
        return _file_list

    @cached_property
    def anno_dict(self) -> Dict[int, Path]:
        # Important to decorate with `cached_property` in general
        parent = 'train' if self.training else 'test'
        _anno_dict = dict((i, v) for i, v in enumerate(
            sorted((self.root_dir / parent).glob('*_masks.png'))))
        if self.specialized_data:
            return dict((i, _anno_dict[i]) for i in self.specialized_idx)
        return _anno_dict
