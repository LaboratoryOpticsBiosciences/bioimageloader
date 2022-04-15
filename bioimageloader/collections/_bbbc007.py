from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union, overload

import albumentations
import cv2
import numpy as np
import tifffile

from ..base import MaskDataset
from ..types import BundledPath
from ..utils import bundle_list, stack_channels, stack_channels_to_rgb


class BBBC007(MaskDataset):
    """Drosophila Kc167 cells

    Outline annotation

    Images were acquired using a motorized Zeiss Axioplan 2 and a Axiocam MRm
    camera, and are provided courtesy of the laboratory of David Sabatini at the
    Whitehead Institute for Biomedical Research. Each image is roughly 512 x 512
    pixels, with cells roughly 25 pixels in dimeter, and 80 cells per image on
    average. The two channels (DNA and actin) of each image are stored in
    separate gray-scale 8-bit TIFF files.

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
    grayscale_mode : {'equal', 'cv2', Sequence[float]}, default: 'equal'
        How to convert to grayscale. If set to 'cv2', it follows opencv
        implementation. Else if set to 'equal', it sums up values along channel
        axis, then divides it by the number of expected channels.
    image_ch : {'DNA', 'actin'}, default: ('DNA', 'actin')
        Which channel(s) to load as image. Make sure to give it as a Sequence
        when choose a single channel. Name matches to `anno_ch`.
    anno_ch : {'DNA', 'actin'}, default: ('DNA',)
        Which channel(s) to load as annotation. Make sure to give it as a
        Sequence when choose a single channel.

    Notes
    -----
    - [4, 5, 11, 14, 15] have 3 channels but they are just all gray scale
      images. Extra work is required in get_image().

    References
    ----------
    .. [1] Jones et al., in the Proceedings of the ICCV Workshop on Computer
       Vision for Biomedical Image Applications (CVBIA), 2005.
    .. [2] https://bbbc.broadinstitute.org/BBBC007

    See Also
    --------
    MaskDataset : Super class
    Dataset : Base class
    DatasetInterface : Interface

    """
    # Dataset's acronym
    acronym = 'BBBC007'

    def __init__(
        self,
        root_dir: str,
        *,
        output: str = 'both',
        transforms: Optional[albumentations.Compose] = None,
        num_samples: Optional[int] = None,
        grayscale: bool = False,
        grayscale_mode: Union[str, Sequence[float]] = 'equal',
        # specific to this dataset
        image_ch: Sequence[str] = ('DNA', 'actin',),
        anno_ch: Sequence[str] = ('DNA',),
        **kwargs
    ):
        self._root_dir = root_dir
        self._output = output
        self._transforms = transforms
        self._num_samples = num_samples
        self._grayscale = grayscale
        self._grayscale_mode = grayscale_mode
        self.image_ch = image_ch
        self.anno_ch = anno_ch
        if not any([ch in ('DNA', 'actin') for ch in image_ch]):
            raise ValueError("Set `anno_ch` in ('nuclei', 'cells') in sequence")
        if not any([ch in ('DNA', 'actin') for ch in anno_ch]):
            raise ValueError("Set `anno_ch` in ('nuclei', 'cells') in sequence")

    @staticmethod
    def _imread_handler(p: Path) -> np.ndarray:
        """Handle irregular images by wrapping tifffile.imread

        Normally two images in a pair have gray scale and only have one channel.
        This means that each image array has shape of (height, width). But there
        are some outliers.

        For example a pair of images below has 3 channels with all having the
        same value (height, width, 3).
        ['BBBC007_v1_images/f113/AS_09125_040701150004_A02f00d0.tif',
         'BBBC007_v1_images/f113/AS_09125_040701150004_A02f00d1.tif']

        6 pairs out of 16 have this issue and this wrapper resolves it.
        """
        img = tifffile.imread(p)
        if img.shape[-1] == 3:
            return img[..., 0]
        return img

    def get_image(self, p: Union[Path, BundledPath]) -> np.ndarray:
        if isinstance(p, Path):
            img = self._imread_handler(p)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = stack_channels_to_rgb(self._imread_handler, p)
        return img

    def get_mask(self, p: Union[Path, BundledPath]) -> np.ndarray:
        if isinstance(p, Path):
            mask = tifffile.imread(p)
        else:
            mask = stack_channels(tifffile.imread, p)
        # dtype=bool originally and bool is not well handled by albumentations
        return 255 * mask.astype(np.uint8)

    @cached_property
    def file_list(self) -> Union[List[Path], List[BundledPath]]:
        file_list: Union[List[Path], List[List[Path]]]
        root_dir = self.root_dir
        parent = 'BBBC007_v1_images'
        _file_list = sorted(root_dir.glob(f'{parent}/*/*.tif'))
        if len(ch := self.image_ch) == 1:
            if ch[0] == 'DNA':
                file_list = _file_list[::2]
            elif ch[0] == 'actin':
                file_list = _file_list[1::2]
            else:
                raise ValueError("Set `image_ch` in ('DNA', 'actin')")
        elif len(ch) == 2:
            file_list = bundle_list(_file_list, 2)
        else:
            raise ValueError("Set `image_ch` in ('DNA', 'actin') or all")
        return file_list

    @cached_property
    def anno_dict(self) -> Union[Dict[int, Path], Dict[int, BundledPath]]:
        root_dir = self.root_dir
        parent = 'BBBC007_v1_outlines'
        _anno_list = sorted(root_dir.glob(f'{parent}/*/*.tif'))
        if len(ch := self.anno_ch) == 1:
            if ch[0] == 'DNA':
                anno_list = _anno_list[::2]
            elif ch[0] == 'actin':
                anno_list = _anno_list[1::2]
            return dict((k, v) for k, v in enumerate(anno_list))
        elif len(ch) == 2:
            anno_blist = bundle_list(_anno_list, 2)
        else:
            raise ValueError("Set `anno_ch` in ('DNA', 'actin') or all")
        return dict((k, v) for k, v in enumerate(anno_blist))
