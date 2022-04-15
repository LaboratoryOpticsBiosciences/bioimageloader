from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import albumentations
import cv2
import numpy as np
import tifffile
from PIL import Image

from ..base import MaskDataset
from ..types import BundledPath
from ..utils import bundle_list, stack_channels_to_rgb


class BBBC006(MaskDataset):
    """Human U2OS cells (out of focus)

    Images were acquired from one 384-well microplate containing U2OS cells
    stained with Hoechst 33342 markers (to label nuclei) were imaged with an
    exposure of 15 and 1000 ms for Hoechst and phalloidin respectively, at 20x
    magnification, 2x binning, and 2 sites per well. For each site, the optimal
    focus was found using laser auto-focusing to find the well bottom. The
    automated microscope was then programmed to collect a z-stack of 32 image
    sets (z = 16 at the optimal focal plane, 15 images above the focal plane, 16
    below) with 2 μm between slices. Each image is 696 x 520 pixels in 16-bit
    TIF format, LZW compression. Each image filename includes either 'w1' to
    denote Hoechst images or 'w2' to denote phalloidin images.

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
    image_ch : {'hoechst', 'phalloidin'}, default: ('hoechst', 'phalloidin')
        Which channel(s) to load as image. Make sure to give it as a Sequence
        when choose a single channel.
    uint8 : bool, default: True
        Whether to convert images to UINT8. It will divide images by a certain
        value so that they have a reasonable range of pixel values when cast
        into UINT8. If set False, no process will be applied. Read more about
        rationales in Notes section.
    z_ind : int, default: 16
        Select one z stack. Default is 16, because 16 is the most in-focus.

    TIF format, LZW compression. Each image filename includes either 'w1' to
    denote Hoechst images or 'w2' to denote phalloidin images.

    Notes
    -----
    - z-stack, z=16 is in-focus ones, sites (s1, s2)
    - Instance segmented
    - 384 wells, 2 sites per well; 384 * 2 = 768 images
    - 2 channels, w1=Hoechst, w2=phalloidin
    - Two channels usually overlap and when overlapped, it's hard to distinguish
      two channels anymore.
    - Saved in UINT16, but UINT12 practically. Max value caps at 4095.

    References
    ----------
    .. [1] https://bbbc.broadinstitute.org/BBBC006

    See Also
    --------
    MaskDataset : Super class
    Dataset : Base class
    DatasetInterface : Interface

    """
    # Dataset's acronym
    acronym = 'BBBC006'

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
        image_ch: Sequence[str] = ('hoechst', 'phalloidin'),
        uint8: bool = True,
        z_ind: int = 16,
        **kwargs
    ):
        self._root_dir = root_dir
        self._output = output
        self._transforms = transforms
        self._num_samples = num_samples
        self._grayscale = grayscale
        self._grayscale_mode = grayscale_mode
        # specific to this dataset
        self.image_ch = image_ch
        self.uint8 = uint8
        self.z_ind = z_ind
        if not any([ch in ('hoechst', 'phalloidin') for ch in image_ch]):
            raise ValueError("Set `anno_ch` in ('hoechst', 'phalloidin') in sequence")

    def get_image(self, p: Union[Path, BundledPath]) -> np.ndarray:
        if isinstance(p, Path):
            img = tifffile.imread(p)
            if self.uint8:
                img = (img / 2**4).astype(np.uint8)
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            return img
        # 2 channels
        img = stack_channels_to_rgb(tifffile.imread, p, 2, 0, 1)
        # UINT12
        if self.uint8:
            img = (img / 2**4).astype(np.uint8)
        return img

    def get_mask(self, p: Path) -> np.ndarray:
        mask = Image.open(p)
        return np.asarray(mask)

    @cached_property
    def file_list(self) -> Union[List[Path], List[BundledPath]]:
        key_ch = {
            'hoechst': 'w1',
            'phalloidin': 'w2',
        }
        root_dir = self.root_dir
        parent = f'BBBC006_v1_images_z_{self.z_ind:02d}'
        _file_list = sorted(root_dir.glob(f'{parent}/*.tif'))
        if len(self.image_ch) == 1:
            return list(filter(
                lambda p: key_ch[self.image_ch[0]] in p.stem,
                _file_list))
        return bundle_list(_file_list, 2)

    @cached_property
    def anno_dict(self) -> Dict[int, Path]:
        root_dir = self.root_dir
        parent = 'BBBC006_v1_labels'
        anno_list = sorted(root_dir.glob(f'{parent}/*.png'))
        anno_dict = dict((k, v) for k, v in enumerate(anno_list))
        return anno_dict
