from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional

import albumentations
import numpy as np
import tifffile
from PIL import Image

from ..base import NucleiDataset
from ..utils import bundle_list, stack_channels


class BBBC006(NucleiDataset):
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

    Notes
    -----
    - z-stack, z=16 is in-focus ones, sites (s1, s2)
    - Instance segmented
    - 384 wells, 2 sites per well; 384 * 2 = 768 images
    - 2 channels, w1=Hoechst, w2=phalloidin
    - Two channels usually overlap and when overlapped, it's hard to distinguish
      two channels anymore.
    - Saved in uint16, but uint12 practically

    .. [1] https://bbbc.broadinstitute.org/BBBC006
    """
    # Dataset's acronym
    acronym = 'BBBC006'

    def __init__(
        self,
        root_dir: str,
        output: str = 'both',
        transforms: Optional[albumentations.Compose] = None,
        num_calls: Optional[int] = None,
        # specific to this dataset
        z_ind: int = 16,
        **kwargs
    ):
        """
        Parameters
        ----------
        root_dir : str
            Path to root directory
        output : {'image', 'mask', 'both'} (default: 'both')
            Change outputs. 'both' returns {'image': image, 'mask': mask}.
        transforms : albumentations.Compose, optional
            An instance of Compose (albumentations pkg) that defines
            augmentation in sequence.
        num_calls : int, optional
            Useful when `transforms` is set. Define the total length of the
            dataset. If it is set, it overrides __len__.
        z_ind : int (default: 16)
            Select one z stack. Default is 16, because 16 is the most in-focus.

        gray_mode : {'sum','PIL'}
            How to convert to gray scale; If 'sum' it will sum along channel
            axis. Else if 'PIL', use pillow gray conversion. Default is 'sum'.

        See Also
        --------
        NucleiDataset : Super class
        DatasetInterface : Interface
        """
        self._root_dir = root_dir
        self._output = output
        self._transforms = transforms
        self._num_calls = num_calls
        self.z_ind = z_ind

    def get_image(self, p: List[Path]) -> np.ndarray:
        # 2 channels
        img = stack_channels(tifffile.imread, p, 2, 0, 1)
        # # uint12
        # Should be done with albumentations.ToFloat()
        # img = (img / 2**4).astype(np.uint8)
        return img

    def get_mask(self, p: Path) -> np.ndarray:
        mask = Image.open(p)
        return np.asarray(mask)

    @cached_property
    def file_list(self) -> List[List[Path]]:
        root_dir = self.root_dir
        parent = f'BBBC006_v1_images_z_{self.z_ind:02d}'
        _file_list = sorted(root_dir.glob(f'{parent}/*.tif'))
        file_list = bundle_list(_file_list, 2)
        return file_list

    @cached_property
    def anno_dict(self) -> Dict[int, Path]:
        root_dir = self.root_dir
        parent = 'BBBC006_v1_labels'
        anno_list = sorted(root_dir.glob(f'{parent}/*.png'))
        anno_dict = dict((k, v) for k, v in enumerate(anno_list))
        return anno_dict
