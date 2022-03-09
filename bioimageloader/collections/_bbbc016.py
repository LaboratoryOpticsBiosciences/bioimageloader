from functools import cached_property
from pathlib import Path
from typing import List, Optional, Sequence, Union

import albumentations
import cv2
import numpy as np
import tifffile

from ..base import Dataset
from ..types import BundledPath
from ..utils import bundle_list, stack_channels_to_rgb


class BBBC016(Dataset):
    """Human U2OS cells transfluor

    This image set is of a Transfluor assay where an orphan GPCR is stably
    integrated into the b-arrestin GFP expressing U2OS cell line. After one hour
    incubation with a compound the cells were fixed with (formaldehyde).

    The plate was read on Cellomics ArrayScan HCS Reader using the GPCR
    Bioapplication. File format is 8-bit TIFF with one image for green channel
    (GFP) and one image for blue channel (DNA). Image size is 512 x 512 pixels.

    Parameters
    ----------
    root_dir : str
        Path to root directory
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
    image_ch : {'GFP', 'DNA'}, default: ('GFP', 'DNA')
        Which channel(s) to load as image. Make sure to give it as a Sequence
        when choose a single channel.

    Notes
    -----
    - 2 channels (G,B), nuclei are Blue

    References
    ----------
    .. [1] https://bbbc.broadinstitute.org/BBBC016

    See Also
    --------
    Dataset : Base class
    DatasetInterface : Interface

    """
    # Dataset's acronym
    acronym = 'BBBC016'

    def __init__(
        self,
        root_dir: str,
        *,
        transforms: Optional[albumentations.Compose] = None,
        num_samples: Optional[int] = None,
        grayscale: bool = False,
        grayscale_mode: Union[str, Sequence[float]] = 'equal',
        # specific to this dataset
        image_ch: Sequence[str] = ('GFP', 'DNA'),
        **kwargs
    ):
        self._root_dir = root_dir
        self._transforms = transforms
        self._num_samples = num_samples
        self._grayscale = grayscale
        self._grayscale_mode = grayscale_mode
        # specific to this dataset
        self.image_ch = image_ch
        if not any([ch in ('GFP', 'DNA') for ch in image_ch]):
            raise ValueError("Set `image_ch` in ('GFP', 'DNA') in sequence")

    def get_image(self, p: Union[Path, BundledPath]) -> np.ndarray:
        if isinstance(p, Path):
            img = tifffile.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = stack_channels_to_rgb(tifffile.imread, p, 2, 1, 0)
        return img

    @cached_property
    def file_list(self) -> Union[List[Path], List[BundledPath]]:
        root_dir = self.root_dir
        parent = 'BBBC016_v1_images'
        file_list = sorted(root_dir.glob(f'{parent}/*.TIF'))
        if len(ch := self.image_ch) == 1:
            if ch[0] == 'DNA':
                return file_list[::2]
            elif ch[0] == 'GFP':
                return file_list[1::2]
        return bundle_list(file_list, 2)
