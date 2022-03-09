from functools import cached_property
from pathlib import Path
from typing import List, Optional, Sequence, Union

import albumentations
import numpy as np

from ..base import Dataset
from ..types import BundledPath
from ..utils import bundle_list, imread_asarray, stack_channels_to_rgb


class BBBC015(Dataset):
    """Human U2OS cells transfluor

    The images are of a human osteosarcoma cell line (U2OS) co-expressing beta2
    (b2AR) adrenergic receptor and arrestin-GFP protein molecules. The receptor
    was modified-type that generates "vesicle-type" spots upon ligand
    stimulation.

    The plate was acquired on iCyte imaging cytometer with iCyte software
    version 2.5.1. Image file format is JPEG with one image for green channel
    and one image for crimson channel. Image size is 1000 x 768 pixels.

    This image set has a portion of a 96-well plate containing 3 replica rows
    and 12 concentration points of isoproterenol. In each well four fields were
    acquired. File name structure: <well-number>_<field>_<channel>.JPG

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
    grayscale_mode : {'cv2', 'equal', Sequence[float]}, default: 'equal'
        How to convert to grayscale. If set to 'cv2', it follows opencv
        implementation. Else if set to 'equal', it sums up values along channel
        axis, then divides it by the number of expected channels.
    image_ch : {'b2AR', 'arrestin'}, default: ('b2AR', 'arrestin')
        Which channel(s) to load as image. Make sure to give it as a Sequence
        when choose a single channel.

    Notes
    -----
    - 2 channels (Green, Crimson?), texture in green channel
    - Crimson channel...?
    - RGB channel is all the same in each image file

    References
    ----------
    .. [1] https://bbbc.broadinstitute.org/BBBC015

    See Also
    --------
    Dataset : Base class
    DatasetInterface : Interface

    """
    # Dataset's acronym
    acronym = 'BBBC015'

    def __init__(
        self,
        root_dir: str,
        *,
        transforms: Optional[albumentations.Compose] = None,
        num_samples: Optional[int] = None,
        grayscale: bool = False,
        grayscale_mode: Union[str, Sequence[float]] = 'equal',
        # specific to this dataset
        image_ch: Sequence[str] = ('b2AR', 'arrestin'),
        **kwargs
    ):
        self._root_dir = root_dir
        self._transforms = transforms
        self._num_samples = num_samples
        self._grayscale = grayscale
        self._grayscale_mode = grayscale_mode
        # specific to this dataset
        self.image_ch = image_ch
        if not any([ch in ('b2AR', 'arrestin') for ch in image_ch]):
            raise ValueError("Set `image_ch` in ('b2AR', 'arrestin') in sequence")

    def get_image(self, p: Union[Path, BundledPath]) -> np.ndarray:
        # RGB is all the same
        if isinstance(p, Path):
            img = imread_asarray(p)
        else:
            img = stack_channels_to_rgb(
                lambda x: imread_asarray(x)[..., 0], p, 1, 0, 2
            )
        return img

    @cached_property
    def file_list(self) -> Union[List[Path], List[BundledPath]]:
        root_dir = self.root_dir
        parent = 'BBBC015_v1_images'
        file_list = sorted(root_dir.glob(f'{parent}/*.JPG'))
        if len(ch := self.image_ch) == 1:
            if ch[0] == 'b2AR':
                return file_list[::2]
            elif ch[0] == 'arrestin':
                return file_list[1::2]
        return bundle_list(file_list, 2)
