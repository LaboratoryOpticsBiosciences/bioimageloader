from functools import cached_property
from pathlib import Path
from typing import List, Optional, Sequence, Union

import albumentations
import cv2
import numpy as np
from PIL import Image

from ..base import Dataset
from ..types import BundledPath
from ..utils import bundle_list, imread_asarray, stack_channels_to_rgb


class BBBC014(Dataset):
    """Human U2OS cells cytoplasm–nucleus translocation

    This 96-well plate has images of cytoplasm to nucleus translocation of the
    transcription factor NFκB in MCF7 (human breast adenocarcinoma cell line)
    and A549 (human alveolar basal epithelial) cells in response to TNFα
    concentration.

    Images are at 10x objective magnification. The plate was acquired at Vitra
    Bioscience on the CellCard reader. For each well there is one field with two
    images: a nuclear counterstain (DAPI) image and a signal stain (FITC) image.
    Image size is 1360 x 1024 pixels. Images are in 8-bit BMP format.

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
    image_ch : {'DAPI', 'FITC'}, default: ('DAPI', 'FITC')
        Which channel(s) to load as image. Make sure to give it as a Sequence
        when choose a single channel.

    Notes
    -----
    - Second channel is usually very clear with a few artifacts
    - Biological annotation
    - CellProfiler's LoadText module format annotation also available (not
      implemented)
    - Zoom in?

    References
    ----------
    .. [1] https://bbbc.broadinstitute.org/BBBC014

    See Also
    --------
    Dataset : Base class
    DatasetInterface : Interface

    """
    # Dataset's acronym
    acronym = 'BBBC014'

    def __init__(
        self,
        root_dir: str,
        *,
        transforms: Optional[albumentations.Compose] = None,
        num_samples: Optional[int] = None,
        grayscale: bool = False,
        grayscale_mode: Union[str, Sequence[float]] = 'equal',
        # specific to this dataset
        image_ch: Sequence[str] = ('DAPI', 'FITC'),
        **kwargs
    ):
        self._root_dir = root_dir
        self._transforms = transforms
        self._num_samples = num_samples
        self._grayscale = grayscale
        self._grayscale_mode = grayscale_mode
        # specific to this dataset
        self.image_ch = image_ch
        if not any([ch in ('DAPI', 'FITC') for ch in image_ch]):
            raise ValueError("Set `image_ch` in ('DAPI', 'FITC') in sequence")

    def get_image(self, p: Union[Path, BundledPath]) -> np.ndarray:
        if isinstance(p, Path):
            img = imread_asarray(p)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = stack_channels_to_rgb(Image.open, p, 1, 2, 0)
        return img

    @cached_property
    def file_list(self) -> Union[List[Path], List[BundledPath]]:
        root_dir = self.root_dir
        parent = 'BBBC014_v1_images'
        file_list = sorted(root_dir.glob(f'{parent}/*.Bmp'), key=self._sort_key)
        if len(ch := self.image_ch) == 1:
            if ch[0] == 'DAPI':
                return file_list[::2]
            elif ch[0] == 'FITC':
                return file_list[1::2]
        return bundle_list(file_list, 2)

    @staticmethod
    def _sort_key(p: Path):
        channel, ind, t, subind, _ = p.stem.split('-')
        return '-'.join([ind, t, subind, channel])
