import os.path
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import albumentations
import numpy as np
import tifffile
from PIL import Image

from ..base import MaskDataset


class DigitalPathology(MaskDataset):
    """Deep learning for digital pathology image analysis: A comprehensive
    tutorial with selected use cases [1]_

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

    Notes
    -----
    - Annotation is partial
    - Boolean mask to UINT8 mask (0, 255)

    References
    ----------
    .. [1] A. Janowczyk and A. Madabhushi, “Deep learning for digital pathology
       image analysis: A comprehensive tutorial with selected use cases,” J
       Pathol Inform, vol. 7, Jul. 2016, doi: 10.4103/2153-3539.186902.

    See Also
    --------
    MaskDataset : Super class
    Dataset : Base class
    DatasetInterface : Interface

    """
    # Dataset's acronym
    acronym = 'DigitPath'

    def __init__(
        self,
        root_dir: str,
        *,
        output: str = 'both',
        transforms: Optional[albumentations.Compose] = None,
        num_samples: Optional[int] = None,
        grayscale: bool = False,
        grayscale_mode: Union[str, Sequence[float]] = 'cv2',
        **kwargs
    ):
        self._root_dir = os.path.join(root_dir, 'nuclei')
        self._output = output
        self._transforms = transforms
        self._num_samples = num_samples
        self._grayscale = grayscale
        self._grayscale_mode = grayscale_mode

    def get_image(self, p: Path) -> np.ndarray:
        tif = tifffile.imread(p)
        return tif

    def get_mask(self, p: Path) -> np.ndarray:
        mask = np.asarray(Image.open(p))
        return 255 * mask.astype(np.uint8)

    def __len__(self):
        if self.num_samples:
            return self.num_samples
        return len(self.file_list)

    @cached_property
    def file_list(self) -> List[Path]:
        root_dir = self.root_dir
        suffix = 'original'
        file_list = sorted(root_dir.glob(f'*{suffix}.tif'))
        return file_list

    @cached_property
    def anno_dict(self) -> Dict[int, Path]:
        root_dir = self.root_dir
        suffix = 'mask'
        anno_list = sorted(root_dir.glob(f'*{suffix}.png'))
        return dict((k, v) for k, v in enumerate(anno_list))
