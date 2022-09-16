import os.path
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import cv2
import numpy as np
import tifffile
from skimage.util import img_as_float32

from ..base import MaskDataset

if TYPE_CHECKING:
    import albumentations


class FRUNet(MaskDataset):
    """FRU-Net: Robust Segmentation of Small Extracellular Vesicles [1]_

    TEM images

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
    normalize : bool, default: True
        Normalize each image by its maximum value and cast it to UINT8.

    Notes
    -----
    - Originally, dtype is UINT16
    - Max value is 20444, but contrast varies a lot. For example, some images
      have value less than 0.05 of 2^16, which makes images not visible.
      Normalization may be needed. Init param ``normalize`` is set to True by
      default for this reason. For each image, it calculates maximum value and
      divide by it.

    References
    ----------
    .. [1] E. Gómez-de-Mariscal, M. Maška, A. Kotrbová, V. Pospíchalová, P.
       Matula, and A. Muñoz-Barrutia, “Deep-Learning-Based Segmentation of Small
       Extracellular Vesicles in Transmission Electron Microscopy Images,”
       Scientific Reports, vol. 9, no. 1, Art. no. 1, Sep. 2019, doi:
       10.1038/s41598-019-49431-3.

    See Also
    --------
    MaskDataset : Super class
    Dataset : Base class
    DatasetInterface : Interface

    """
    # Dataset's acronym
    acronym = 'FRUNet'

    def __init__(
        self,
        root_dir: str,
        *,
        output: str = 'both',
        transforms: Optional['albumentations.Compose'] = None,
        num_samples: Optional[int] = None,
        # specific to this dataset
        normalize: bool = True,
        **kwargs
    ):
        self._root_dir = os.path.join(root_dir, 'code', 'data')
        self._output = output
        self._transforms = transforms
        self._num_samples = num_samples
        # specific to this dataset
        self.normalize = normalize

    def get_image(self, p: Path) -> np.ndarray:
        tif = tifffile.imread(p)
        if self.normalize:
            v = tif.max()
            tif = tif / np.float32(v)
            tif = cv2.cvtColor(tif, cv2.COLOR_GRAY2RGB)
            return tif
        return img_as_float32(tif)

    def get_mask(self, p: Path) -> np.ndarray:
        mask = tifffile.imread(p)
        return mask.astype(np.int16)

    @cached_property
    def file_list(self) -> List[Path]:
        root_dir = self.root_dir
        file_list = sorted(root_dir.glob('dataset_*/*.tif'))
        return file_list

    @cached_property
    def anno_dict(self) -> Dict[int, Path]:
        root_dir = self.root_dir
        anno_list = sorted(root_dir.glob('annotations_*/*.tif'))
        anno_dict = dict((k, v) for k, v in enumerate(anno_list))
        return anno_dict
