import re
from functools import cached_property, partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import albumentations
import numpy as np
import tifffile

from ..base import MaskDataset


class UCSB(MaskDataset):
    """A biosegmentation benchmark for evaluation of bioimage analysis methods

    Parameters
    ----------
    root_dir : str
        Path to root directory
    output : {'both',' image', 'mask'}, default: 'both'
        Change outputs. 'both' returns {'image': image, 'mask': mask}.
    transforms : albumentations.Compose, optional
        An instance of Compose (albumentations pkg) that defines
        augmentation in sequence.
    num_samples : int, optional
        Useful when ``transforms`` is set. Define the total length of the
        dataset. If it is set, it overwrites ``__len__``.
    grayscale : bool, default: False
        Convert images to grayscale
    grayscale_mode : {'cv2', 'equal', Sequence[float]}, default: 'cv2'
        How to convert to grayscale. If set to 'cv2', it follows opencv
        implementation. Else if set to 'equal', it sums up values along
        channel axis, then divides it by the number of expected channels.
    category : {'benign', 'malignant'}, default: ('malignant',)
        Select which category of output you want

    See Also
    --------
    MaskDataset : Super class
    Dataset : Base class
    DatasetInterface : Interface

    Notes
    -----
    - 32 'benign', 26 'malignant' images (58 images in total)
    - 58x768x896 -> ~600 patches. Thus, the defulat `num_samples=900` (x1.5).
    - Images are not fully annotated

    References
    ----------
    .. [1] E. Drelie Gelasca, B. Obara, D. Fedorov, K. Kvilekval, and B.
       Manjunath, “A biosegmentation benchmark for evaluation of bioimage
       analysis methods,” BMC Bioinformatics, vol. 10, p. 368, Nov. 2009, doi:
       10.1186/1471-2105-10-368.
    """
    # Dataset's acronym
    acronym = 'UCSB'

    def __init__(
        self,
        root_dir: str,
        *,
        output: str = 'both',
        transforms: Optional[albumentations.Compose] = None,
        num_samples: Optional[int] = None,
        grayscale: bool = False,
        grayscale_mode: Union[str, Sequence[float]] = 'cv2',
        # specific to this dataset
        category: Sequence[str] = ('malignant',),
        **kwargs
    ):
        self._root_dir = root_dir
        self._output = output
        self._transforms = transforms
        self._num_samples = num_samples
        self._grayscale = grayscale
        self._grayscale_mode = grayscale_mode
        # specific to this dataset
        self.category = category
        if not any([cat in ('benign', 'malignant') for cat in category]):
            raise ValueError("Set `category` in ('benign', 'malignant') in sequence")

    def get_image(self, p: Path) -> np.ndarray:
        tif = tifffile.imread(p)
        return tif

    def get_mask(self, p: Path) -> np.ndarray:
        tif = tifffile.imread(p)
        return tif

    @staticmethod
    def _filter_category(p: Path, category: str):
        return re.search(category, p.stem)

    @cached_property
    def file_list(self) -> List[Path]:
        root_dir = self.root_dir
        parent = 'Breast Cancer Cells'
        file_list = root_dir.glob(f'{parent}/*.tif')
        if len(cat := self.category) == 1:
            file_list = filter(partial(self._filter_category, category=cat[0]),
                               file_list)
        return sorted(file_list)

    @cached_property
    def anno_dict(self) -> Dict[int, Path]:
        root_dir = self.root_dir
        parent = 'Breast Cancer Cells GroundTruth'
        anno_list = sorted(root_dir.glob(f'{parent}/*.TIF'))
        if len(cat := self.category) == 1:
            anno_list = filter(partial(self._filter_category, category=cat[0]),
                               anno_list)
        anno_dict = dict((k, v) for k, v in enumerate(anno_list))
        return anno_dict
