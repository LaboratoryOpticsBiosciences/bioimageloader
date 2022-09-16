"""StarDist data[1]_ is a subset of DSB2018

.. [1] https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip
"""

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


class StarDist(MaskDataset):
    """Dataset for StarDist [1]_, [2]_

    Cell Detection with Star-convex Polygons

    StarDist data is a subset of Data Science Bowl 2018 [3]_

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
    training : bool, default: True
        Load training set if True, else load testing one

    Notes
    -----
    - StarDist data is a subset of Data Science Bowl 2018 [3]_. Choose only one,
      do not mix them.
    - ``root_dir`` is not 'dsb2018' even though the archive name is 'dsb2018',
      because it conflicts with the original DSB2018. Make a new directory.
    - All images have grayscale

    References
    ----------
    .. [1] U. Schmidt, M. Weigert, C. Broaddus, and G. Myers, “Cell Detection
       with Star-convex Polygons,” arXiv:1806.03535 [cs], vol. 11071, pp.
       265–273, 2018, doi: 10.1007/978-3-030-00934-2_30.
    .. [2] https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip
    .. [3] https://www.kaggle.com/c/data-science-bowl-2018/

    See Also
    --------
    DSB2018
    MaskDataset : Super class
    Dataset : Base class
    DatasetInterface : Interface

    """
    # Set acronym
    acronym = 'StarDist'

    def __init__(
        self,
        root_dir: str,
        *,
        output: str = 'both',
        transforms: Optional['albumentations.Compose'] = None,
        num_samples: Optional[int] = None,
        # specific to this dataset
        training: bool = True,
        **kwargs
    ):
        self._root_dir = os.path.join(root_dir, 'dsb2018')
        self._output = output
        self._transforms = transforms
        self._num_samples = num_samples
        # specific to this one here
        self.training = training

    def get_image(self, p: Path) -> np.ndarray:
        img = tifffile.imread(p)
        img = img_as_float32(img)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    def get_mask(self, p: Path) -> np.ndarray:
        mask = tifffile.imread(p)
        # originally uint16, which pytorch doesn't like
        return mask.astype(np.int16)

    @cached_property
    def file_list(self) -> List[Path]:
        # Call MaskDataset.root_dir
        parent = 'train' if self.training else 'test'
        return sorted((self.root_dir / parent / 'images').glob('*.tif'))

    @cached_property
    def anno_dict(self) -> Dict[int, Path]:
        return dict((i, p.parent.with_name('masks') / p.name)
                    for i, p in enumerate(self.file_list))
