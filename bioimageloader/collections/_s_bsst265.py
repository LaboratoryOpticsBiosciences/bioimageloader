import os.path
from functools import cached_property
from pathlib import Path
from typing import Dict, Optional

import albumentations
import cv2
import numpy as np
import tifffile

from ..base import MaskDataset


class S_BSST265(MaskDataset):
    """An annotated fluorescence image dataset for training nuclear segmentation
    methods [1]_

    Immuno Fluorescence (IF) images, designed for ML

    Parameters
    ----------
    root_dir : str
        Path to root directory
    output : {'both', 'image', 'mask'}, default: 'both'
        Change outputs. 'both' returns {'image': image, 'mask': mask}.
    transforms : albumentations.Compose, optional
        An instance of Compose (albumentations pkg) that defines
        augmentation in sequence.
    num_samples : int, optional
        Useful when ``transforms`` is set. Define the total length of the
        dataset. If it is set, it overwrites ``__len__``.

    Notes
    -----
    - All images have grayscale though some have 3 channels
    - rawimages: Raw nuclear images in TIFF format
    - groundtruth: Annotated masks in TIFF format
    - groundtruth_svgs: SVG-Files for each annotated masks and corresponding raw
      image in JPEG format
    - singlecell_groundtruth: Groundtruth for randomly selected nuclei of the
      testset (25 nuclei per testset class, a subset of all nuclei of the
      testset classes; human experts can compete with this low number of nuclei
      per subset by calculating Dice coefficients between their annotations and
      the groundtruth annotations)
    - visualized_groundtruth: Visualization of groundtruth masks in PNG format
    - visualized_singlecell_groundtruth: Visualization of groundtruth for
      randomly selected nuclei in PNG format
    - Find more info in README.txt inside the root directory

    References
    ----------
    .. [1] F. Kromp et al., “An annotated fluorescence image dataset for
       training nuclear segmentation methods,” Scientific Data, vol. 7, no. 1,
       Art. no. 1, Aug. 2020, doi: 10.1038/s41597-020-00608-w.

    See Also
    --------
    MaskDataset : Super class
    Dataset : Base class
    DatasetInterface : Interface

    """
    # Dataset's acronym
    acronym = 'S_BSST265'

    def __init__(
        self,
        # Interface requirement
        root_dir: str,
        *,
        output: str = 'both',
        transforms: Optional[albumentations.Compose] = None,
        num_samples: Optional[int] = None,
        **kwargs
    ):
        # Interface and super-class arguments
        self._root_dir = os.path.join(root_dir, 'S-BSST265')
        self._output = output
        self._transforms = transforms
        self._num_samples = num_samples

    def get_image(self, p: Path) -> np.ndarray:
        tif = tifffile.imread(p)
        if tif.shape[-1] != 3:
            tif = cv2.cvtColor(tif, cv2.COLOR_GRAY2RGB)
        return tif

    def get_mask(self, p: Path) -> np.ndarray:
        tif = tifffile.imread(p)
        return tif.astype(np.int16)

    @cached_property
    def file_list(self) -> list:
        root_dir = self.root_dir
        parent = 'rawimages'
        file_list = sorted(
            root_dir.glob(f'{parent}/*.tif'), key=self._sort_key
        )
        return file_list

    @cached_property
    def anno_dict(self) -> Dict[int, Path]:
        root_dir = self.root_dir
        parent = 'groundtruth'
        anno_list = sorted(
            root_dir.glob(f'{parent}/*.tif'), key=self._sort_key
        )
        anno_dict = dict((k, v) for k, v in enumerate(anno_list))
        return anno_dict

    @staticmethod
    def _sort_key(p, zfill=2):
        split = p.stem.split('_')
        return '_'.join([s.zfill(zfill) for s in split])
