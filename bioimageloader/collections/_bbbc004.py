from functools import cached_property
from pathlib import Path
from typing import List, Optional, Sequence, Union, Dict

import albumentations
import cv2
import numpy as np
import tifffile


from ..base import MaskDataset

class BBBC004(MaskDataset):
    """Synthetic cells

    Biological application

    One of the principal challenges in counting or segmenting nuclei is dealing with clustered nuclei.
    To help assess algorithms' performance in this regard, this synthetic image set consists of five
    subsets with increasing degree of clustering.

    Images

    Five subsets of 20 images each are provided. Each image contains 300 objects, but the objects overlap
    and cluster with different probabilities in the five subsets. The images were generated with the SIMCEP
    simulating platform for fluorescent cell population images (Lehmussola et al., IEEE T. Med. Imaging, 2007 and Lehmussola et al., P. IEEE, 2008).

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
    References
    ----------
    .. [1] https://bbbc.broadinstitute.org/BBBC004


    See Also
    --------
    MaskDataset : Super class
    DatasetInterface : Interface
    """
    # Set acronym
    acronym = 'BBBC004'

    def __init__(
        self,
        root_dir: str,
        *,  # only keyword param
        output: str = 'both',
        transforms: Optional[albumentations.Compose] = None,
        num_samples: Optional[int] = None,
        # specific to this dataset
        **kwargs
    ):
        self._root_dir = root_dir
        self._output = output
        self._transforms = transforms
        self._num_samples = num_samples
        # specific to this one here

    def get_image(self, p: Path) -> np.ndarray:
        img = tifffile.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def get_mask(self, p: Path) -> np.ndarray:
        mask = tifffile.imread(p)
        mask = mask[:, :, 2]
        return mask

    @cached_property
    def file_list(self) -> List[Path]:
        # Important to decorate with `cached_property` in general
        root_dir = self.root_dir
        parent = '*_images'
        file_list = sorted(root_dir.glob(f'{parent}/*.tif'), key=lambda x: int(x.stem.split("GRAY")[0]))
        return file_list

    @cached_property
    def anno_dict(self) -> Dict[int, Path]:
        # Important to decorate with `cached_property` in general
        root_dir = self.root_dir
        parent = '*_foreground'
        anno_list = sorted(root_dir.glob(f'{parent}/*.tif'), key=lambda x: int(x.stem))
        anno_dict = dict((k, v) for k, v in enumerate(anno_list))
        return anno_dict
