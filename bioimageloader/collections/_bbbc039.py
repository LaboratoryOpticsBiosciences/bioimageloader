from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional

import albumentations
import cv2
import numpy as np
import scipy.ndimage as ndi
import tifffile
from PIL import Image

from ..base import MaskDataset


class BBBC039(MaskDataset):
    """Nuclei of U2OS cells in a chemical screen [1]_

    This data set has a total of 200 fields of view of nuclei captured with
    fluorescence microscopy using the Hoechst stain. These images are a sample
    of the larger BBBC022 chemical screen. The images are stored as TIFF files
    with 520x696 pixels at 16 bits.

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
    - Split (training/valiadation/test)
        - `training=True` combines 'training' with 'validation'
    - Annotate objs not touching each other with 1 and use 2, 3, ... for the
      touching ones. It is great and clever, but it does not follow the form of
      other instance segmented masks. ``get_mask()`` will make a instance
      labeled mask (each obj has unique labels). After labeling max label is 231
      for training, and 202 for test. So having masks of dtype UINT8 is fine.
    - Max label is 3 (in original annotation)
    - Sample of larger BBBC022 and did manual segmentation
    - Possible overlap some with DSB2018
    - Mask is png but (instance) value is only stored in RED channel
    - Maximum value is 2**12

    References
    ----------
    .. [1] https://bbbc.broadinstitute.org/BBBC039

    See Also
    --------
    MaskDataset : Super class
    Dataset : Base class
    DatasetInterface : Interface

    """

    # Dataset's acronym
    acronym = 'BBBC039'

    def __init__(
        self,
        root_dir: str,
        *,
        output: str = 'both',
        transforms: Optional[albumentations.Compose] = None,
        num_samples: Optional[int] = None,
        # specific to this dataset
        training: bool = True,
        **kwargs
    ):
        self._root_dir = root_dir
        self._output = output
        self._transforms = transforms
        self._num_samples = num_samples
        # specific to this dataset
        self.training = training

    def get_image(self, p: Path) -> np.ndarray:
        img = tifffile.imread(p)
        img = (img / 2**4).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def get_mask(self, p: Path) -> np.ndarray:
        mask = np.asarray(Image.open(p))[..., 0]
        max_val = mask.max()
        if max_val == 0:
            return mask
        else:
            inst_mask, n_labels = ndi.label(mask == 1, output='uint8')  # int32 by default
            for m in range(2, max_val+1):
                labeled, n = ndi.label(mask == m, output='uint8')
                inst_mask += np.where(labeled == 0, 0, labeled + n_labels)
                n_labels += n
        return inst_mask

    @cached_property
    def file_list(self) -> List[Path]:
        root_dir = self.root_dir
        parent = root_dir / 'images'
        file_list = []
        for name in self.ids:
            p = parent / name
            file_list.append(p.with_suffix('.tif'))
        return file_list

    @cached_property
    def anno_dict(self) -> Dict[int, Path]:
        root_dir = self.root_dir
        parent = root_dir / 'masks'
        anno_list = []
        for name in self.ids:
            p = parent / name
            anno_list.append(p)
        return dict((k, v) for k, v in enumerate(anno_list))

    @cached_property
    def ids(self) -> list:
        def _readlines(path):
            with open(path, 'r') as f:
                lines = f.readlines()
            return list(map(lambda s: s.strip(), lines))
        meta_dir = self.root_dir / 'metadata'
        if self.training:
            # Combine training and validation
            meta_file = meta_dir / 'training.txt'
            _ids = _readlines(meta_file)
            meta_file = meta_dir / 'validation.txt'
            _ids += _readlines(meta_file)
        else:
            meta_file = meta_dir / 'test.txt'
            _ids = _readlines(meta_file)
        return _ids
