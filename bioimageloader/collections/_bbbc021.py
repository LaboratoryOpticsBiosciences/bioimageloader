import re
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


class BBBC021(Dataset):
    """Human MCF7 cells – compound-profiling experiment [1]_

    The images are of MCF-7 breast cancer cells treated for 24 h with a
    collection of 113 small molecules at eight concentrations. The cells were
    fixed, labeled for DNA, F-actin, and Β-tubulin, and imaged by fluorescent
    microscopy as described [Caie et al. Molecular Cancer Therapeutics, 2010].

    There are 39,600 image files (13,200 fields of view imaged in three
    channels) in TIFF format. We provide the images in 55 ZIP archives, one for
    each microtiter plate. The archives are ~750 MB each.

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
    uint8 : bool, default: True
        Whether to convert images to UINT8. It will divide images by a certain
        value so that they have a reasonable range of pixel values when cast
        into UINT8. If set False, no process will be applied. Read more about
        rationales in Notes section.
    image_ch : {'DNA', 'actin'}, default: ('DNA', 'actin', 'tublin')
        Which channel(s) to load as image. Make sure to give it as a Sequence
        when choose a single channel.

    Notes
    -----
    - HUGE dataset
    - 3 channels
        - w1 (DNA) -> Blue
        - w2 (actin?) -> Green
        - w4 (tublin??)-> Red
    - UINT16

    References
    ----------
    .. [1] https://bbbc.broadinstitute.org/BBBC021

    See Also
    --------
    Dataset : Base class
    DatasetInterface : Interface

    """
    # Dataset's acronym
    acronym = 'BBBC021'

    def __init__(
        self,
        root_dir: str,
        *,
        transforms: Optional[albumentations.Compose] = None,
        num_samples: Optional[int] = None,
        grayscale: bool = False,
        grayscale_mode: Union[str, Sequence[float]] = 'equal',
        # # specific to this dataset
        uint8: bool = True,
        denominator: float = 2**8,
        image_ch: Sequence[str] = ('DNA', 'actin', 'tublin'),
        **kwargs
    ):
        self._root_dir = root_dir
        self._transforms = transforms
        self._num_samples = num_samples
        self._grayscale = grayscale
        self._grayscale_mode = grayscale_mode
        self.uint8 = uint8
        self.denominator = denominator  # ***
        self.image_ch = image_ch
        if not any([ch in ('DNA', 'actin', 'tublin') for ch in image_ch]):
            raise ValueError("Set `image_ch` in ('DNA', 'actin', 'tublin') in sequence")

    def get_image(self, p: Union[Path, List[Path]]) -> np.ndarray:
        # 3 channels; DAPI(w1), Tubulin(w2), Actin(w4)
        if isinstance(p, Path):
            img = tifffile.imread(p)
            if self.uint8:
                img = (img / self.denominator).astype(np.uint8)
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            return img
        if len(self.image_ch) == 2:
            def _map_ch_to_ind(p: Path):
                if 'w1' in p.stem:
                    # w1 'DNA' blue
                    return 2
                elif 'w2' in p.stem:
                    # w2 'actin' green
                    return 1
                # w4 'tublin' red
                return 0
            order = map(_map_ch_to_ind, p)
            img = stack_channels_to_rgb(tifffile.imread, p, *order)
            if self.uint8:
                img = (img / self.denominator).astype(np.uint8)
            return img
        img = stack_channels_to_rgb(tifffile.imread, p, 2, 1, 0)
        if self.uint8:
            img = (img / self.denominator).astype(np.uint8)
        return img

    @cached_property
    def file_list(self) -> Union[List[Path], List[BundledPath]]:
        root_dir = self.root_dir
        _file_list = sorted(root_dir.glob('Week*/*.tif'))
        if len(ch := self.image_ch) == 1:
            if ch[0] == 'DNA':
                return _file_list[::3]
            elif ch[0] == 'actin':
                return _file_list[1::3]
            elif ch[0] == 'tublin':
                return _file_list[2::3]
            else:
                raise ValueError
        elif len(ch) == 2:
            map_to_pat = {
                'DNA': 'w1',     # blue
                'actin': 'w2',   # green
                'tublin': 'w4',  # red
            }
            # regex pattern
            pat = '(' + '|'.join(map_to_pat[c] for c in ch) + ')'
            return bundle_list(
                list(filter(lambda p: re.search(pat, p.stem), _file_list)), 2
            )
        return bundle_list(_file_list, 3)
