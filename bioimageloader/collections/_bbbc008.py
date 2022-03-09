from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import albumentations
import cv2
import numpy as np
import tifffile

from ..base import MaskDataset
from ..types import BundledPath
from ..utils import bundle_list, stack_channels, stack_channels_to_rgb


class BBBC008(MaskDataset):
    """Human HT29 colon-cancer cells [1]_

    F/B semantic segmentation

    The image set consists of 12 images. The samples were stained with Hoechst
    (channel 1), pH3 (channel 2), and phalloidin (channel 3). Hoechst labels
    DNA, which is present in the nucleus. Phalloidin labels actin, which is
    present in the cytoplasm. The last stain, pH3, indicates cells in mitosis;
    whereas this was important for Moffat et al.'s screen, it is irrelevant for
    segmentation and counting, so this channel is left out.

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
        Useful when `transforms` is set. Define the total length of the dataset.
        If it is set, it overrides ``__len__``.
    grayscale : bool, default: False
        Convert images to grayscale
    grayscale_mode : {'cv2', 'equal', Sequence[float]}, default: 'equal'
        How to convert to grayscale. If set to 'cv2', it follows opencv
        implementation. Else if set to 'equal', it sums up values along channel
        axis, then divides it by the number of expected channels.
    image_ch : {'DNA', 'actin'}, default: ('DNA', 'actin')
        Which channel(s) to load as image. Make sure to give it as a Sequence
        when choose a single channel.
    anno_ch : {'DNA', 'actin'}, default: ('DNA',)
        Which channel(s) to load as annotation. Make sure to give it as a
        Sequence when choose a single channel.

    Notes
    -----
    - Annotation F/B: BG=1, FG=0; very annoying...

    References
    ----------
    .. [1] https://bbbc.broadinstitute.org/BBBC008
    .. [2] Carpenter et al., Genome Biology, 2006

    See Also
    --------
    MaskDataset : Super class
    Dataset : Base class
    DatasetInterface : Interface

    """
    # Dataset's acronym
    acronym = 'BBBC008'

    def __init__(
        self,
        root_dir: str,
        *,
        output: str = 'both',
        transforms: Optional[albumentations.Compose] = None,
        num_samples: Optional[int] = None,
        grayscale: bool = False,
        grayscale_mode: Union[str, Sequence[float]] = 'equal',
        # specific to this dataset
        image_ch: Sequence[str] = ('DNA', 'actin',),
        anno_ch: Sequence[str] = ('DNA',),
        **kwargs
    ):
        self._root_dir = root_dir
        self._output = output
        self._transforms = transforms
        self._num_samples = num_samples
        self._grayscale = grayscale
        self._grayscale_mode = grayscale_mode
        # specific to this dataset
        self.image_ch = image_ch
        self.anno_ch = anno_ch
        if not any([ch in ('DNA', 'actin') for ch in anno_ch]):
            raise ValueError("Set `anno_ch` in ('DNA', 'actin') in sequence")

    def get_image(self, p: Union[Path, BundledPath]) -> np.ndarray:
        if isinstance(p, Path):
            img = tifffile.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            # ch1 to red, ch3 to blue
            img = stack_channels_to_rgb(tifffile.imread, p, 0, 2, 1)
        return img

    def get_mask(self, p: Union[Path, BundledPath]) -> np.ndarray:
        if isinstance(p, Path):
            mask = tifffile.imread(p)
        else:
            mask = stack_channels(tifffile.imread, p)
        # dtype=bool originally and bool is not well handled by albumentations
        return 255 * (~mask).astype(np.uint8)

    @cached_property
    def file_list(self) -> Union[List[Path], List[BundledPath]]:
        parent = 'human_ht29_colon_cancer_2_images'
        file_list = sorted(self.root_dir.glob(f'{parent}/*.tif'))
        if len(ch := self.image_ch) == 1:
            if ch[0] == 'DNA':
                file_list = file_list[::2]
            elif ch[0] == 'actin':
                file_list = file_list[1::2]
            else:
                raise ValueError("Set `anno_ch` in ('DNA', 'actin')")
            return file_list
        elif len(ch) == 2:
            file_blist = bundle_list(file_list, 2)
        else:
            raise ValueError("Set `anno_ch` in ('DNA', 'actin') or all")
        return file_blist

    @cached_property
    def anno_dict(self) -> Union[Dict[int, Path], Dict[int, BundledPath]]:
        parent = 'human_ht29_colon_cancer_2_foreground'
        anno_list = sorted(self.root_dir.glob(f'{parent}/*.tif'))
        if len(ch := self.anno_ch) == 1:
            if ch[0] == 'DNA':
                anno_list = anno_list[::2]
            elif ch[0] == 'actin':
                anno_list = anno_list[1::2]
            return dict((k, v) for k, v in enumerate(anno_list))
        elif len(ch) == 2:
            anno_blist = bundle_list(anno_list, 2)
        else:
            raise ValueError
        return dict((k, v) for k, v in enumerate(anno_blist))
