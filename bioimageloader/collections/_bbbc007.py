from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional

import albumentations
import numpy as np
import tifffile

from ..base import NucleiDataset
from ..utils import bundle_list, stack_channels


class BBBC007(NucleiDataset):
    """Drosophila Kc167 cells

    Outline annotation

    Images were acquired using a motorized Zeiss Axioplan 2 and a Axiocam MRm
    camera, and are provided courtesy of the laboratory of David Sabatini at the
    Whitehead Institute for Biomedical Research. Each image is roughly 512 x 512
    pixels, with cells roughly 25 pixels in dimeter, and 80 cells per image on
    average. The two channels (DNA and actin) of each image are stored in
    separate gray-scale 8-bit TIFF files.

    Notes
    -----
    - [4, 5, 11, 14, 15] have 3 channels but they are just all gray scale
        images. Extra work is required in get_image().

    [BBBC007](https://bbbc.broadinstitute.org/BBBC007)

    .. [1]Jones et al., in the Proceedings of the ICCV Workshop on Computer
       Vision for Biomedical Image Applications (CVBIA), 2005.
    """
    # Dataset's acronym
    acronym = 'BBBC007'

    def __init__(
        self,
        root_dir: str,
        output: str = 'both',
        transforms: Optional[albumentations.Compose] = None,
        num_calls: Optional[int] = None,
        # specific to this dataset
        # anno_ch: Optional[str]='DNA',
        **kwargs
    ):
        """
        Parameters
        ----------
        root_dir : str
            Path to root directory
        output : {'image', 'mask', 'both'} (default: 'both')
            Change outputs. 'both' returns {'image': image, 'mask': mask}.
        transforms : albumentations.Compose, optional
            An instance of Compose (albumentations pkg) that defines
            augmentation in sequence.
        num_calls : int, optional
            Useful when `transforms` is set. Define the total length of the
            dataset. If it is set, it overrides __len__.

        anno_ch : {'DNA','actin'}, optional
            Which channel to use as annotation, default is 'DNA'. Set it to None
            to load both. Default is 'DNA'.

        See Also
        --------
        NucleiDataset : Super class
        DatasetInterface : Interface
        """
        self._root_dir = root_dir
        self._output = output
        self._transforms = transforms
        self._num_calls = num_calls
        # self.anno_ch = anno_ch

    def get_image(self, lst_p: List[Path]) -> np.ndarray:
        img = stack_channels(tifffile.imread, lst_p)
        return img

    def get_mask(self, p: List[Path]) -> np.ndarray:
        # mask = None
        # if self.anno_ch:
        #     mask = tifffile.imread(p)
        #     mask = binary_fill_holes_edge(mask, 50)  # 50 seems enough
        # else:
        #     mask = stack_channels(tifffile.imread, p)
        #     for c in range(mask.shape[-1]):
        #         mask[..., c] = binary_fill_holes_edge(mask[..., c], 50)
        mask = stack_channels(tifffile.imread, p)
        return mask

    @cached_property
    def file_list(self) -> List[List[Path]]:
        root_dir = self.root_dir
        parent = 'BBBC007_v1_images'
        _file_list = sorted(root_dir.glob(f'{parent}/*/*.tif'))
        file_list = bundle_list(_file_list, 2)
        return file_list

    @cached_property
    def anno_dict(self) -> Dict[int, List[Path]]:
        root_dir = self.root_dir
        parent = 'BBBC007_v1_outlines'
        _anno_list = sorted(root_dir.glob(f'{parent}/*/*.tif'))
        # if cat := self.anno_ch:
        #     if cat == 'DNA':
        #         anno_list = anno_list[::2]
        #     elif cat == 'actin':
        #         anno_list = anno_list[1::2]
        #     else:
        #         raise NotImplementedError("Set `anno_ch` to {'DNA', 'actin', None}")
        # else:
        anno_list = bundle_list(_anno_list, 2)
        anno_dict = dict((k, v) for k, v in enumerate(anno_list))
        return anno_dict
