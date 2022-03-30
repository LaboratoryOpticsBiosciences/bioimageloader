import concurrent.futures
import re
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import albumentations
import cv2
import numpy as np
import tifffile

from ..base import MaskDataset
from ..types import BundledPath


class BBBC020(MaskDataset):
    """Murine bone-marrow derived macrophages

    The image set consists of 25 images, each consisting of three channels. The
    samples were stained with DAPI and CD11b/APC. In addition to this, a merged
    image is provided. DAPI labels the nuclei and CD11b/APC the cell surface.

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
    grayscale_mode : {'equal', 'cv2', Sequence[float]}, default: 'equal'
        How to convert to grayscale. If set to 'cv2', it follows opencv
        implementation. Else if set to 'equal', it sums up values along channel
        axis, then divides it by the number of expected channels.
    image_ch : {'cell', 'nuclei'}, default: ('cell', 'nuclei')
        Which channel(s) to load as image. Make sure to give it as a Sequence
        when choose a single channel.
    anno_ch : {'nuclei', 'cells'}, default: ('nuclei',)
        Which channel(s) to load as annotation. Make sure to give it as a
        Sequence when choose a single channel.
    drop_missing_pairs : bool, default: True
        Valid only if `output='both'`. It will drop images that do not have mask
        pairs.

    Warnings
    --------
    5 annotations are missing: ind={17,18,19,20,21}
    [jw-30min 1, jw-30min 2, jw-30min 3, jw-30min 4, jw-30min 5]

        - ./BBBC020_v1_images/jw-30min 1/jw-30min 1_(c1+c5).TIF
        - ./BBBC020_v1_images/jw-30min 2/jw-30min 2_(c1+c5).TIF
        - ./BBBC020_v1_images/jw-30min 3/jw-30min 3_(c1+c5).TIF
        - ./BBBC020_v1_images/jw-30min 4/jw-30min 4_(c1+c5).TIF
        - ./BBBC020_v1_images/jw-30min 5/jw-30min 5_(c1+c5).TIF

    - BBC020_v1_outlines_nuclei/jw-15min 5_c5_43.TIF exists but corrupted

    Notes
    -----
    - Anotations are instance segmented where each of them is saved as a single
      image file. It loads and aggregates them as a single array. Label loaded
      after will override the one loaded before. If you do not want this
      behavior, make a subclass out of this class and override ``get_mask()``
      method, accordingly.
    - 2 channels; R channel is the same as G, R==G!=B
        Assign 0 to red channel
    - BBBC has received a complaint that "BBB020_v1_outlines_nuclei" appears
      incomplete and we have been unable to obtain the missing images from the
      original contributor.
    - Nuclei anno looks good
    - Should separte nuclei and cells annotation; if ``anno_ch=None``,
      ``anno_dict`` becomes a mess.

    References
    ----------
    .. [1] https://bbbc.broadinstitute.org/BBBC020

    See Also
    --------
    MaskDataset : Super class
    Dataset : Base class
    DatasetInterface : Interface

    """

    # Dataset's acronym
    acronym = 'BBBC020'

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
        image_ch: Sequence[str] = ('nuclei', 'cells'),
        anno_ch: Sequence[str] = ('nuclei',),
        drop_missing_pairs: bool = True,
        **kwargs
    ):
        self._root_dir = root_dir
        self._output = output
        self._transforms = transforms
        self._num_samples = num_samples
        self._grayscale = grayscale
        self._grayscale_mode = grayscale_mode
        # specific to this dataset
        self._num_channels = 2  # explicit for `grayscale`
        self.image_ch = image_ch
        self.anno_ch = anno_ch
        if not any([ch in ('nuclei', 'cells') for ch in image_ch]):
            raise ValueError("Set `image_ch` in ('nuclei', 'cells') in sequence")
        if not any([ch in ('nuclei', 'cells') for ch in anno_ch]):
            raise ValueError("Set `anno_ch` in ('nuclei', 'cells') in sequence")
        self.drop_missing_pairs = drop_missing_pairs

        if self.output == 'both' and self.drop_missing_pairs:
            self.file_list, self.anno_dict = self._drop_missing_pairs()

    def get_image(self, p: Path) -> np.ndarray:
        img = tifffile.imread(p)
        # R==G, zero 0
        img[..., 0] = 0
        if len(ch := self.image_ch) == 1:
            if ch[0] == 'cells':
                return cv2.cvtColor(img[..., 1], cv2.COLOR_GRAY2RGB)
            elif ch[0] == 'nuclei':
                return cv2.cvtColor(img[..., 2], cv2.COLOR_GRAY2RGB)
            else:
                raise ValueError
        return img

    def get_mask(self, lst_p: Union[BundledPath, List[BundledPath]]) -> np.ndarray:
        def _assign_index(
            mask: np.ndarray,
            fn: Union[str, Path],
            ind: int
        ):
            """For threading"""
            tif: np.ndarray = tifffile.imread(fn)
            idx_nz = tif.nonzero()
            mask[idx_nz] = ind

        if len(self.anno_ch) == 1:
            tif: np.ndarray = tifffile.imread(lst_p[0])
            mask = np.zeros_like(tif)
            idx_nz = tif.nonzero()
            mask[idx_nz] = 1
            with concurrent.futures.ThreadPoolExecutor() as executor:
                [executor.submit(_assign_index, mask, p, ind)
                 for ind, p in enumerate(lst_p[1:], 2)]
        elif len(self.anno_ch) == 2:
            # mask_cells
            lst_p_cells = lst_p[0]
            tif: np.ndarray = tifffile.imread(lst_p_cells[0])
            mask_cells = np.zeros_like(tif)
            idx_nz = tif.nonzero()
            mask_cells[idx_nz] = 1
            # mask_nuclei
            lst_p_nuclei = lst_p[1]
            tif: np.ndarray = tifffile.imread(lst_p_nuclei[0])
            mask_nuclei = np.zeros_like(tif)
            idx_nz = tif.nonzero()
            mask_nuclei[idx_nz] = 1
            # threading
            with concurrent.futures.ThreadPoolExecutor() as executor:
                [executor.submit(_assign_index, mask_cells, p, ind)
                 for ind, p in enumerate(lst_p_cells[1:], 2)]
                [executor.submit(_assign_index, mask_nuclei, p, ind)
                 for ind, p in enumerate(lst_p_nuclei[1:], 2)]
            # 'cells' (ch=Green) first then 'nuclei' (ch=Blue)
            mask = np.stack((mask_cells, mask_nuclei), axis=-1)
        return mask

    @cached_property
    def file_list(self) -> List[Path]:
        root_dir = self.root_dir
        parent = 'BBBC020_v1_images'
        if len(ch := self.image_ch) == 1:
            if ch[0] == 'cells':
                return sorted(root_dir.glob(f'{parent}/*/*_c1.TIF'))
            elif ch[0] == 'nuclei':
                return sorted(root_dir.glob(f'{parent}/*/*_c5.TIF'))
            else:
                raise ValueError
        file_list = sorted(root_dir.glob(f'{parent}/*/*_(c1+c5).TIF'))
        return file_list

    @staticmethod
    def _sort_key(p: Path):
        res = re.search(r'\d+$', p.stem)
        if res is None:
            raise ValueError
        return int(res.group())

    @cached_property
    def anno_dict(self) -> Dict[int, BundledPath]:
        def _filter_valid_file(p: Path):
            return p.stat().st_size > 0

        root_dir = self.root_dir
        anno_dict = {}
        for i, p in enumerate(self.file_list):
            k = p.parent.stem
            # parent = root_dir / 'BBBC020_v1_outlines_*'
            if len(ch := self.anno_ch) == 1:
                parent = root_dir / f'BBBC020_v1_outlines_{ch[0]}'
                anno_list = sorted(
                    parent.glob(f'{k}_*.TIF'),
                    key=self._sort_key
                )
                anno_list = list(filter(_filter_valid_file, anno_list))
            elif len(ch) == 2:
                anno_list_cells = sorted(
                    root_dir.glob(f'BBBC020_v1_outlines_nuclei/{k}_*.TIF'),
                    key=self._sort_key
                )
                anno_list_cells = list(filter(_filter_valid_file,
                                              anno_list_cells))
                anno_list_nuclei = sorted(
                    root_dir.glob(f'BBBC020_v1_outlines_cells/{k}_*.TIF'),
                    key=self._sort_key
                )
                anno_list_nuclei = list(filter(_filter_valid_file,
                                               anno_list_nuclei))
                if anno_list_cells and anno_list_nuclei:
                    # 'cells' (ch=Green) first then 'nuclei' (ch=Blue)
                    anno_list = [anno_list_cells, anno_list_nuclei]
                else:
                    anno_list = []
            else:
                raise ValueError("Set `anno_ch` in ('nuclei', 'cells')")
            if anno_list:
                anno_dict[i] = anno_list
        return anno_dict
