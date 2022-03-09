from functools import cached_property
from pathlib import Path
from typing import List, Optional

import albumentations
import cv2
import numpy as np
import tifffile

from ..base import Dataset


class BBBC002(Dataset):
    """Drosophila Kc167 cells

    There are 10 fields of view of each sample, for a total of 50 fields of
    view. The images were acquired on a Zeiss Axiovert 200M microscope. The
    images provided here are a single channel, DNA. The image size is 512 x 512
    pixels. The images are provided as 8-bit TIFF files.

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

    Notes
    -----
    - Cell count available
    - ImageJ RoI available for 3 tiles
        - CPvalid1_48_40x_Tiles_p0151DAPI_ROIs.zip
        - CPvalid1_340_40x_Tiles_p1175DAPI_ROIs.zip
        - CPvalid1_nodsRNA_40x_Tiles_p0219DAPI_ROIs.zip

    References
    ----------
    .. [1] https://bbbc.broadinstitute.org/BBBC002

    See Also
    --------
    Dataset : Base class
    DatasetInterface : Interface

    """
    # Dataset's acronym
    acronym = 'BBBC002'

    def __init__(
        self,
        root_dir: str,
        *,
        transforms: Optional[albumentations.Compose] = None,
        num_samples: Optional[int] = None,
        **kwargs
    ):
        self._root_dir = root_dir
        self._transforms = transforms
        self._num_samples = num_samples

    def get_image(self, p: Path) -> np.ndarray:
        img = tifffile.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    @cached_property
    def file_list(self) -> List[Path]:
        root_dir = self.root_dir
        parent = 'drosophila_kc167_1_images'
        file_list = sorted(root_dir.glob(f'{parent}/*.TIF'))
        return file_list
