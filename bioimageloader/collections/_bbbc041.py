import json
import os.path
from functools import cached_property
from pathlib import Path
from typing import List, Optional, Sequence, Union

import albumentations
import numpy as np

from ..base import MaskDataset
from ..utils import imread_asarray


class BBBC041(MaskDataset):
    """P. vivax (malaria) infected human blood smears

    Images are in .png or .jpg format. There are 3 sets of images consisting of
    1364 images (~80,000 cells) with different researchers having prepared each
    one: from Brazil (Stefanie Lopes), from Southeast Asia (Benoit Malleret),
    and time course (Gabriel Rangel). Blood smears were stained with Giemsa
    reagent.

    Notes
    -----
    - 1208/120 training/test split
    - .png and .jpg format; Mostly png, some jpg
    - Two resolution; seems depending on the format

    References
    ----------
    .. [1] https://bbbc.broadinstitute.org/BBBC041
    .. [2] https://github.com/tobsecret/Awesome_Malaria_Parasite_Imaging_Datasets
    """

    # Dataset's acronym
    acronym = 'BBBC041'

    def __init__(
        self,
        root_dir: str,
        *,
        transforms: Optional[albumentations.Compose] = None,
        num_calls: Optional[int] = None,
        grayscale: bool = False,
        grayscale_mode: Union[str, Sequence[float]] = 'cv2',
        # specific to this dataset
        training: bool = True,
        **kwargs
    ):
        """
        Parameters
        ----------
        root_dir : str
            Path to root directory
        transforms : albumentations.Compose, optional
            An instance of Compose (albumentations pkg) that defines
            augmentation in sequence.
        num_calls : int, optional
            Useful when `transforms` is set. Define the total length of the
            dataset. If it is set, it overrides __len__.
        grayscale : bool (default: False)
            Convert images to grayscale
        grayscale_mode : {'cv2', 'equal', Sequence[float]} (default: 'cv2')
            How to convert to grayscale. If set to 'cv2', it follows opencv
            implementation. Else if set to 'equal', it sums up values along
            channel axis, then divides it by the number of expected channels.
        training : bool (default: True)
            Load training set if True, else load testing one

        See Also
        --------
        MaskDataset : Super class
        DatasetInterface : Interface
        """
        self._root_dir = os.path.join(root_dir, 'malaria')
        self._transforms = transforms
        self._num_calls = num_calls
        self._grayscale = grayscale
        self._grayscale_mode = grayscale_mode
        # specific to this one here
        self.training = training

    def get_image(self, p: Path) -> np.ndarray:
        img = imread_asarray(p)
        return img

    @cached_property
    def file_list(self) -> List[Path]:
        root_dir = self.root_dir
        file_list = []
        for name in self.ids:
            # Dont' know why they put '/' at front in json
            file_list.append(root_dir / name.lstrip('/'))
        return file_list

    @cached_property
    def ids(self) -> list:
        return [d['image']['pathname'] for d in self.metadata]

    @cached_property
    def metadata(self) -> List[dict]:
        name = 'training.json' if self.training else 'test.json'
        p = self.root_dir / name
        with open(p, 'r') as f:
            data = json.load(f)
        return data
