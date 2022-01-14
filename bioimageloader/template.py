from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Union

import albumentations
import numpy as np

from bioimageloader.base import NucleiDataset


class Template(NucleiDataset):
    """Template
    """

    # Set acronym
    acronym = ''

    def __init__(
        self,
        root_dir: str,
        # optional
        output: str = 'both',
        transforms: Optional[albumentations.Compose] = None,
        num_calls: Optional[int] = None,
        # specific to this one here
        *args, **kwargs
    ):
        """
        Parameters
        ----------
        root_dir : str or pathlib.Path
            Path to root directory
        output : {'image', 'mask', 'both'}
            Change outputs. 'both' returns {'image': image, 'mask': mask}.
            (default: 'both')
        transforms : albumentations.Compose, optional
            An instance of Compose (albumentations pkg) that defines
            augmentation in sequence.
        num_calls : int
            Useful when `transforms` is set. Define the total length of the
            dataset. If it is set, it overrides __len__.

        See Also
        --------
        NucleiDataset : Super class
        DatasetInterface : Interface

        """
        self._root_dir = root_dir
        # optional
        self._output = output
        self._transforms = transforms
        self._num_calls = num_calls
        # specific to this one here

    def get_image(self, p: Path) -> np.ndarray:
        ...

    def get_mask(self, p: Path) -> np.ndarray:
        ...

    @cached_property
    def file_list(self) -> Union[List[Path], List[List[Path]]]:
        ...

    @cached_property
    def anno_dict(self) -> Dict[int, Union[Path, List[Path]]]:
        ...
