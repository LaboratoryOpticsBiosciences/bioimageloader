"""Define a base Dataset and its interface
"""

import abc
from pathlib import Path
from typing import Dict, Optional

import albumentations
import numpy as np


class DatasetInterface(metaclass=abc.ABCMeta):
    """Interface

    Required
    --------
    properties:
        [acronym, root_dir, file_list]
    methods:
        [get_image]

    Optional
    --------
    properties:
        [output, anno_dict, get_mask, transforms, num_calls]
        _output, _anno_dict, _get_mask, _transforms, _num_calls
    methods:
        [get_mask]

    Common
    ------
    properties:
        [__repr__, __len__]
    methos:
        [__getitem__]


    Attributes
    ----------
    methods:
        [__getitem__, __repr__, get_image, get_mask]
    properties:
        [acronym, root_dir, file_list]

    Abstract Methods (Iterable)
    ---------------------------
    __getitem__
    __len__
    __contains

    Mixin Methods
    -------------
    __contains__, __iter__, __reversed__, index, and count
    """

    @abc.abstractmethod
    def __repr__(self):  # common
        """Get summary info"""
        ...

    @property
    @abc.abstractmethod
    def __len__(self):  # common
        """Number of calls"""
        ...

    @abc.abstractmethod
    def __getitem__(self, ind):  # common
        """Given index returns a dictionary of key(s) and array"""
        ...

    @abc.abstractmethod
    def get_image(self, key):  # required
        """Get an image"""
        ...

    @property
    @classmethod
    @abc.abstractmethod
    def acronym(cls):  # required
        """Assign acroym for a subclass dataset"""
        ...

    @property
    @abc.abstractmethod
    def root_dir(self):  # required
        """Path to root directory"""
        ...

    @property
    def file_list(self):  # required
        """A list of pathes to image files"""
        ...


class NucleiDataset(DatasetInterface):
    """Concrete super class for neclei datasets

    It defines a few common methods and properties.

    Methods
    -------
    __repr__
    __getitem__
    resize

    Attributes
    ----------
    overview_talbe
    root_dir
    file_list
    output
    overview_talbe
    resolution

    Requirements for subclass
    -------------------------
        methods:
            [get_image, get_mask (optional)]
        properties:
            [acronym, __len__, _root_dir,  _output (optional),
            _resize (optional)]

    NOTE:
    * Somehow `for d in _dataset` doesn't work properly
    Use
    ```
    for ind in len(_dataset):
        d = _dataset[ind]
    ```

    """
    @property
    def root_dir(self) -> Path:
        if hasattr(self, '_root_dir'):
            _root_dir = getattr(self, '_root_dir')
            if not isinstance(_root_dir, Path):
                return Path(_root_dir)
            return _root_dir
        raise NotImplementedError('Attr `_root_dir` not defined')

    def __repr__(self):
        """Print summary info for a subclass"""
        return self.acronym

    def __len__(self):
        if self.num_calls is not None:
            return self.num_calls
        return len(self.file_list)

    def __getitem__(self, ind: int) -> Dict[str, np.ndarray]:
        """Get item by indexing. Transfrom item to Tensor if specified."""
        # Randomize `ind` when `num_calls` set
        if self.num_calls is not None:
            if ind >= self.num_calls:
                raise IndexError('list index out of range')
            ind_max = len(self.file_list)
            if (self.output != 'image') and (self.anno_dict is not None):
                ind_max = len(self.anno_dict)
            ind = np.random.randint(0, ind_max)
        # `output="image"`
        if self.output == 'image':
            p = self.file_list[ind]
            image = self.get_image(p)
            if self.transforms:
                image = self.transforms(image=image)['image']
            return {'image': image}
        # `output="mask"`
        elif self.output == 'mask':
            pm = self.anno_dict[ind]
            mask = self.get_mask(pm)
            if self.transforms is not None:
                mask = self.transforms(mask=mask)['mask']
                # # Filtering out empty masks
                # while mask.max() == 0:
                #     mask = self.transforms.augment_image(mask)
            return {'mask': mask}
        # both image and gt
        elif self.output == 'both':
            # 'image'
            p = self.file_list[ind]
            image = self.get_image(p)
            # 'mask'
            pm = self.anno_dict[ind]
            mask = self.get_mask(pm)
            # Make sure to apply the same augmentation both to image and mask
            if self.transforms is not None:
                augmented = self.transforms(image=image, mask=mask)
                image, mask = augmented['image'], augmented['mask']
            return {'image': image, 'mask': mask}
        else:
            raise NotImplementedError("Choose one ['image', 'mask', 'both']")

    @property
    def output(self) -> str:
        """If not defined, it is set to 'image'"""
        if hasattr(self, '_output'):
            return self._output
        return 'image'

    @output.setter
    def output(self, val):
        self._output = val

    @property
    def transforms(self) -> Optional[albumentations.Compose]:
        """Transform images and masks"""
        if hasattr(self, '_transforms'):
            return getattr(self, '_transforms')
        return None

    @property
    def num_calls(self) -> Optional[int]:
        """Number of calls that will override __len__"""
        if hasattr(self, '_num_calls'):
            return getattr(self, '_num_calls')
        return None

    @property
    def anno_dict(self) -> dict:
        """A dictionary of pathes to annotation files"""
        ...

    def get_mask(self, key) -> np.ndarray:
        ...

    def _drop_missing_pairs(self) -> tuple:
        """Drop images and reindex the anno list (dict)

        Sometimes, not all images have annotation. For consistence, this func
        simply drops those images missing annotation.

        For example,
        - MurphyLab
        - BBBC018
        - BBBC020
        """
        file_list = getattr(self, 'file_list')
        anno_dict = getattr(self, 'anno_dict')
        _diff = set(range(len(file_list))).difference(set(anno_dict))
        diff = sorted(_diff)
        # logger.info(f'{self.acronym}:Dropping indices: {diff}')
        for i, ind in enumerate(diff):
            file_list.pop(ind-i)
        anno_dict = dict((i, v) for i, v in enumerate(anno_dict.values()))
        return file_list, anno_dict
