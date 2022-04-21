"""Define a base class and its interface

``Dataset`` is the base of all datasets

``MaskDataset`` is the base of datasets that have mask annotation
"""

import abc
import inspect
import random
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Union

import albumentations
import cv2
import numpy as np


class DatasetInterface(metaclass=abc.ABCMeta):
    """Dataset interface

    Attributes
    ----------
    __repr__
    __len__
    __getitem__
    acronym
    root_dir
    file_list

    Methods
    -------
    get_image

    """

    @abc.abstractmethod
    def __repr__(self):  # common
        """Print info of dataset"""
        ...

    @abc.abstractmethod
    def __len__(self):  # common
        ...

    @abc.abstractmethod
    def __getitem__(self, ind):  # common
        """Given index returns a dictionary of key(s) and array"""
        ...

    @abc.abstractmethod
    def get_image(self, key):  # required
        """Get an image"""
        ...

    @classmethod
    @property
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


class Dataset(DatasetInterface):
    """Base to define common attributes and methods for [`MaskDataset`, ...]

    Attributes
    ----------
    __repr__
    __len__
    __iter__
    root_dir
    output
    transforms
    num_samples
    grayscale : optional
    grayscale_mode : optional
    num_channels : optional

    Methods
    -------
    __getitem__
    _drop_missing_pairs
    to_gray

    Notes
    -----
    Required attributes in subclass
        - ``anno_dict``
        - ``__getitem__()``
        - ``get_image()``

    """

    def __repr__(self):
        signature = inspect.signature(self.__init__)
        params = list(signature.parameters.keys())
        # remove 'kwargs'. try/except would be a better choice.
        if 'kwargs' in params:
            params.remove('kwargs')
        init_args_str = '(' + ', '.join(f'{k}={getattr(self, k)}' for k in params) + ')'
        return self.acronym + init_args_str

    def __len__(self):
        """Length of dataset. Can be overwritten with ``num_samples``"""
        if self.num_samples is not None:
            return self.num_samples
        return len(self.file_list)

    def __iter__(self):
        return IterDataset(self)

    @property
    def root_dir(self) -> Path:
        if hasattr(self, '_root_dir'):
            _root_dir = getattr(self, '_root_dir')
            if not isinstance(_root_dir, Path):
                return Path(_root_dir)
            return _root_dir
        raise NotImplementedError("Attr `_root_dir` not defined")

    @property
    def output(self) -> str:
        """Determine return(s) when called, fixed to 'image'"""
        return 'image'

    @property
    def transforms(self) -> Optional[albumentations.Compose]:
        """Transform images and masks"""
        if hasattr(self, '_transforms'):
            return getattr(self, '_transforms')
        return None

    @property
    def num_samples(self) -> Optional[int]:
        """Number of calls that will override __len__"""
        if hasattr(self, '_num_samples'):
            return getattr(self, '_num_samples')
        return None

    @num_samples.setter
    def num_samples(self, val):
        self._num_samples = val

    @property
    def grayscale(self) -> Optional[bool]:
        """Flag for grayscale conversion"""
        if hasattr(self, '_grayscale'):
            return getattr(self, '_grayscale')
        return None

    @grayscale.setter
    def grayscale(self, val):
        self._grayscale = val

    @property
    def grayscale_mode(self) -> Optional[Union[str, Sequence[float]]]:
        """Determine grayscale mode one of {'cv2', 'equal', Sequence[float]}
        """
        if hasattr(self, '_grayscale_mode'):
            return getattr(self, '_grayscale_mode')
        return None

    @grayscale_mode.setter
    def grayscale_mode(self, val):
        self._grayscale_mode = val

    @property
    def num_channels(self) -> Optional[int]:
        """Number of image channels used for `to_gray()`"""
        if hasattr(self, '_num_channels'):
            return getattr(self, '_num_channels')
        return None

    def __getitem__(self, ind: int) -> Dict[str, np.ndarray]:
        """Get image

        Dataset does not any annotation available. It will only load 'image'.

        Parameters
        ----------
        ind : int
            Index to get path(s) from ``file_list`` attribute

        Attributes
        ----------
        self.file_list

        Other Parameters
        ----------------
        self._transforms
        self._num_samples
        self._grayscale
        self._grayscale_mode
        self._num_channels

        """
        # Randomize `ind` when `num_samples` set
        if self.num_samples is not None:
            if ind >= self.num_samples:
                raise IndexError('list index out of range')
            ind_max = len(self.file_list)
            ind = random.randrange(0, ind_max)
        # `output="image"`
        p = self.file_list[ind]
        image = self.get_image(p)
        if self.grayscale:
            num_channels = self.num_channels
            # exception
            if hasattr(self, 'image_ch') and len(self.image_ch) == 1:
                # BBBC020: `_num_channels=2`. When `image_ch` is set to one
                # channel, output images become grayscale and `to_gray()`
                # got `num_channels=2`.
                num_channels = 3
            if num_channels is None:
                num_channels = len(p) if isinstance(p, list) else 3
            image = self.to_gray(
                image,
                grayscale_mode=self.grayscale_mode,
                num_channels=num_channels
            )
        if self.transforms:
            image = self.transforms(image=image)['image']
        return {'image': image}

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

    @staticmethod
    def to_gray(
        arr: np.ndarray,
        grayscale_mode: Optional[Union[str, Sequence[float]]] = None,
        num_channels: int = 3,
    ) -> np.ndarray:
        """Convert bioimage to grayscale

        Parameters
        ----------
        arr : image array
            Numpy image array whose shape is (h, w, 3)
        grayscale_mode : str or sequence of float, optional
            Choose a strategy for gray conversion. Three options are availble.
            Either one of {'cv2', 'equal'} or be a sequence of float numbers,
            which indicate linear weights of each channel.
        num_channels : int
            Explicitly set number of channels for `grayscale_mode='equal'`.
        """
        if isinstance(grayscale_mode, str):
            if grayscale_mode == 'cv2':
                if arr.shape[-1] != 3:
                    raise ValueError("Image arr should have RGB channels")
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif grayscale_mode == 'equal':
                # Expect (h, w, ch) shape of array
                arr = (arr.sum(axis=-1) / num_channels).astype(arr.dtype)
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            else:
                raise ValueError(f"Wrong `grayscale_mode={grayscale_mode}`")
        else:
            raise NotImplementedError("`grayscale_mode`")
        return arr


class MaskDataset(Dataset):
    """Base for datasets with mask annotation

    Define ``__getitem__`` method to load mask annotation paired with image.
    Pre-defined attributes are prefixed with a single underscore to distinguish
    them from those specific to a dataset. It is required to implement two
    methods: ``get_image()`` and ``get_mask()`` as well as ``acronym`` and
    ``_root_dir`` for each subclass.

    Attributes
    ----------
    output
    anno_dict

    Methods
    -------
    __getitem__

    Notes
    -----
    Required attributes in subclass
        - ``acronym``
        - ``_root_dir``
        - ``_output``
        - ``_grayscale`` (optional)
        - ``_grayscale_mode`` (optional)
        - ``_num_channels`` (optional)
        - ``get_image()``
        - ``get_mask()`` (optional)

    See Also
    --------
    Dataset : super class

    """
    @property
    def output(self) -> str:
        """Determine return(s) when called"""
        return self._output

    @output.setter
    def output(self, val):
        self._output = val

    @property
    def anno_dict(self) -> Dict[int, Any]:
        """Dictionary of pathes to annotation files"""
        raise NotImplementedError

    def __getitem__(self, ind: int) -> Dict[str, np.ndarray]:
        """Get image, mask, or both depending on ``output`` argument

        For MaskDataset, available output types are ['image', 'mask', 'both'].

        Parameters
        ----------
        ind : int
            Index to get path(s) from ``file_list`` attribute

        Attributes
        ----------
        self.output
        self.file_list
        self.anno_dict

        Other Parameters
        ----------------
        self._transforms
        self._num_samples
        self._grayscale
        self._grayscale_mode
        self._num_channels

        """
        # Randomize `ind` when `num_samples` set
        if self.num_samples is not None:
            if ind >= self.num_samples:
                raise IndexError('list index out of range')
            ind_max = len(self.file_list)
            if (self.output != 'image') and (self.anno_dict is not None):
                ind_max = len(self.anno_dict)
            ind = random.randrange(0, ind_max)
        # `output="image"`
        if self.output == 'image':
            p = self.file_list[ind]
            image = self.get_image(p)
            if self.grayscale:
                num_channels = self.num_channels
                # exception
                if hasattr(self, 'image_ch') and len(self.image_ch) == 1:
                    # BBBC020: `_num_channels=2`. When `image_ch` is set to one
                    # channel, output images become grayscale and `to_gray()`
                    # got `num_channels=2`.
                    num_channels = 3
                if num_channels is None:
                    num_channels = len(p) if isinstance(p, list) else 3
                image = self.to_gray(
                    image,
                    grayscale_mode=self.grayscale_mode,
                    num_channels=num_channels
                )
            if self.transforms:
                image = self.transforms(image=image)['image']
            return {'image': image}
        # `output="mask"`
        elif self.output == 'mask':
            pm = self.anno_dict[ind]
            mask = self.get_mask(pm)
            _image = np.zeros_like(mask, dtype=np.uint8)  # dummy image
            if self.transforms is not None:
                mask = self.transforms(image=_image, mask=mask)['mask']
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
            if self.grayscale:
                num_channels = self.num_channels
                # exception
                if hasattr(self, 'image_ch') and len(self.image_ch) == 1:
                    # BBBC020: `_num_channels=2`. When `image_ch` is set to one
                    # channel, output images become grayscale and `to_gray()`
                    # got `num_channels=2`.
                    num_channels = 3
                if num_channels is None:
                    num_channels = len(p) if isinstance(p, list) else 3
                image = self.to_gray(
                    image,
                    grayscale_mode=self.grayscale_mode,
                    num_channels=num_channels
                )
            # Make sure to apply the same augmentation both to image and mask
            if self.transforms is not None:
                augmented = self.transforms(image=image, mask=mask)
                image, mask = augmented['image'], augmented['mask']
            return {'image': image, 'mask': mask}
        else:
            raise NotImplementedError("Choose one ['image', 'mask', 'both']")

    def get_mask(self, key) -> np.ndarray:
        """Get a mask"""
        raise NotImplementedError


class IterDataset(Iterator):
    """Iterable
    """
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.ind = 0
        self.end = len(self.dataset)

    def __next__(self):
        if self.ind == self.end:
            raise StopIteration
        data = self.dataset[self.ind]
        self.ind += 1
        return data
