"""Define a base Dataset and its interface
"""

import abc
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypeVar, Union, Callable

import cv2
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from PIL import Image
from torch.utils.data import Dataset  # to be compatible with pytorch
from torchvision import transforms

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


class DatasetInterface(Dataset, metaclass=abc.ABCMeta):
    """Interface

    Attributes
    ----------
    torch.utils.data.Dataset:
        methods:
            [__getitem__]
            [__len__]
    cls:
        methods:
            [__repr__, get_image, get_mask]
        properties:
            [acronym, overview_talbe, root_dir, file_list]

    Abstract Methods
    ----------------
    __getitem__
    __len__
    __contains

    Mixin Methods
    -------------
    __contains__, __iter__, __reversed__, index, and count
    """

    @abc.abstractmethod
    def __repr__(self):
        """Get summary info"""
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, ind):
        """Given index returns a dictionary of key(s) and array"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_image(self, key):
        """Get an image."""
        raise NotImplementedError

    @property
    @classmethod
    @abc.abstractmethod
    def acronym(cls):
        """Assign acroym for a subclass dataset"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def overview_table(self):
        """Read and store the overview table"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def root_dir(self):
        """Path to root directory"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def tensor(self):
        """Apply base_transform in order to return Tensor."""
        return NotImplementedError

    @property
    @abc.abstractmethod
    def file_list(self):
        """A list of pathes to image files"""
        raise NotImplementedError


class NucleiDataset(DatasetInterface):
    """Concrete super class for neclei datasets

    It defines a few common methods and properties.

    Methods
    -------
    __repr__
    __getitem__
    build_totensor
    resize

    Attributes
    ----------
    overview_talbe
    root_dir
    file_list
    totensor_image
    totensor_mask
    output
    tensor
    overview_talbe
    resolution

    Requirements for subclass:
        torch.utils.data.Dataset:
                methods:
                    [__len__ (optional)]
        subcls:
                methods:
                    [get_image, get_mask (optional)]
                properties:
                    [acronym, _root_dir, _tensor, _output (optional), _resize
                     (optional)]

    NOTE:
    * Somehow `for d in _dataset` doesn't work properly
    Use
    ```
    for ind in len(_dataset):
        d = _dataset[ind]
    ```

    """

    def __init__(self, *args, **kwargs):
        # Float number will cast array into torch.float32
        self.totensor_image = self.build_totensor(255.0)
        self.totensor_mask = self.build_totensor(1.0)

    def __repr__(self):
        """Print summary info for a subclass"""
        return self.acronym

    def __getitem__(self, ind) -> Dict[str, np.ndarray]:
        """Get item by indexing. Transfrom item to Tensor if specified."""
        # `augmenters` from imgaug library
        augmenters: Optional[iaa.Sequential] = None
        if hasattr(self, 'augmenters') and getattr(self, 'augmenters'):
            augmenters = getattr(self, 'augmenters')
        # Randomize `ind` when `num_calls` set
        if hasattr(self, 'num_calls') and getattr(self, 'num_calls'):
            num_calls = getattr(self, 'num_calls')
            if ind >= num_calls:
                raise IndexError('list index out of range')
            ind_max = len(self.file_list)
            if self.output != 'image' and hasattr(self, 'anno_dict'):
                ind_max = len(getattr(self, 'anno_dict'))
            ind = np.random.randint(0, ind_max)
        # `output=image`
        if self.output == 'image':
            p = self.file_list[ind]
            image = self.get_image(p)
            if augmenters:
                image = augmenters.augment_image(image)
            if self.tensor:
                image = self.totensor_image(image)
            return {'image': image}
        # `output=gt`
        elif self.output == 'mask':
            anno_dict = getattr(self, 'anno_dict')
            get_mask = getattr(self, 'get_mask')
            pm = anno_dict[ind]
            mask = get_mask(pm)
            if augmenters:
                mask = augmenters.augment_image(mask)
                # # Filtering out empty masks
                # while mask.max() == 0:
                #     mask = self.augmenters.augment_image(mask)
            if self.tensor:
                mask = self.totensor_mask(mask)
            return {'mask': mask}
        # both image and gt
        elif self.output == 'both':
            # 'image'
            p = self.file_list[ind]
            image = self.get_image(p)
            # 'mask'
            anno_dict = getattr(self, 'anno_dict')
            get_mask = getattr(self, 'get_mask')
            pm = anno_dict[ind]
            mask = get_mask(pm)
            # Make sure to apply the same augmentation both to image and mask
            if augmenters:
                segmap = SegmentationMapsOnImage(mask, mask.shape)
                image, mask = augmenters(image=image, segmentation_maps=segmap)
                mask = mask.get_arr()
            if self.tensor:
                image = self.totensor_image(image)
                mask = self.totensor_mask(mask)
            return {'image': image, 'mask': mask}
        else:
            raise NotImplementedError("Choose one ['image', 'mask', 'both']")

    def info(self):
        print(self.overview_table.loc[self.acronym])

    def build_totensor(self, val: float) -> Callable:
        """Base transformation to cast an image array to a tensor

        Note that mask output won't be boolean. In case of instance GT, they
        will have floating values between (0.0, 1.0].
        """
        return transforms.Compose([
                # transforms.Lambda(lambda x: np.array(x)),
                # ToTensor works in range[0,255]
                # transforms.Lambda(lambda x: (255*(x/x.max())).astype(np.uint8)),
                # transforms.Lambda(lambda x: (255*(x/x.max())).astype(np.uint8)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(val))])

    def _resize_arr(
        self,
        arr: Union[np.ndarray, Image.Image],
        interpolation: int
    ) -> Union[np.ndarray, Image.Image]:
        """Resize image array. Bilinear interpolation for 'image', no
        interpolation for 'mask'.

        Parameters
        ----------
        arr : numpy.ndarray or PIL.Image.Image
            Image array or Pillow image object
        interpolation : {0,1,2}
            Interpolation argument for ``cv2.resize``.
            0: Nearest-neighbor
            1: Bi-linear
            2: Bi-cubic
            https://docs.opencv.org/4.5.2/d1/d4f/imgproc_2include_2opencv2_2imgproc_8hpp.html
            https://docs.opencv.org/4.5.2/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121

        """
        if isinstance(arr, Image.Image):
            # PIL automatically resample with nearest if arr is grayscale
            arr = np.array(arr)
        return cv2.resize(arr, self.resize, interpolation=interpolation)

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

    @property
    def root_dir(self) -> Path:
        """Define `self._root_dir` within `__init__()` method in subcls."""
        _root_dir = getattr(self, '_root_dir')
        if not isinstance(_root_dir, Path):
            _root_dir = Path(_root_dir)
        assert _root_dir.exists(), f"`root_dir='{_root_dir}'` does not exist."
        return _root_dir

    @property
    def output(self) -> str:
        """If not defined, it is set to 'image'"""
        return self._output if hasattr(self, '_output') else 'image'

    @output.setter
    def output(self, val):
        self._output = val

    @property
    def resize(self):
        return self._resize

    @property
    def tensor(self) -> bool:
        """Apply base_transform in order to return Tensor."""
        if hasattr(self, '_tensor'):
            return getattr(self, '_tensor')
        return False

    @cached_property
    def overview_table(self) -> pd.DataFrame:
        """Read and store the overview table"""
        table = pd.read_table('images/table_overview.txt',
                              sep=r'\s+\|\s',
                              engine='python',
                              index_col=0)
        return table

    @property
    def resolution(self) -> Tuple[int, ...]:
        res = self.overview_table.loc[self.acronym, 'Resolution']
        return tuple(map(int, res.strip('()').split(',')))

    @property
    def indices(self) -> Optional[List[int]]:
        if hasattr(self, '_indices'):
            _indices = getattr(self, '_indices')
            if _indices is not None:
                return sorted(_indices)
        return None
