from typing import TYPE_CHECKING, IO

if TYPE_CHECKING:
    import zipfile
    from pathlib import Path

from functools import cached_property
from typing import Dict, List, Optional, Sequence, Tuple, Union

import albumentations
import numpy as np

from ..base import MaskDataset
from ..utils import stack_channels_to_rgb


class TissueNetV1(MaskDataset):
    """TissueNet v1.0 [1]_

    The TissueNet data is composed of a train, val, and test split. The train
    split is composed of images which are 512x512 pixels. During training, we
    select random crops of size 256x256. The val and test splits are each
    composed of size 256x256 pixel images so that they can be passed directly to
    the model without cropping to evaluate calbacks and test accuracy. The val
    dataset is composed of three parts: the data at the original image
    resolution, 0.5X resolution, and 2X resolution. This is because we don't
    perform scaling (or any type of data augmentation) when evaluating val
    accuracy, but we want to ensure that we evaluate each epoch's loss against a
    range of image sizes

    License: Modified Apache License Version 2.0. Read the license for more.

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
    selected_subset : {'train', 'val', 'test'}, default: 'train'
        Select a subset
    image_ch : {'cells', 'nuclei'}, default: ('cells', 'nuclei')
        Which channel(s) to load as image. Make sure to give it as a Sequence
        when choose a single channel.
    anno_ch : {'cells', 'nuclei'}, default: ('cells', 'nuclei')
        Which channel(s) to load as annotation. Make sure to give it as a
        Sequence when choose a single channel.
    uint8 : bool, default: True
        Whether to convert images to UINT8. It will divide images by a certain
        value so that they have a reasonable range of pixel values when cast
        into UINT8. If set False, no process will be applied. Read more about
        rationales in Notes section.
    selected_tissue : str, default: 'all'
        Print `self.valid_tissues` for valid list
    selected_platform : str, default: 'all'
        Print `self.valid_platforms` for valid list

    Notes
    -----
    - TissueNet v1.0 was the dataset for [1]_ paper and released in 2021
    - stored in .npz format
    - train, val, test are all big .npy raw file
    - .npy file comes with a header whose size is 128 bytes and ends with
      newline char
    - there is no `file_list`
    - image has channel order [nuclei, cells] but mask has [cells, nuclei]

    References
    ----------
    .. [1] N. F. Greenwald et al., “Whole-cell segmentation of tissue images
    with human-level performance using large-scale data annotation and deep
    learning,” Nat Biotechnol, vol. 40, no. 4, Art. no. 4, Apr. 2022, doi:
    10.1038/s41587-021-01094-0.

    See Also
    --------
    MaskDataset : Super class
    Dataset : Base class
    DatasetInterface : Interface

    """
    # Dataset's acronym
    acronym = 'TissueNetV1'

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
        selected_subset: str = 'train',
        image_ch: Sequence[str] = ('cells', 'nuclei'),
        anno_ch: Sequence[str] = ('cells', 'nuclei'),
        selected_tissue: str = 'all',
        selected_platform: str = 'all',
        uint8: bool = True,
        **kwargs
    ):
        self._root_dir = root_dir
        self._output = output
        self._transforms = transforms
        self._num_samples = num_samples
        self._grayscale = grayscale
        self._grayscale_mode = grayscale_mode
        # specific to this dataset
        self.selected_subset = selected_subset
        self.image_ch = image_ch
        self.anno_ch = anno_ch
        self.selected_tissue = selected_tissue
        self.selected_platform = selected_platform
        self.uint8 = uint8
        if selected_subset not in ('train', 'val', 'test'):
            raise ValueError("Set `selected_subset` to one of ('train', 'val', 'test')")
        if not any([ch in ('cells', 'nuclei') for ch in image_ch]):
            raise ValueError("Set `image_ch` in ('cells', 'nuclei') in sequence")
        if not any([ch in ('cells', 'nuclei') for ch in anno_ch]):
            raise ValueError("Set `anno_ch` in ('cells', 'nuclei') in sequence")

        self._npz = np.load(self.search_npz())
        self._npz_zip: 'zipfile.ZipFile' = self._npz.zip
        self.tissue_list = self._npz['tissue_list']
        self.platform_list = self._npz['platform_list']
        self._f_X: IO[bytes] = self._npz_zip.open('X.npy')
        self._f_y: IO[bytes] = self._npz_zip.open('y.npy')

        self._validate_selected()

    def search_npz(self) -> 'Path':
        """Search a subset .npz file based on params

        Parameters
        ----------
        self.root_dir
        self.selected_subset
        """
        target = f'tissuenet_v1.0_{self.selected_subset}.npz'
        try:
            p = next(filter(lambda s: s.name == target, self.root_dir.iterdir()))
        except StopIteration:
            raise FileNotFoundError(f"File '{self.root_dir / target}' not found")
        return p

    def __exit__(self):
        # make sure to close files
        self._f_X.close()
        self._f_y.close()

    @cached_property
    def valid_tissues(self):
        """Store valid tissue types"""
        return np.unique(self.tissue_list)

    @cached_property
    def valid_platforms(self):
        """Store valid tissue types"""
        return np.unique(self.platform_list)

    def _validate_selected(self):
        """validate selected_tissue and selected_platform"""
        if self.selected_tissue not in self.valid_tissues and self.selected_tissue != 'all':
            raise ValueError('Selected tissue must be either be part of the valid_tissues list, or all')
        if self.selected_platform not in self.valid_platforms and self.selected_platform != 'all':
            raise ValueError('Selected platform must be either be part of the valid_platforms list, or all')

    @property
    def image_shape(self):
        return (512, 512, 2)

    @property
    def image_dtype(self):
        return 'float32'

    @property
    def mask_shape(self):
        return (512, 512, 2)

    @property
    def mask_dtype(self):
        return 'int32'

    @cached_property
    def _image_header_offset(self):
        return self._seek_header(self._f_X)

    @cached_property
    def _mask_header_offset(self):
        return self._seek_header(self._f_y)

    def get_image(self, p: int) -> np.ndarray:
        img = self._read_chunk(
            self._f_X,
            chunk=self.image_shape,
            dtype=self.image_dtype,
            ind=p,
            header_offset=self._image_header_offset
        )
        if len(image_ch := self.image_ch) == 1:
            ch = image_ch[0]
            if ch == 'nuclei':
                return img[..., 0]
            elif ch == 'cells':
                return img[..., 1]
        img_rgb = stack_channels_to_rgb([img[..., i] for i in range(2)], 1, 2)
        return (255 * img_rgb).astype(np.uint8) if self.uint8 else img_rgb

    def get_mask(self, p: str) -> np.ndarray:
        mask = self._read_chunk(
            self._f_y,
            chunk=self.mask_shape,
            dtype=self.mask_dtype,
            ind=int(p),
            header_offset=self._mask_header_offset
        )
        if len(anno_ch := self.anno_ch) == 1:
            ch = anno_ch[0]
            if ch == 'cells':
                return mask[..., 0]
            elif ch == 'nuclei':
                return mask[..., 1]
        return mask

    @cached_property
    def file_list(self) -> List[int]:
        """Dummy file list
        """
        if self.selected_tissue == 'all':
            tissue_idx = np.repeat(True, len(self.tissue_list))
        else:
            tissue_idx = self.tissue_list == self.selected_tissue

        if self.selected_platform == 'all':
            platform_idx = np.repeat(True, len(self.platform_list))
        else:
            platform_idx = self.platform_list == self.selected_platform

        combined_idx = tissue_idx * platform_idx
        idx = np.where(combined_idx)[0]
        return idx.tolist()

    @cached_property
    def anno_dict(self) -> Dict[int, str]:
        """Dummy annotation dictionary
        """
        return dict((i, str(i)) for i in self.file_list)

    @classmethod
    def _seek_header(cls, f: IO[bytes]):
        if f.tell() != 0:
            f.seek(0)
        f.readline()
        return f.tell()

    @classmethod
    def _read_chunk(
        cls,
        f: IO[bytes],
        chunk: Tuple[int],
        dtype: str,
        ind: int,
        header_offset: int = 0,
    ) -> np.ndarray:
        chunk_nbytes = np.dtype(dtype).itemsize * np.prod(chunk)
        f.seek(chunk_nbytes * ind + header_offset)
        buf = f.read(chunk_nbytes)
        arr = np.frombuffer(buf, dtype=dtype)
        return arr.reshape(chunk)
