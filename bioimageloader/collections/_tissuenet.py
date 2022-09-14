from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    import zipfile
    import pandas as pd

from pathlib import Path
from functools import cached_property
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple, Union
import warnings

import albumentations
import numpy as np
import tifffile

from ..base import MaskDataset
from ..utils import stack_channels_to_rgb


class TissueNetV1(MaskDataset):
    """TissueNet v1.0 [1]_. Download data from deepcell.org [2]_.

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

    License: Modified Apache License Version 2.0. Read the included license for
    more.

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
    use_unzipped : bool, default: False
        Unzip .npz file for small memory consumption and fast random access.
        Read ``unzip()`` for more.
    in_memory: bool, default: False
        Load the whole data in memory. Memory footprint will be about 10GB+. If
        your memory pool is large enough, this method will guarantee the fastest
        loading speed. This argument is only valid when ``use_unzipped`` is set
        to False.

    Notes
    -----
    - TissueNet v1.0 was the dataset for [1]_ paper and released on July 2021
    - Data is stored in .npz format
    - Train, val, test are all big .npy raw file
    - Highly recommend using ``unzip()`` to unzip .npy files for fast random
      accessing with small memory footprint. Then set ``use_unzipped`` argument
      to True to load them.
    - If you have a large memory available (10GB+), you could use ``in_memory``
      option.
    - when ``use_unzipped`` is set to False, ``file_list`` and ``anno_dict`` are
      dummy lists.
    - .npy file comes with a header whose size is 128 bytes and ends with
      newline char
    - image has channel order [nuclei, cells] but mask has [cells, nuclei]

    References
    ----------
    .. [1] N. F. Greenwald et al., “Whole-cell segmentation of tissue images
    with human-level performance using large-scale data annotation and deep
    learning,” Nat Biotechnol, vol. 40, no. 4, Art. no. 4, Apr. 2022, doi:
    10.1038/s41587-021-01094-0.
    .. [2] https://www.deepcell.org/

    See Also
    --------
    MaskDataset : Super class
    Dataset : Base class
    DatasetInterface : Interface

    """
    # Dataset's acronym
    acronym = 'TissueNetV1'
    # allowed argument sets
    valid_subset = ('train', 'val', 'test')
    valid_image_ch = ('cells', 'nuclei')
    valid_anno_ch = ('cells', 'nuclei')

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
        use_unzipped: bool = False,
        in_memory: bool = False,
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
        self.use_unzipped = use_unzipped
        self.in_memory = in_memory
        # check some arguemnts
        if use_unzipped:
            assert not in_memory, "`in_memory` and `use_unzipped` cannot be set True at the same time"
        if in_memory:
            assert not use_unzipped, "`in_memory` and `use_unzipped` cannot be set True at the same time"
        if selected_subset not in self.valid_subset:
            raise ValueError(f"Set `selected_subset` to one of {self.valid_subset}")
        if not any([ch in self.valid_image_ch for ch in image_ch]):
            raise ValueError(f"Set `image_ch` in {self.valid_image_ch} in sequence")
        if not any([ch in self.valid_anno_ch for ch in anno_ch]):
            raise ValueError(f"Set `anno_ch` in {self.valid_anno_ch} in sequence")

        if use_unzipped:
            # load unzipped files
            if not self.is_unzipped:
                raise FileNotFoundError(
                    f"'{self.root_unzip}' is not found. Call `self.unzip()` "
                    "before setting `use_unzipped` argument",
                )
            self.tissue_list = np.load(self.root_unzip / 'tissue_list.npy')
            self.platform_list = np.load(self.root_unzip / 'platform_list.npy')
        else:
            # load raw files
            self._npz = np.load(self.search_npz())
            self.tissue_list = self._npz['tissue_list']
            self.platform_list = self._npz['platform_list']
            if in_memory:
                # load the whole raw files in memory
                warnings.warn(
                    "Loading big raw .npy files in memory. It roughly requires "
                    "10GB+ of memory and takes some time. If you'd like to "
                    "reduce the memory footprint, consider using "
                    "`use_unzipped` argument."
                )
                self.X = self._npz['X']
                self.y = self._npz['y']
            else:
                # load each chunk at a time. random accessing becomes an issue
                warnings.warn(
                    f"Loading {self.__class__.__name__} can be slow because it "
                    "will load compressed buffer directly. Consider using "
                    "`self.unzip()` to unzip dataset and set "
                    "`use_unzipped=True`."
                )
                self._npz_zip: 'zipfile.ZipFile' = self._npz.zip
                self._f_X: IO[bytes] = self._npz_zip.open('X.npy')
                self._f_y: IO[bytes] = self._npz_zip.open('y.npy')

        self._validate_selected()

    def search_npz(self) -> Path:
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
        if hasattr(self, '_f_X'):
            self._f_X.close()
        if hasattr(self, '_f_y'):
            self._f_y.close()

    @cached_property
    def root_unzip(self) -> Path:
        p = self.search_npz()
        # drop .npy extension
        return p.parent / p.stem

    @cached_property
    def is_unzipped(self):
        """Check if unzipped files exist or if ``self.unzip`` was called before.
        See ``self.unzip`` for more"""
        if self.root_unzip.is_dir() and self.root_unzip.exists():
            return True
        return False

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

    @cached_property
    def image_shape(self):
        if self.selected_subset == 'train':
            return (512, 512, 2)
        return (256, 256, 2)

    @cached_property
    def image_dtype(self):
        return 'float32'

    @cached_property
    def mask_shape(self):
        if self.selected_subset == 'train':
            return (512, 512, 2)
        return (256, 256, 2)

    @cached_property
    def mask_dtype(self):
        return 'int32'

    @cached_property
    def _image_header_offset(self):
        return self._seek_header(self._f_X)

    @cached_property
    def _mask_header_offset(self):
        return self._seek_header(self._f_y)

    def get_image(self, p: Union[Path, int]) -> np.ndarray:
        if isinstance(p, Path):
            # is_unzipped == True
            img = tifffile.imread(p)
        else:
            if self.in_memory:
                img = self.X[p]
            else:
                # otherwise, seek for starting byte and load a chunk
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

    def get_mask(self, p: Union[Path, str]) -> np.ndarray:
        if isinstance(p, Path):
            # is_unzipped == True
            mask = tifffile.imread(p)
        else:
            if self.in_memory:
                mask = self.y[int(p)]
            else:
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
    def file_list(self) -> Union[List[Path], List[int]]:
        # tissue
        if self.selected_tissue == 'all':
            tissue_idx = np.repeat(True, len(self.tissue_list))
        else:
            tissue_idx = self.tissue_list == self.selected_tissue
        # platform
        if self.selected_platform == 'all':
            platform_idx = np.repeat(True, len(self.platform_list))
        else:
            platform_idx = self.platform_list == self.selected_platform
        combined_idx = tissue_idx * platform_idx
        idx = np.where(combined_idx)[0]
        idx = idx.tolist()

        if self.in_memory:
            return idx
        if self.is_unzipped:
            subdir = self.root_unzip / 'X'
            return [sorted(subdir.iterdir())[i] for i in idx]
        # dummy file list
        return idx

    @cached_property
    def anno_dict(self) -> Union[Dict[int, Path], Dict[int, str]]:
        if self.in_memory:
            return dict((i, str(i)) for i in self.file_list)
        if self.is_unzipped:
            subdir = self.root_unzip / 'y'
            return dict((i, subdir / p.name) for i, p in enumerate(self.file_list))
        # Dummy annotation dictionary
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

    @classmethod
    def generate_summary(
        cls,
        root_dir: str,
    ) -> 'pd.DataFrame':
        import pandas as pd

        """Generate summary of numbers of data points depending on platforms,
        tissues, and subsets ('train', 'val', 'test')

        Parameters
        ----------
        root_dir : str
            Path to root directory
        """
        tissue = cls(
            root_dir,
            selected_tissue='all',
            selected_platform='all',
        )
        # define multi-indexed DataFrame
        valid_platforms = tissue.valid_platforms
        valid_tissues = tissue.valid_tissues
        multi_index = pd.MultiIndex.from_product(
            [valid_platforms, valid_tissues],
            names=['platform', 'tissue']
        )
        df_tissuenet = pd.DataFrame(index=multi_index,
                                    columns=cls.valid_subset)
        df_tissuenet = df_tissuenet.fillna(0)
        # Fill the table
        for selected_subset in cls.valid_subset:
            total = 0
            for plat, tiss in product(valid_platforms, valid_tissues):
                _tissue = cls(
                    root_dir,
                    selected_subset=selected_subset,
                    selected_platform=plat,
                    selected_tissue=tiss,
                )
                total += len(_tissue)
                # print(f'{plat:10s} {tiss:10s}: {len(_tissue)}')
                df_tissuenet.loc[plat, tiss][selected_subset] = len(_tissue)
            # print(f'total ({selected_subset:5s}): {total}')
        # Add Total at both ends of index and column
        df_tissuenet.loc['Total', :] = df_tissuenet.sum().values
        df_tissuenet['Total'] = df_tissuenet.sum(axis=1)
        df_tissuenet = df_tissuenet.convert_dtypes()
        return df_tissuenet

    def unzip(self, compression='zlib'):
        """Unzip .npz file to allow faster loading

        This method will create a directory within the `root_dir` and extract
        all files inside .npz file. Metadata will be extracted as they are, but
        images and masks are not. They are two big raw .npy files and will be
        split to individual files. Images will be extracted to a directory "X"
        and masks to a directory "y" with 6-zero-padded index in tiff format.
        The unzipped data can be loaded by setting `use_unzipped=True` when
        initializing a new instance. Note that it will require a few GBs for
        each "train", "val", and "test" set.

        The original format .npz is a zipped numpy format. It contains multiple
        .npy files and they are basically raw files. Accessing them sequentially
        is not an issue and can be done really fast, but the issue is random
        accessing. Jumping inside a big raw file is very slow. The slower the
        loading step becomes, the further away the next buffer is from the
        current one. Imagine you do random shuffling and loading process will
        become a huge bottleneck.

        Parameters
        ----------
        compression : bool, default: 'zlib'
            compression argument to ``tifffile.imwrite()``. Default 'zlib' is
            equivalent to deflate algorithm, which the original raw data is
            compressed with.

        """
        assert self.selected_tissue == 'all' and self.selected_platform == 'all'

        def _filter_metafiles(p: 'zipfile.ZipInfo'):
            if p.filename in ['X.npy', 'y.npy']:
                return False
            return True

        if self.is_unzipped:
            warnings.warn(
                f"{self._npz_zip.filename} is possibly already unzipped",
                stacklevel=2
            )
            return

        # if not unzipped
        npz = self._npz_zip
        # make outputdir
        outdir = self.root_dir / Path(npz.filename).stem
        outdir.mkdir()
        # extract metadata
        metafiles = list(filter(_filter_metafiles, npz.filelist))
        for f in metafiles:
            npz.extract(f, outdir)
        # extract X and y
        outdir_X = outdir / 'X'
        outdir_y = outdir / 'y'
        outdir_X.mkdir()
        outdir_y.mkdir()
        print(f'Save {len(self)} images and masks\nUnzipping...')
        for i in range(len(self)):
            img = self._read_chunk(
                self._f_X,
                chunk=self.image_shape,
                dtype=self.image_dtype,
                ind=i,
                header_offset=self._image_header_offset
            )
            mask = self._read_chunk(
                self._f_y,
                chunk=self.mask_shape,
                dtype=self.mask_dtype,
                ind=i,
                header_offset=self._mask_header_offset
            )
            # write X
            tifffile.imwrite(
                outdir_X / f'{i:06d}.tif',
                img,
                compression=compression
            )
            # write y
            tifffile.imwrite(
                outdir_y / f'{i:06d}.tif',
                mask,
                compression=compression
            )
            print(i, end=' ')
        print(f"\nMake another {self.__class__.__name__} instance with setting "
              "`use_unzipped=True` to load unzipped files.")
