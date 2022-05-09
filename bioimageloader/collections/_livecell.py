import warnings
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional

import albumentations
import cv2
import numpy as np
import tifffile
from pycocotools import coco

from ..base import MaskDataset


class LIVECell(MaskDataset):
    """LIVECEll: A large-scale dataset for label-free live cell segmentation
    [1]_

    “LIVECell - A large-scale dataset for label-free live cell segmentation” by
    Edlund et. al. 2021 [2]_

    Light microscopy is a cheap, accessible, non-invasive modality that when
    combined with well-established protocols of two-dimensional cell culture
    facilitates high-throughput quantitative imaging to study biological
    phenomena. Accurate segmentation of individual cells enables exploration of
    complex biological questions, but this requires sophisticated imaging
    processing pipelines due to the low contrast and high object density. Deep
    learning-based methods are considered state-of-the-art for most computer
    vision problems but require vast amounts of annotated data, for which there
    is no suitable resource available in the field of label-free cellular
    imaging. To address this gap we present LIVECell, a high-quality, manually
    annotated and expert-validated dataset that is the largest of its kind to
    date, consisting of over 1.6 million cells from a diverse set of cell
    morphologies and culture densities. To further demonstrate its utility, we
    provide convolutional neural network-based models trained and evaluated on
    LIVECell.

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
    training : bool, default: True
        Load training set if True, else load testing one
    mask_tif : bool, default: False
        Use saved COCO annotations as tif mask images in a new ./root_dir/masks
        directory. It will greatly improve loading speed. Available after
        calling ``save_coco_to_tif()``.

    Notes
    -----
    - Annotation in MS COCO format [3]_. Parsing it takes time`.
    - Currently not supporting dynamically parsing COCO annotation due to slow
      speed. Pre-parse masks in .tif format by calling ``save_coco_to_tif()``.
    - Validation set is originally separted from training set. Currently they
      are combined ``training=True``.
    - Single cells subsets are not covered

    References
    ----------
    .. [1] https://sartorius-research.github.io/LIVECell/
    .. [2] https://www.nature.com/articles/s41592-021-01249-6
    .. [3] https://cocodataset.org/

    See Also
    --------
    MaskDataset : Super class
    Dataset : Base class
    DatasetInterface : Interface

    """
    # Set acronym
    acronym = 'LIVECell'

    def __init__(
        self,
        root_dir: str,
        *,
        output: str = 'both',
        transforms: Optional[albumentations.Compose] = None,
        num_samples: Optional[int] = None,
        # specific to this dataset
        training: bool = True,
        mask_tif: bool = False,
        **kwargs
    ):
        self._root_dir = root_dir
        self._output = output
        self._transforms = transforms
        self._num_samples = num_samples
        # specific to this one here
        self.training = training
        self.mask_tif = mask_tif

        if not self.mask_tif:
            msg = ("LIVECell dataset does not currently support dynamically "
                   "parsing annotation in MS COCO format due to performance "
                   "issue. Please create masks in .tif format by calling "
                   "`save_coco_to_tif()` and then set `mask_tif=True`")
            warnings.warn(msg, stacklevel=2)

        if self.mask_tif:
            # check
            mask_train_dir = self.root_dir / 'masks' / 'livecell_train_val_masks'
            mask_test_dir = self.root_dir / 'masks' / 'livecell_test_masks'
            if not mask_train_dir.exists() or not mask_test_dir.exists():
                raise Exception("No masks in .tif format.")
            if not any(mask_train_dir.iterdir()) or not any(mask_test_dir.iterdir()):
                raise Exception("No masks in .tif format.")

    def get_image(self, p: Path) -> np.ndarray:
        img = tifffile.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def get_mask(self, p: Path) -> np.ndarray:
        mask = tifffile.imread(p)
        return mask

    @cached_property
    def file_list(self) -> List[Path]:
        # Call MaskDataset.root_dir
        root_dir = self.root_dir
        parent = (root_dir / 'images' / 'livecell_train_val_images' if self.training
                  else root_dir / 'images' / 'livecell_test_masks')
        return sorted(parent.glob('*.tif'))

    @cached_property
    def anno_dict(self) -> Dict[int, Path]:
        root_dir = self.root_dir
        # parent = 'masks/livecell_train_val_masks' if self.training else 'masks/livecell_test_masks'
        parent = (root_dir / 'masks' / 'livecell_train_val_masks' if self.training
                  else root_dir / 'masks' / 'livecell_test_masks')
        return dict((k, v) for k, v in enumerate(
            sorted(parent.glob('*.tif'))
        ))

    def save_coco_to_tif(self):
        """Save the masks as tif files

        Read training/val or test annotations from json. Make tif files under
        'masks/livecell_train_val_masks' and 'masks/livecell_test_masks'.

        Initialize a new instance with setting ``mask_tif=True`` to load saved
        masks.
        """
        print("making instances masks and saving as tif files")
        # makedirs
        mask_train_dir = self.root_dir / 'masks' / 'livecell_train_val_masks'
        mask_test_dir = self.root_dir / 'masks' / 'livecell_test_masks'
        mask_train_dir.mkdir(parents=True, exist_ok=True)
        mask_test_dir.mkdir(parents=True, exist_ok=True)
        # training
        self.coco_tr = coco.COCO(self.root_dir  / 'livecell_coco_train.json')
        img_tr = self.coco_tr.loadImgs(self.coco_tr.getImgIds())
        self.coco_val = coco.COCO(self.root_dir / 'livecell_coco_val.json')
        img_val = self.coco_val.loadImgs(self.coco_val.getImgIds())
        self.anno_dictionary = img_val + img_tr
        # loop
        print(f'training total: {len(self.anno_dictionary)}')
        for ind, img in enumerate(self.anno_dictionary, 1):
            try:
                annIds = self.coco_tr.getAnnIds(imgIds=img["id"], iscrowd=None)
                anns = self.coco_tr.loadAnns(annIds)
                mask = self.coco_tr.annToMask(anns[0])
                mask = mask.astype(np.int32)
                for i in range(len(anns)):
                    mask |= self.coco_tr.annToMask(anns[i]) * i
            except:
                annIds = self.coco_val.getAnnIds(imgIds=img["id"], iscrowd=None)
                anns = self.coco_val.loadAnns(annIds)
                mask = self.coco_val.annToMask(anns[0])
                mask = mask.astype(np.int32)
                for i in range(len(anns)):
                    mask |= self.coco_val.annToMask(anns[i]) * i
            tifffile.imsave(mask_train_dir / img['file_name'], mask)
            print(ind, end=' ')
        print("Done!")
        # test
        print("making instances masks and saving as tif files")
        self.coco_te = coco.COCO(self.root_dir / 'livecell_coco_test.json')
        img_te = self.coco_te.loadImgs(self.coco_te.getImgIds())
        self.anno_dictionary = img_te
        print(f'test total: {len(self.anno_dictionary)}')
        for ind, img in enumerate(self.anno_dictionary, 1):
            annIds = self.coco_te.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = self.coco_te.loadAnns(annIds)
            mask = self.coco_te.annToMask(anns[0])
            mask = mask.astype(np.int32)
            for i in range(len(anns)):
                mask |= self.coco_te.annToMask(anns[i]) * i
            tifffile.imsave(mask_test_dir / img['file_name'], mask)
            print(ind, end=' ')
        print("Done!")
