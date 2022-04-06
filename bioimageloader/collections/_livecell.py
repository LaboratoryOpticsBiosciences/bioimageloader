from functools import cached_property
from pathlib import Path
from typing import List, Optional

import albumentations
import cv2
import numpy as np
import tifffile
from pycocotools import coco

from ..base import MaskDataset

class LIVECell(MaskDataset):
    """LIVECEll
    A large-scale dataset for label-free live cell segmentation

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
    save_tif : bool, default: True
        Save COCO annotations as tif mask images in a new ./root_dir/masks directory. 

    References
    ----------
    .. [1] https://sartorius-research.github.io/LIVECell/
    .. [2] https://www.nature.com/articles/s41592-021-01249-6

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
        save_tif: bool = True,
        **kwargs
    ):
        self._root_dir = root_dir
        self._output = output
        self._transforms = transforms
        self._num_samples = num_samples
        # specific to this one here
        self.training = training
        self.save_tif = save_tif
     
        # Read training/val or test annotations from json
        # Save the masks as tif files if save_tif = True
        if not os.path.exists(root_dir + "/masks"):
            os.makedirs(root_dir + "/masks")
        if not os.path.exists(root_dir + "/masks/livecell_train_val_masks"):
            os.makedirs(root_dir + "/masks/livecell_train_val_masks")
        if not os.path.exists(root_dir + "/masks/livecell_test_masks"):
            os.makedirs(root_dir + "/masks/livecell_test_masks")
            
        if self.training and self.save_tif:
            print("making instances masks and saving as tif files")
            for img in tqdm(self.anno_dictionary):
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
                tifffile.imsave(root_dir + "/masks/livecell_train_val_masks/" + img["file_name"], mask)
            print("Done!")

        if not self.training and self.save_tif:
            print("making instances masks and saving as tif files")
            self.coco_te = coco.COCO(root_dir + "/livecell_coco_test.json")
            img_te = self.coco_te.loadImgs(self.coco_te.getImgIds())
            self.anno_dictionary = img_te
            for img in tqdm(self.anno_dictionary): 
                annIds = self.coco_te.getAnnIds(imgIds=img["id"], iscrowd=None)
                anns = self.coco_te.loadAnns(annIds)
                mask = self.coco_te.annToMask(anns[0])
                mask = mask.astype(np.int32)
                for i in range(len(anns)):
                    mask |= self.coco_te.annToMask(anns[i]) * i
                tifffile.imsave(root_dir + "/masks/livecell_test_masks/" + img["file_name"], mask)
            print("Done!")
            
    def get_image(self, p: Path) -> np.ndarray:
        img = tifffile.imread(p)
        return img

    def get_mask(self, p: Path) -> np.ndarray:
        mask = tifffile.imread(p)
        return mask 

    @cached_property
    def file_list(self) -> List[Path]:
        # Call MaskDataset.root_dir
        root_dir = self.root_dir
        parent = 'images/livecell_train_val_images' if self.training else 'images/livecell_test_images'
        return sorted(root_dir.glob(f'{parent}/*.tif'))

    @cached_property
    def anno_dict(self) -> List[Path]:
        root_dir = self.root_dir
        parent = 'images/livecell_train_val_images' if self.training else 'images/livecell_test_images'
        return sorted(root_dir.glob(f'{parent}/*.tif'))
