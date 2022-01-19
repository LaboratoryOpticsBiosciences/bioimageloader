import os.path
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional

import albumentations
import numpy as np
from gimpformats.gimpXcfDocument import GimpDocument
from PIL import Image

from ..base import NucleiDataset


class MurphyLab(NucleiDataset):
    """Nuclei Segmentation In Microscope Cell Images: A Hand-Segmented Dataset
    And Comparison Of Algorithms

    Two annotation formats; Photoshop and GIMP, 97 segmented images (out of 100)

    Notes
    -----
    - 3 missing segmentations: ind={31, 43, 75}
        ./data/images/dna-images/gnf/dna-31.png
        ./data/images/dna-images/gnf/dna-43.png
        ./data/images/dna-images/ic100/dna-25.png
    - Manually filled annotation to make masks using GIMP
    - 2009_ISBI_2DNuclei_code_data/data/images/segmented-lpc/ic100/dna-15.xcf
      does not have 'borders' layer like the others.  This one alone has
      'border' layer.

    .. [1] L. P. Coelho, A. Shariff, and R. F. Murphy, “Nuclear segmentation in
       microscope cell images: A hand-segmented dataset and comparison of
       algorithms,” in 2009 IEEE International Symposium on Biomedical Imaging:
       From Nano to Macro, Jun. 2009, pp. 518–521, doi:
       10.1109/ISBI.2009.5193098.
    """

    # Dataset's acronym
    acronym = 'MurphyLab'

    def __init__(
        self,
        # Interface requirement
        root_dir: str,
        output: str = 'both',
        transforms: Optional[albumentations.Compose] = None,
        num_calls: Optional[int] = None,
        # Specific to this dataset
        drop_missing_pairs: bool = True,
        **kwargs
    ):
        """
        Parameters
        ---------
        root_dir : str or pathlib.Path
            Path to root directory
        output : {'image','mask','both'} (default: 'both')
            Change outputs. 'both' returns {'image': image, 'mask': mask}.
        transforms : albumentations.Compose, optional
            An instance of Compose (albumentations pkg) that defines
            augmentation in sequence.
        num_calls : int, optional
            Useful when `transforms` is set. Define the total length of the
            dataset. If it is set, it overrides __len__.
        drop_missing_pairs : bool (default: True)
            Valid only if `output='both'`. It will drop images that do not have
            mask pairs.

        See Also
        --------
        NucleiDataset : Super class
        DatasetInterface : Interface

        """
        # Interface and super-class arguments
        self._root_dir = os.path.join(root_dir, 'data', 'images')
        self._output = output
        self._transforms = transforms
        self._num_calls = num_calls

        self.drop_missing_pairs = drop_missing_pairs
        if self.output == 'both' and self.drop_missing_pairs:
            self.file_list, self.anno_dict = self._drop_missing_pairs()

    def get_image(self, p: Path) -> np.ndarray:
        img = Image.open(p)
        img = img.convert('L')
        return np.asarray(img)

    def get_mask(self, p: Path) -> np.ndarray:
        doc = GimpDocument(p.as_posix())
        layers = [layer.name for layer in doc.layers]
        layer_borders = doc.getLayer(layers.index('borders'))
        mask = layer_borders.image
        return np.asarray(mask)

    @cached_property
    def file_list(self) -> List[Path]:
        # if hasattr(self, '_file_list'):
        #     # Sort of setter by _drop_missing_pairs
        #     return getattr(self, '_file_list')
        root_dir = self.root_dir
        parent = 'dna-images'
        # dna-images
        file_list = sorted(
            root_dir.glob(f'{parent}/*/*.png'), key=self._sort_key
        )
        return file_list

    @classmethod
    def _sort_key(cls, p, zfill=2):
        split = p.stem.split('-')
        return '-'.join([p.parent.stem] + [s.zfill(zfill) for s in split])

    @cached_property
    def anno_dict(self) -> Dict[int, Path]:
        # if hasattr(self, '_anno_dict'):
        #     print('Nope?')
        #     # Sort of setter by _drop_missing_pairs
        #     return getattr(self, '_anno_dict')
        anno_dict = {}
        for i, p in enumerate(self.file_list):
            p_anno = '/'.join([p.parent.stem, p.stem + '.xcf'])
            # Ignore 'segmented-ashariff`. It seems that Ashariff got bored
            # after 10 images.
            anno = p.parents[2] / 'segmented-lpc' / p_anno
            if anno.exists():
                anno_dict[i] = anno
        return anno_dict
