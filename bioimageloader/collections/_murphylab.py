import os.path
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional

import albumentations
import numpy as np
try:
    from gimpformats.gimpXcfDocument import GimpDocument
except ModuleNotFoundError as e:
    print("Install `gimpformats` pkg")
    raise e

from PIL import Image

from ..base import MaskDataset


class MurphyLab(MaskDataset):
    """Nuclei Segmentation In Microscope Cell Images: A Hand-Segmented Dataset
    And Comparison Of Algorithms [1]_

    Parameters
    ----------
    root_dir : str or pathlib.Path
        Path to root directory
    output : {'both', 'image', 'mask'}, default: 'both'
        Change outputs. 'both' returns {'image': image, 'mask': mask}.
    transforms : albumentations.Compose, optional
        An instance of Compose (albumentations pkg) that defines
        augmentation in sequence.
    num_samples : int, optional
        Useful when ``transforms`` is set. Define the total length of the
        dataset. If it is set, it overwrites ``__len__``.
    drop_missing_pairs : bool, default: True
        Valid only if `output='both'`. It will drop images that do not have
        mask pairs.
    drop_broken_files : bool, default: True
        Drop broken files that cannot be read
    filled_mask : bool, default: False
        Use saved filled masks through `fill_save_mask()` method instead of
        default boundary masks. If one would want to use manually modified
        masks, the annotation files should have the same name as '*.xcf'
        with modified suffix by '.png'.

    Warnings
    --------
    This dataset has many issues whose details can be found below. The simpleset
    way is to drop those that cause isseus. It is recommended to not opt out
    ``drop_missing_pairs()`` and ``drop_broken_files()``. Otherwise, it will
    meet exceptions.

    If one wants filled hole, ``fill_save_mask()`` function will fill holes with
    some tricks to handle edge cases and save them as .png format. Then set
    ``filled_mask`` argument to True to load them.

    Read more in Notes section

    Notes
    -----
    - 4 channel PNG format annotation mask even though mask is binary. But it is
      not grayscale binary. They put value 255 only in red channel.
    - Two annotation formats; Photoshop and GIMP. It seems that two annotators
      worked separately. segmented-ashariff will be ignored. In total, 97
      segmented images (out of 100)
    - 3 missing segmentations: ind={31, 43, 75}
        ./data/images/dna-images/gnf/dna-31.png

        ./data/images/dna-images/gnf/dna-43.png

        ./data/images/dna-images/ic100/dna-25.png

    - Manually filled annotation to make masks using GIMP
    - 2009_ISBI_2DNuclei_code_data/data/images/segmented-lpc/ic100/dna-15.xcf
      does not have 'borders' layer like the others.  This one alone has
      'border' layer.

    References
    ----------
    .. [1] L. P. Coelho, A. Shariff, and R. F. Murphy, “Nuclear segmentation in
       microscope cell images: A hand-segmented dataset and comparison of
       algorithms,” in 2009 IEEE International Symposium on Biomedical Imaging:
       From Nano to Macro, Jun. 2009, pp. 518–521, doi:
       10.1109/ISBI.2009.5193098.

    See Also
    --------
    MaskDataset : Super class
    Dataset : Base class
    DatasetInterface : Interface

    """
    # Dataset's acronym
    acronym = 'MurphyLab'

    def __init__(
        self,
        # Interface requirement
        root_dir: str,
        *,
        output: str = 'both',
        transforms: Optional[albumentations.Compose] = None,
        num_samples: Optional[int] = None,
        # Specific to this dataset
        drop_missing_pairs: bool = True,
        drop_broken_files: bool = True,
        filled_mask: bool = False,
        **kwargs
    ):
        # Interface and super-class arguments
        self._root_dir = os.path.join(root_dir, 'data', 'images')
        self._output = output
        self._transforms = transforms
        self._num_samples = num_samples
        # Specific to this dataset
        self.drop_missing_pairs = drop_missing_pairs
        self.drop_broken_files = drop_broken_files
        self.filled_mask = filled_mask

        if self.output == 'both' and self.drop_missing_pairs:
            self.file_list, self.anno_dict = self._drop_missing_pairs()
        if self.output == 'both' and self.drop_broken_files:
            self.file_list, self.anno_dict = self._drop_broken_files()

    def get_image(self, p: Path) -> np.ndarray:
        img = Image.open(p)
        return np.asarray(img)

    def get_mask(self, p: Path) -> np.ndarray:
        if self.filled_mask:
            mask = Image.open(p)
            return np.asarray(mask, dtype=np.float32)
        doc = GimpDocument(p.as_posix())
        layers = [layer.name for layer in doc.layers]
        try:
            ind_layer = layers.index('borders')
        except ValueError:
            # '/data/images/segmented-lpc/ic100/dna-15.xcf' do not have
            # 'borders' but 'border' layer.
            ind_layer = layers.index('border')
        # get image
        layer_borders = doc.getLayer(ind_layer)
        mask = layer_borders.image
        return np.asarray(mask)[..., 0]

    @cached_property
    def file_list(self) -> List[Path]:
        root_dir = self.root_dir
        parent = 'dna-images'
        # dna-images
        file_list = sorted(
            root_dir.glob(f'{parent}/*/*.png'), key=self._sort_key
        )
        return file_list

    @staticmethod
    def _sort_key(p, zfill=2):
        split = p.stem.split('-')
        return '-'.join([p.parent.stem] + [s.zfill(zfill) for s in split])

    @cached_property
    def anno_dict(self) -> Dict[int, Path]:
        ext = '.png' if self.filled_mask else '.xcf'
        anno_dict = {}
        for i, p in enumerate(self.file_list):
            p_anno = '/'.join([p.parent.stem, p.stem + ext])
            # Ignore 'segmented-ashariff`. It seems that Ashariff got bored
            # after 10 images.
            anno = p.parents[2] / 'segmented-lpc' / p_anno
            if anno.exists():
                anno_dict[i] = anno
        return anno_dict

    def _drop_broken_files(self):
        """Drop broken files

        '/data/images/segmented-lpc/ic100/dna-46.xcf' cannnot be read by
        ``gimpformats``
        """
        file_list = self.file_list
        anno_dict = self.anno_dict
        for i, p in anno_dict.items():
            # if p.name == 'dna-46.xcf':
            if '/'.join([p.parent.name, p.name]) == 'ic100/dna-46.xcf':
                file_list.pop(i)
                anno_dict.pop(i)
                break
        anno_dict = dict((i, v) for i, v in enumerate(anno_dict.values()))
        return file_list, anno_dict

    def fill_save_mask(self):
        """Fill holes from boundary mask with some tricks

        Requires scipy and scikit-image. Install depencency with pip option
        ``pip install bioimageloader[process]``.

        Note that this does not result perfect filled masks. Those not entirely
        closed by this algorithm (36, 40, 63).

        Other issues: ``ind=63``: 'border' not 'borders', ``ind=93``
        ``GimpDocument`` cannot read it...
        """
        from scipy.ndimage import binary_fill_holes
        from skimage.morphology import dilation, erosion

        def fill_holes(
            img: np.ndarray,
            w_edge: int,
            w_pad: int
        ) -> np.ndarray:
            # cut edges because many are not closed
            edge_cut = img[w_edge:-w_edge, w_edge:-w_edge]
            # dilate to connect some boundaries
            dilated = dilation(edge_cut)
            # pad with refletion mode to close bounary at edges
            filled_pad = binary_fill_holes(
                np.pad(dilated, w_pad, mode='reflect')
            )
            # back to original shape
            w_rev = w_pad - w_edge
            filled_pad = filled_pad[w_rev:-w_rev, w_rev:-w_rev]
            # erode, because we dilated
            filled_pad = erosion(filled_pad)
            return filled_pad

        for k, p in self.anno_dict.items():
            # read gimp document .xcf
            try:
                doc = GimpDocument(p.as_posix())
            except TypeError:
                # '/data/images/segmented-lpc/ic100/dna-46.xcf' cannnot be read
                # through gimpformats library. You can open it GIMP though.
                print(f"Cannot open '{k}: {p}' with gimpformats lib")
                continue
            # get 'borders' layer
            layers = [layer.name for layer in doc.layers]
            try:
                ind_layer = layers.index('borders')
            except ValueError:
                # '/data/images/segmented-lpc/ic100/dna-15.xcf' do not have
                # 'borders' but 'border' layer.
                print(f"Exception layer['border'] '{k}: {p}'")
                ind_layer = layers.index('border')
            # get image
            layer_borders = doc.getLayer(ind_layer)
            mask = layer_borders.image
            mask = np.asarray(mask)
            # print(k, mask.shape)
            # fill_holes, it has 3 channels and red channel has annotation
            filled = fill_holes(mask[..., 0], w_edge=15, w_pad=200)
            # save
            img = Image.fromarray(filled)
            print(f"Saving '{k}: {p}'")
            img.save(p.with_suffix('.png'))
