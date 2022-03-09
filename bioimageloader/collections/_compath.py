import xml.etree.ElementTree as ET
from functools import cached_property
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import albumentations
import numpy as np
import tifffile
from PIL import Image
from skimage.draw import polygon

from ..base import MaskDataset


class ComputationalPathology(MaskDataset):
    """A Dataset and a Technique for Generalized Nuclear Segmentation for
    Computational Pathology [1]_

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
        Useful when ```transforms``` is set. Define the total length of the
        dataset. If it is set, it overwrites ``__len__``.
    grayscale : bool, default: False
        Convert images to grayscale
    grayscale_mode : {'cv2', 'equal', Sequence[float]}, default: 'cv2'
        How to convert to grayscale. If set to 'cv2', it follows opencv
        implementation. Else if set to 'equal', it sums up values along channel
        axis, then divides it by the number of expected channels.
    mask_tif : bool, default: False
        Instead of parsing every xml file to reconstruct mask image arrays, use
        pre-drawn mask tif files which should reside in the same folder as
        annotation xml files.

    Notes
    -----
    - Resolution of all images is (1000,1000)
    - gt is converted from annotation recorded in xml format
    - gt has dtype of torch.float64, converted from numpy.uint16, and it has
      value 'num_objects' * 255 because it is base-transformed
    - The origianl dataset provides annotation in xml format, which takes
      long time to parse and to reconstruct mask images dynamically during
      training. Drawing masks beforehand makes training much faster. Use
      ``mask_tif`` in that case.
    - When ``augmenters`` is provided, set the ``num_samples`` argument
      30x1000x1000 -> 16x30=480 patches. Thus, the default ``num_samples=720``
      (x1.5)
    - dtype of 'gt' is int16. However, to make batching easier, it will be
      casted to float32
    - Be careful about types of augmenters; avoid interpolation

    References
    ----------
    .. [1] N. Kumar, R. Verma, S. Sharma, S. Bhargava, A. Vahadane, and A.
       Sethi, “A Dataset and a Technique for Generalized Nuclear Segmentation
       for Computational Pathology,” IEEE Transactions on Medical Imaging, vol.
       36, no. 7, pp. 1550–1560, Jul. 2017, doi: 10.1109/TMI.2017.2677499.

    See Also
    --------
    MaskDataset : Super class
    Dataset : Base class
    DatasetInterface : Interface

    """
    # Dataset's acronym
    acronym = 'ComPath'
    # Hard code resolution to parse annotation (.xml)
    _resolution = (1000, 1000)

    def __init__(
        self,
        root_dir: str,
        *,
        output: str = 'both',
        transforms: Optional[albumentations.Compose] = None,
        num_samples: Optional[int] = None,
        grayscale: bool = False,
        grayscale_mode: Union[str, Sequence[float]] = 'cv2',
        # specific to this dataset
        mask_tif: bool = False,
        **kwargs
    ):
        self._root_dir = root_dir
        self._output = output
        self._transforms = transforms
        self._num_samples = num_samples
        self._grayscale = grayscale
        self._grayscale_mode = grayscale_mode
        # specific to this dataset
        self.mask_tif = mask_tif

    def get_image(self, p: Path) -> np.ndarray:
        img = Image.open(p)
        if img.mode == 'RGBA':
            img = img.convert(mode='RGB')
        return np.asarray(img)

    def get_mask(self, p: Path) -> np.ndarray:
        if self.mask_tif:
            if p.suffix == '.xml':
                raise ValueError(
                    "Use `save_xml_to_tif()` then set `mask_tif` to True"
                )
            mask = tifffile.imread(p)
            return mask.astype(np.int16)
        # Parse xml
        mask = self._parse_xml_to_array(p)
        return mask

    @classmethod
    def _parse_xml_to_array(cls, f_anno) -> np.ndarray:
        """This dataset provides annotation in .xml format

        Consider pre-generating mask image using ``save_xml_to_tif()``
        """
        tree = ET.parse(f_anno)
        root = tree.getroot()

        rr = []
        cc = []
        for region in root.iter('Region'):
            r = []
            c = []
            # print(region.attrib)
            # print(region.find('Vertices'))
            if (vertices := region.find('Vertices')) is not None:
                for v in vertices:
                    # print(v.attrib)
                    r.append(v.attrib['Y'])
                    c.append(v.attrib['X'])
                rr.append(np.array(r, dtype=np.float16))
                cc.append(np.array(c, dtype=np.float16))
        # X, Y = anno['X'], anno['Y']
        mask = np.zeros(cls._resolution, dtype=np.int16)
        for i, (x, y) in enumerate(zip(cc, rr), 1):
            r, c = polygon(y, x, shape=cls._resolution)
            if len(rr) == 0 and len(cc) == 0:
                continue
            mask[r, c] = i
        return mask
        # return {'X': cc, 'Y': rr}

    def save_xml_to_tif(self):
        """Parse .xml to mask and write it as tiff file

        Having masks in images is much faster than parsing .xml for each call.
        This func iterates through ``anno_dict``, parse and save each in .tif
        format in the same annotation directory. Re-initiate an instance with
        ``mask_tif`` argument to load them.
        """
        if self.output not in ['mask', 'both']:
            raise ValueError("Set output either to 'mask' or 'both'")
        for i, p in self.anno_dict.items():
            mask = self._parse_xml_to_array(p)
            fname = p.with_suffix('.tif')
            tifffile.imwrite(
                fname,
                data=mask,
                compression='zlib'
            )
            print(f"[{i}/{len(self.anno_dict) - 1}] Wrote '{fname}'")

    @cached_property
    def file_list(self) -> list:
        root_dir = self.root_dir
        parent = 'Tissue images'
        file_list = sorted(root_dir.glob(f'{parent}/*.png'))
        return file_list

    @cached_property
    def anno_dict(self) -> Dict[int, Path]:
        root_dir = self.root_dir
        parent = 'Annotations'
        ext = 'xml'
        if self.mask_tif:
            ext = 'tif'
        anno_dict = dict((k, v) for k, v in enumerate(
            sorted(root_dir.glob(f'{parent}/*.{ext}'))
            ))
        return anno_dict
