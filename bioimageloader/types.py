from pathlib import Path
from typing import Dict, List, TypeVar


T = TypeVar('T')

# Keep all ext in lower case
PIL_IMAGE_EXT = [
    '.png',
    '.jpg',
    '.jpeg',
]
TIFFFILE_IMAGE_EXT = [
    '.tif',
    '.tiff',
]
KNOWN_IMAGE_EXT = PIL_IMAGE_EXT + TIFFFILE_IMAGE_EXT
Bundled = List[T]
BundledPath = Bundled[Path]


class DatasetList(list):
    """List of Datasets
    """
    def __init__(self, x):
        for _x in x:
            self.append(_x)

    def set_training(self, val: bool):
        """Iterate config and set all ``training`` to given value

        It only affects those that have ``training`` kwarg.

        """
        attr = 'training'
        for dset in self:
            if hasattr(dset, attr):
                setattr(dset, attr, val)
                # clear cached_property
                if hasattr(dset, 'file_list'):
                    delattr(dset, 'file_list')
                if hasattr(dset, 'anno_dict'):
                    delattr(dset, 'anno_dict')

    def foreach_sample_by_indices(self, indices: Dict[str, List[int]]):
        """Iter Datasets and sample by indices

        Set attribute `_indices`

        Arguments
        ---------
        indices : dictionary
            Key is acronym and value is list of indices
        """
        from .utils import split_dataset_by_indices

        for i, dset in enumerate(self):
            if (name := dset.acronym) in indices:
                _indices = indices[name]
                self[i] = split_dataset_by_indices(dset, _indices)
                setattr(self[i], '_indices', _indices)

# def split_dataset_by_indices(
#     dataset: MaskDatasetProto,
#     indices: Sequence[int],
# ) -> List[MaskDataset]:
#     """Split dataset given indices
#     """
#     return subset(dataset, indices)
