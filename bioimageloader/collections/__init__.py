"""Collection of public bioimage datasets
"""

# import should not be sorted by isort
# #--- MaskDataset (mask anno) ---# #
#   full anno
#     instance
from ._dsb2018 import DSB2018
from ._compath import ComputationalPathology
from ._frunet import FRUNet
from ._s_bsst265 import S_BSST265
from ._bbbc006 import BBBC006
from ._bbbc020 import BBBC020
from ._bbbc039 import BBBC039
#     semantic (fg/bg)
from ._tnbc import TNBC
from ._bbbc008 import BBBC008
#     semantic (boundary, outline)
from ._murphylab import MurphyLab
from ._bbbc007 import BBBC007
from ._bbbc018 import BBBC018
#   partial anno
#     instance
#     semantic (fg/bg)
from ._digitpath import DigitalPathology
from ._ucsb import UCSB
# #--- Dataset (no anno) ---# #
from ._bbbc002 import BBBC002  # a few annotated
from ._bbbc013 import BBBC013
from ._bbbc014 import BBBC014
from ._bbbc015 import BBBC015
from ._bbbc016 import BBBC016
from ._bbbc026 import BBBC026
from ._bbbc041 import BBBC041
from ._bbbc021 import BBBC021  # huge dataset 132,000 images


# Keep this list sorted
__all__ = [
    'BBBC002',
    'BBBC006',
    'BBBC007',
    'BBBC008',
    'BBBC013',
    'BBBC014',
    'BBBC015',
    'BBBC016',
    'BBBC018',
    'BBBC020',
    'BBBC021',
    'BBBC026',
    'BBBC039',
    'BBBC041',
    'ComputationalPathology',
    'DSB2018',
    'DigitalPathology',
    'FRUNet',
    'MurphyLab',
    'S_BSST265',
    'TNBC',
    'UCSB',
]

assert __all__ == sorted(__all__), "Keep collections sorted"
