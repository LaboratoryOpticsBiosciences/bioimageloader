from ._dsb2018 import DSB2018
from ._tnbc import TNBC
from ._compath import ComputationalPathology
from ._s_bsst265 import S_BSST265
from ._murphylab import MurphyLab
from ._bbbc006 import BBBC006
from ._bbbc007 import BBBC007
from ._bbbc008 import BBBC008
from ._bbbc018 import BBBC018
from ._bbbc020 import BBBC020
from ._bbbc039 import BBBC039
# partial anno
from ._digitpath import DigitalPathology
from ._ucsb import UCSB
from ._bbbc002 import BBBC002  # very few
# no anno
from ._bbbc013 import BBBC013  # very few
from ._bbbc014 import BBBC014


__all__ = [
    'DSB2018',
    'TNBC',
    'ComputationalPathology',
    'S_BSST265',
    'MurphyLab',
    'BBBC006',
    'BBBC007',
    'BBBC008',
    'BBBC018',
    'BBBC020',
    'BBBC039',
    'DigitalPathology',
    'UCSB',
    'BBBC002',
    'BBBC013',
    'BBBC014',
]
