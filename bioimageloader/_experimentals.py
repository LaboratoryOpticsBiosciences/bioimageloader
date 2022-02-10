from typing import List, Dict
from .base import Dataset
from .collections import *
from .collections import __all__ as ALL_COLLECTIONS


ROOTS = {
    # anno
    'DSB2018'                 : '../Data/DSB2018',
    'TNBC'                    : '../Data/TNBC_NucleiSegmentation',
    'ComputationalPathology'  : '../Data/ComputationalPathology',
    'S_BSST265'               : '../Data/BioStudies',
    'MurphyLab'               : '../Data/2009_ISBI_2DNuclei_code_data',
    'BBBC006'                 : '../Data/bbbc/006',
    'BBBC007'                 : '../Data/bbbc/007',
    'BBBC008'                 : '../Data/bbbc/008',
    'BBBC018'                 : '../Data/bbbc/018',
    'BBBC020'                 : '../Data/bbbc/020',
    'BBBC039'                 : '../Data/bbbc/039',
    # partial anno
    'DigitalPathology'        : '../Data/DigitalPathology',
    'UCSB'                    : '../Data/UCSB_BioSegmentation',
    'BBBC002'                 : '../Data/bbbc/002',
    # no anno
    'BBBC013'                 : '../Data/bbbc/013',
    'BBBC014'                 : '../Data/bbbc/014',
    'BBBC015'                 : '../Data/bbbc/015',
    'BBBC016'                 : '../Data/bbbc/016',
    'BBBC026'                 : '../Data/bbbc/026',
    'BBBC041'                 : '../Data/bbbc/041',
    'FRUNet'                  : '../Data/FRU_processing',
    'BBBC021'                 : '../Data/bbbc/021',
}


def load_all_datasets(roots: Dict[str, str] = ROOTS) -> List[Dataset]:
    """Load all available datasets

    You may not want to use it. I use it to take samples from each one for
    visuaization.

    """
    datasets: List[Dataset] = []
    for dataset in ALL_COLLECTIONS:
        exec(f'datasets.append({dataset}("{roots[dataset]}"))')
    return datasets


def cycle_colors(
    cm,
    length: int
):
    """Cycle matplotlib's categorical color maps

    cm : matplotlib.colors.ListedColormap
        Colormap, ex) plt.cm.tab20
    length : int
        Length you want

    """
    unique = cm.colors
    len_unique = len(unique)
    if length <= len_unique:
        return unique[:length]
    colors = []
    cycle = -1
    for i in range(length):
        if i % len_unique == 0:
            cycle += 1
        if i >= len_unique:
            i -= len_unique * cycle
        colors.append(unique[i])
    return colors


def to_hex_color(x):
    """Ignore alpha"""
    if isinstance(x[0], float):
        x = [int(255*_x) for _x in x]
    return '#{:02X}{:02X}{:02X}'.format(x[0], x[1], x[2])
