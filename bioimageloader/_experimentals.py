from typing import List, Dict
from .base import Dataset
from .collections import *
from .collections import __all__ as ALL_COLLECTIONS


ROOTS = {
    # Mask anno
    #   Instance
    'DSB2018'                 : '../Data/DSB2018',
    'ComputationalPathology'  : '../Data/ComputationalPathology',
    'FRUNet'                  : '../Data/FRU_processing',
    'S_BSST265'               : '../Data/BioStudies',
    'BBBC006'                 : '../Data/bbbc/006',
    'BBBC020'                 : '../Data/bbbc/020',
    'BBBC039'                 : '../Data/bbbc/039',
    #   Semantic (fg/bg)
    'TNBC'                    : '../Data/TNBC_NucleiSegmentation',
    'MurphyLab'               : '../Data/2009_ISBI_2DNuclei_code_data',
    'BBBC007'                 : '../Data/bbbc/007',
    'BBBC008'                 : '../Data/bbbc/008',
    'BBBC018'                 : '../Data/bbbc/018',
    # Partial mask anno
    'DigitalPathology'        : '../Data/DigitalPathology',
    'UCSB'                    : '../Data/UCSB_BioSegmentation',
    # no anno
    'BBBC002'                 : '../Data/bbbc/002',
    'BBBC013'                 : '../Data/bbbc/013',
    'BBBC014'                 : '../Data/bbbc/014',
    'BBBC015'                 : '../Data/bbbc/015',
    'BBBC016'                 : '../Data/bbbc/016',
    'BBBC026'                 : '../Data/bbbc/026',
    'BBBC041'                 : '../Data/bbbc/041/malaria',  # malaria
    'BBBC021'                 : '../Data/bbbc/021',  # huge number 132,000
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
