from typing import List, Optional, Union, Dict

import albumentations
import yaml

from .base import Dataset
from .collections import *


class Config(dict):
    """Construct config from a yaml file

    Parameters
    ----------
    filename : str
        Path to config file
    """
    def __init__(self, filename):
        self.filename = filename
        cfg = Config.read_yaml(filename)
        for k, i in cfg.items():
            self[k] = i

    @staticmethod
    def read_yaml(filename: str) -> dict:
        with open(filename, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg


def datasets_from_config(
    config: Config,
    transforms: Optional[Union[albumentations.Compose, Dict[str, albumentations.Compose]]] = None,
) -> List[Dataset]:
    """Load multiple datasets from a yaml file

    Parameters
    ----------
    config : configuration object
        Config instance that contains acronyms and arguments to initialize
        each dataset
    transforms : albumentations.Compose or dictionary, optional
        Either apply a single composed transformations for every datasets or
        pass a dictionary that defines transformations for each dataset with
        keys being the acronyms of datsets.

    """
    datasets: List[Dataset] = []
    for dataset, kwargs in config.items():
        if isinstance(transforms, dict):
            exec(f'datasets.append({dataset}(transforms=transforms[dataset], **kwargs))')
        else:
            exec(f'datasets.append({dataset}(transforms=transforms, **kwargs))')
    return datasets
