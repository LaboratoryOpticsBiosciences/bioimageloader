from typing import List, Optional

import albumentations
import yaml

from .base import Dataset
from .collections import *


class Config(dict):
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


def datasets_from_cfg(
    config: Config,
    transforms: Optional[albumentations.Compose] = None,
) -> List[Dataset]:
    datasets: List[Dataset] = []
    for dataset, kwargs in config.items():
        exec(f'datasets.append({dataset}(transforms=transforms, **kwargs))')
    return datasets
