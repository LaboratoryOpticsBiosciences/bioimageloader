import os.path
import warnings
from functools import cached_property
from typing import Dict, List, Optional, Union

import albumentations
import yaml

from .base import Dataset
from .collections import *
from .types import DatasetList


class Config(dict):
    """Construct config from a yaml file

    Parameters
    ----------
    filename : str
        Path to config file
    """
    def __init__(self, filename=None):
        if filename is not None:
            self.filename = filename
            cfg = self._read_yaml()
            for k, v in cfg.items():
                self[k] = v

    def _read_yaml(self) -> dict:
        with open(self.filename, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg

    @staticmethod
    def from_dict(d: dict):
        return _Config(d)

    def load_datasets(
        self,
        transforms: Optional[Union[albumentations.Compose, Dict[str, albumentations.Compose]]] = None,
    ) -> DatasetList:
        """Load multiple datasets from a yaml file

        Note that when you provide a dictionray for ``transforms``, keys should
        be the class names, not their acronyms.

        Parameters
        ----------
        config : configuration object
            Config instance that contains acronyms and arguments to initialize
            each dataset
        transforms : albumentations.Compose or dictionary, optional
            Either apply a single composed transformations for every datasets or
            pass a dictionary that defines transformations for each dataset with
            keys being the class names of collections.

        """
        datasets: List[Dataset] = []
        for dataset, kwargs in self.items():
            if isinstance(transforms, dict):
                if dataset in transforms:
                    exec(f'datasets.append({dataset}(transforms=transforms[dataset], **kwargs))')
                else:
                    exec(f'datasets.append({dataset}(**kwargs))')
            else:
                exec(f'datasets.append({dataset}(transforms=transforms, **kwargs))')
        return DatasetList(datasets)

    @cached_property
    def commonpath(self):
        commonpath = os.path.commonpath([
            p for p in map(lambda x: x['root_dir'], self.values())
        ])
        return commonpath

    def replace_commonpath(self, new: str):
        """Replace common path for all ``root_dir`` with a new one

        All ``root_dir`` should have a commonpath. Do not put trailing '/' at
        the end.

        You made a config with root_dir being relative path. You do not need to
        replace them manually with this method.
        """
        for k in self.keys():
            p = self[k]['root_dir']
            self[k]['root_dir'] = p.replace(self.commonpath, new.rstrip('/'))
        delattr(self, 'commonpath')

    def set_training(self, val: bool):
        """Iterate config and set all ``training`` to given value

        It only affects those that have ``training`` kwarg.

        """
        attr = 'training'
        warnings.warn(f"This method only set those that have `{attr}` kwarg.",
                      stacklevel=2)
        for k in self.keys():
            if attr in self[k]:
                self[k][attr] = val

    def set_ouput(self, val: str):
        """Iterate config and set all ``output`` to given value

        It only affects those that have ``output`` kwarg.

        """
        attr = 'output'
        warnings.warn(f"This method only set those that have `{attr}` kwarg.",
                      stacklevel=2)
        for k in self.keys():
            if attr in self[k]:
                self[k][attr] = val

    def set_grayscale(self, val: bool):
        """Iterate config and set all ``grayscale`` to given value

        It only affects those that have ``grayscale`` kwarg.

        """
        attr = 'grayscale'
        warnings.warn(f"This method only set those that have `{attr}` kwarg.",
                      stacklevel=2)
        for k in self.keys():
            if attr in self[k]:
                self[k][attr] = val


class _Config(Config):
    """Config.from_dict()"""
    def __init__(self, d: dict):
        for k, v in d.items():
            self[k] = v
