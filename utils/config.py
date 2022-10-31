

import yaml
import os
import shutil
from addict import Dict

class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no "
                                f"attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex

class Config:

    def __init__(self, cfg_dict=None, filename=None):
        super().__setattr__("_cfg_dict", cfg_dict)
        super().__setattr__("_filename", filename)

    @staticmethod
    def fromfile(filename: str):
        """
            Args:
                path: the path of *.yml
        """
        with open(filename, "r") as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)

        return Config(ConfigDict(cfg_dict), filename)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)
    
    def __setattr__(self, name, value):
        self._cfg_dict.__setattr__(name, value)


    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setitem__(self, name, value):
        return self._cfg_dict.__setitem__(name, value)

    def __len__(self):
        return len(self._cfg_dict)

    def __repr__(self):
        return f'Config (path: {self._filename}): {self._cfg_dict}'

    def dump(self, file=None):
        with open(file, 'w') as f:
            yaml.dump(dict(self._cfg_dict), f)

    @property
    def pretty_text(self):
        text = ""
        cfg = self._cfg_dict
        for k in cfg:
            text += f"{k}: {cfg[k]}\n"
        return text
