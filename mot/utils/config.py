import os
import sys
import shutil
from typing import Dict
from importlib import import_module


class Config:
    keywords = ['tracker', 'detector', 'encoders', 'metric', 'matcher', 'predictor', 'secondary_matcher']

    def __init__(self, d: Dict = None):
        if d is not None:
            for k, v in d.items():
                if type(v) is dict:
                    v = Config(v)
                if type(v) is list:
                    v = [Config(item) for item in v]
                setattr(self, k, v)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if k[0] != '_' and k not in self.keywords}


def cfg_from_file(f: str) -> Config:
    shutil.copy(os.path.abspath(os.path.expanduser(f)), '_temp_config.py')
    module = import_module('_temp_config')
    cfg_dict = {
        k: v for k, v in module.__dict__.items()
    }
    cfg = Config(cfg_dict)
    del sys.modules['_temp_config']
    return cfg
