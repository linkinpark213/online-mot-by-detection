import os
import sys
import shutil
from typing import Dict
from importlib import import_module


class Config:
    keywords = ['tracker', 'detector', 'encoders', 'metric', 'matcher', 'predictor', 'detection_filters',
                'secondary_matcher']

    def __init__(self, d: Dict, filepath: str):
        if d is not None:
            for k, v in d.items():
                if k == 'include':
                    d_ = cfg_from_file(os.path.join(os.path.dirname(os.path.abspath(filepath)), v))
                    for k_, v_ in d_.__dict__.items():
                        setattr(self, k_, v_)
                else:
                    if type(v) is dict:
                        if 'type' in v:
                            v = Config(v, filepath)
                        elif 'include' in v:
                            v = cfg_from_file(os.path.join(os.path.dirname(os.path.abspath(filepath)), v['include']))
                        else:
                            raise ValueError('Expected \'type\' or \'include\' in a config dict')
                    if type(v) is list:
                        v = [Config(item, filepath) for item in v]
                    setattr(self, k, v)

    def to_dict(self, ignore_keywords: bool = False):
        return {k: v for k, v in self.__dict__.items() if
                k[0] != '_' and (not ignore_keywords or k not in self.keywords)}


def cfg_from_file(f: str) -> Config:
    shutil.copy(os.path.abspath(os.path.expanduser(f)), '_temp_config.py')
    module = import_module('_temp_config')
    cfg_dict = {
        k: v for k, v in module.__dict__.items()
    }
    del sys.modules['_temp_config']
    return Config(cfg_dict, f)
