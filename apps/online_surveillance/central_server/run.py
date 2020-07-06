import os
import sys
import shutil
import logging
import argparse
from importlib import import_module

from structures.mtracker import NetworkMCTracker


def load_config(f):
    shutil.copy(os.path.abspath(os.path.expanduser(f)), os.path.join(os.getcwd(), '_temp_config.py'))
    module = import_module('_temp_config')
    cfg_dict = {
        k: v for k, v in module.__dict__.items() if k[0] != '_'
    }
    del sys.modules['_temp_config']
    os.remove(os.path.join(os.getcwd(), '_temp_config.py'))
    return cfg_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Network multi-camera tracker')
    parser.add_argument('config', type=str, help='Config file for multi-camera tracker')
    parser.add_argument('--port', type=int, default=5558, required=False,
                        help='TCP Port for sending global tracker state. Default: 5558.')
    args = parser.parse_args()

    cfg_dict = load_config(args.config)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)s:%(name)s: %(asctime)s %(message)s')

    mtracker = NetworkMCTracker(**cfg_dict)

    mtracker.run()
