import os
import sys
import time
import shutil
import logging
import argparse
import traceback
from importlib import import_module

import mot.utils

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


def run(mtracker, args, **kwargs):
    writer = mot.utils.get_result_writer(args.save_result)

    while mtracker.running:
        try:
            # Update identity pool and gallery with a frequency.
            if time.time() - mtracker.last_update_time > mtracker.gallery_update_freq:
                mtracker.tick()
                for camID, localID, globalID, distance in mtracker.matches:
                    writer.write('{:d} {:d} {:d} {:.3f}\n'.format(camID, localID, globalID, distance))
                for camID, localID, globalID in mtracker.unmatched_tracklets:
                    writer.write('{:d} {:d} {:d} 1\n'.format(camID, localID, globalID))
                mtracker.last_update_time = time.time()
                mtracker.log()

        except:
            traceback.print_exc()
            mtracker.terminate()

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Network multi-camera tracker')
    parser.add_argument('config', type=str, help='Config file for multi-camera tracker')
    parser.add_argument('--port', type=int, default=5558, required=False,
                        help='TCP Port for sending global tracker state. Default: 5558.')
    parser.add_argument('--save-log', type=str, default='', required=False, help='Path to save log.')
    parser.add_argument('--save-result', type=str, default='', required=False, help='Path to save results.')
    args = parser.parse_args()

    cfg_dict = load_config(args.config)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)s:%(name)s: %(asctime)s %(message)s')

    if args.save_log != '':
        save_log_dir = os.path.abspath(os.path.dirname(args.save_log))
        if not os.path.isdir(save_log_dir):
            logging.warning('Result saving path {} doens\'t exist. Creating...'.format(save_log_dir))
            os.makedirs(save_log_dir)

        handler = logging.FileHandler(args.save_log, mode='w+')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s:%(name)s: %(asctime)s %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger('MTMCT')
        logger.addHandler(handler)

    mtracker = NetworkMCTracker(**cfg_dict)

    run(mtracker, args)
