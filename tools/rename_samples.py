import re
import os
import shutil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str, help='Path to the images in a camera.')
    parser.add_argument('cam', type=int, help='ID of the camera')
    args = parser.parse_args()

    pattern = re.compile('l([0-9]*)_g([-0-9]*).jpg')

    files = sorted(os.listdir(args.img_path))
    for filename in files:
        localID, globalID = pattern.match(filename).groups()
        if globalID != '-1':
            src_path = os.path.join(args.img_path, filename)
            dst_path = os.path.join(args.img_path, 'g{}_c{}_l{}.jpg'.format(globalID, args.cam, localID))
            print('Moving {} to {}'.format(src_path, dst_path))
            shutil.move(src_path, dst_path)
