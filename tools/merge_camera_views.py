import cv2
import math
import argparse
import numpy as np


def merge(captures, writer, resolution):
    w, h = resolution
    n = 0
    nGrid = int(math.sqrt(len(captures))) + (1 if math.sqrt(len(captures)) - int(math.sqrt(len(captures))) > 0 else 0)
    gridW, gridH = w // nGrid, h // nGrid
    while True:
        n += 1
        print('Frame #{}'.format(n))
        image = np.zeros((resolution[1], resolution[0], 3)).astype(np.uint8)
        for i, capture in enumerate(captures):
            ret, frame = capture.read()
            if not ret:
                return
            frame = cv2.resize(frame, (gridW, gridH))
            image[
            gridH * (i // nGrid): gridH * (1 + i // nGrid),
            gridW * (i % nGrid): gridW * (1 + i % nGrid),
            :] = frame

        writer.write(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', type=int, required=False, default=30, help='Frames per second.')
    parser.add_argument('files', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    captures = []
    writer = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (1920, 1080))
    for file in args.files:
        captures.append(cv2.VideoCapture(file))

    merge(captures, writer, (1920, 1080))

    writer.release()
    print('Finished merging to out.mp4')
