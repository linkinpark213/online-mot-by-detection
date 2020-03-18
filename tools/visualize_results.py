import os
import cv2
import argparse
import numpy as np
import mot.utils.visualize as vis


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str)
    parser.add_argument("result_path", type=str)
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--output_path", type=str, default='videos/')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not os.path.isfile(args.video_path):
        raise FileNotFoundError('Video file path invalid: {}'.format(args.video_path))
    if not os.path.isfile(args.result_path):
        raise FileNotFoundError('Result file path invalid: {}'.format(args.result_path))
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    results = np.loadtxt(args.result_path, delimiter=',')
    results[:, 4] = results[:, 4] + results[:, 2]
    results[:, 5] = results[:, 5] + results[:, 3]
    capture = cv2.VideoCapture(args.video_path)
    writer = None
    frame_no = 0
    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        if writer is None:
            writer = cv2.VideoWriter(os.path.join(args.output_path, args.video_path.split('/')[-1]),
                                     cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

        boxes = results[np.where(results[:, 0] == frame_no)]

        frame = vis.snapshot_from_results(frame, boxes, frame_no)

        writer.write(frame)
        if args.display:
            cv2.imshow('Frame', frame)
            cv2.waitKey(1)

        frame_no += 1

    if writer is not None:
        writer.release()
